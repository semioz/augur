import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar


logger = logging.getLogger("uvicorn.error")

_T = TypeVar("_T")


class BatchGenerator(Protocol[_T]):
    def generate_batch(self, requests: list["GenerationRequest"]) -> list[_T]: ...


class ContinuousGenerator(Protocol):
    def prefill(self, states: list["ActiveRequest"]) -> list[int]: ...

    def decode(self, states: list["ActiveRequest"]) -> list[int]: ...

    def release(self, states: list["ActiveRequest"]) -> None: ...


@dataclass
class GenerationParams:
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    stop: list[str]

    @property
    def params(self) -> GenerationParams:
        return GenerationParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


@dataclass
class ActiveRequest:
    request: GenerationRequest
    slot: int
    token_queue: asyncio.Queue[int | None]
    generated_token_ids: list[int] = field(default_factory=list)
    pending_token: int | None = None


class RequestScheduler:
    def __init__(self) -> None:
        self._requests: dict[str, GenerationRequest] = {}
        self._waiting: deque[str] = deque()
        self._running: set[str] = set()

    @property
    def num_waiting(self) -> int:
        return sum(1 for request_id in self._waiting if request_id in self._requests)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def add_request(self, request: GenerationRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._requests[request.request_id] = request
        self._waiting.append(request.request_id)

    def get_request(self, request_id: str) -> GenerationRequest | None:
        return self._requests.get(request_id)

    def peek_waiting(self) -> GenerationRequest:
        for request_id in self._waiting:
            request = self._requests.get(request_id)
            if request is not None:
                return request
        raise IndexError("peek from an empty queue")

    def cancel_request(self, request_id: str) -> bool:
        return self._remove_request(request_id)

    def finish_request(self, request_id: str) -> bool:
        return self._remove_request(request_id)

    def _remove_request(self, request_id: str) -> bool:
        request = self._requests.pop(request_id, None)
        if request is None:
            return False
        self._running.discard(request_id)
        return True

    def pop_batch(self, max_batch_size: int) -> list[GenerationRequest]:
        if self.num_waiting == 0:
            return []
        return self.pop_matching(max_batch_size, self.peek_waiting().params)

    def pop_matching(
        self,
        max_batch_size: int,
        params: GenerationParams,
    ) -> list[GenerationRequest]:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        batch = []
        remaining = deque()
        while self._waiting:
            request_id = self._waiting.popleft()
            request = self._requests.get(request_id)
            if request is None:
                continue
            if len(batch) == max_batch_size or request.params != params:
                remaining.append(request_id)
                continue
            self._running.add(request_id)
            batch.append(request)
        self._waiting = remaining
        return batch


class AsyncContinuousScheduler:
    def __init__(
        self,
        generator: ContinuousGenerator,
        *,
        max_slots: int = 8,
        eos_token_id: int | None = None,
    ) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self._generator = generator
        self._max_slots = max_slots
        self._eos_token_id = eos_token_id
        self._scheduler = RequestScheduler()
        self._active: dict[str, ActiveRequest] = {}
        self._queues: dict[str, asyncio.Queue[int | None]] = {}
        self._cancelled: set[str] = set()
        self._ready = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        self._closed = True
        self._ready.set()
        if self._worker is not None:
            await self._worker
            self._worker = None

    async def stream(self, request: GenerationRequest) -> AsyncIterator[int]:
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        self.start()
        queue: asyncio.Queue[int | None] = asyncio.Queue()
        self._scheduler.add_request(request)
        self._queues[request.request_id] = queue
        self._ready.set()
        try:
            while (token := await queue.get()) is not None:
                yield token
        finally:
            self.cancel(request.request_id)

    def cancel(self, request_id: str) -> bool:
        if self._scheduler.get_request(request_id) is None:
            return False
        self._cancelled.add(request_id)
        self._ready.set()
        return True

    async def _run(self) -> None:
        while not self._closed:
            if not self._active and self._scheduler.num_waiting == 0:
                self._ready.clear()
                await self._ready.wait()
                continue

            self._finish_cancelled()
            decodable = [state for state in self._active.values() if state.pending_token is not None]
            if decodable:
                self._publish(decodable, await asyncio.to_thread(self._generator.decode, decodable))
            await self._admit()
            await asyncio.sleep(0)

        self._finish(list(self._active.values()))

    async def _admit(self) -> None:
        capacity = self._max_slots - len(self._active)
        if capacity <= 0 or self._scheduler.num_waiting == 0:
            return
        if self._active:
            params = next(iter(self._active.values())).request.params
            requests = self._scheduler.pop_matching(capacity, params)
        else:
            requests = self._scheduler.pop_batch(capacity)
        if not requests:
            return
        used_slots = {state.slot for state in self._active.values()}
        states = [
            ActiveRequest(request, slot, self._queues[request.request_id])
            for request, slot in zip(requests, (slot for slot in range(self._max_slots) if slot not in used_slots))
        ]
        self._active.update({state.request.request_id: state for state in states})
        self._publish(states, await asyncio.to_thread(self._generator.prefill, states))

    def _publish(self, states: list[ActiveRequest], tokens: list[int]) -> None:
        if len(states) != len(tokens):
            raise RuntimeError("continuous generator returned the wrong number of tokens")
        finished = []
        for state, token in zip(states, tokens):
            state.generated_token_ids.append(token)
            state.pending_token = token
            self._queues[state.request.request_id].put_nowait(token)
            if token == self._eos_token_id or len(state.generated_token_ids) >= state.request.max_new_tokens:
                finished.append(state)
        self._finish(finished)

    def _finish_cancelled(self) -> None:
        self._finish([state for request_id, state in self._active.items() if request_id in self._cancelled])
        for request_id in self._cancelled.copy():
            if self._scheduler.cancel_request(request_id):
                queue = self._queues.pop(request_id, None)
                if queue is not None:
                    queue.put_nowait(None)
            self._cancelled.discard(request_id)

    def _finish(self, states: list[ActiveRequest]) -> None:
        if not states:
            return
        self._generator.release(states)
        for state in states:
            request_id = state.request.request_id
            self._active.pop(request_id, None)
            self._scheduler.finish_request(request_id)
            queue = self._queues.pop(request_id, None)
            if queue is not None:
                queue.put_nowait(None)


class AsyncBatchScheduler(Generic[_T]):
    def __init__(
        self,
        generator: BatchGenerator[_T],
        max_batch_size: int = 8,
        batch_window_seconds: float = 0.005,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if batch_window_seconds < 0:
            raise ValueError("batch_window_seconds must be non-negative")

        self._generator = generator
        self._max_batch_size = max_batch_size
        self._batch_window_seconds = batch_window_seconds
        self._scheduler = RequestScheduler()
        self._futures: dict[str, asyncio.Future[_T]] = {}
        self._ready = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def shutdown(self) -> None:
        self._closed = True
        self._ready.set()
        if self._worker is not None:
            await self._worker
            self._worker = None

    async def generate(self, request: GenerationRequest) -> _T:
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        if self._worker is None:
            self.start()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[_T] = loop.create_future()
        self._scheduler.add_request(request)
        self._futures[request.request_id] = future
        self._ready.set()
        try:
            return await future
        except asyncio.CancelledError:
            self._scheduler.cancel_request(request.request_id)
            if not future.done():
                future.cancel()
            raise

    async def _run(self) -> None:
        while True:
            await self._ready.wait()
            self._ready.clear()
            if self._closed:
                break
            if self._batch_window_seconds > 0:
                await asyncio.sleep(self._batch_window_seconds)

            while self._scheduler.num_waiting > 0:
                batch = self._scheduler.pop_batch(self._max_batch_size)
                if not batch:
                    break
                await self._process_batch(batch)

    async def _process_batch(self, batch: list[GenerationRequest]) -> None:
        logger.info(
            "scheduler batch size=%d request_ids=%s",
            len(batch),
            [request.request_id for request in batch],
        )
        try:
            results = await asyncio.to_thread(self._generator.generate_batch, batch)
            if len(results) != len(batch):
                raise RuntimeError("batch generator returned the wrong number of results")
        except Exception as exc:
            for request in batch:
                future = self._futures.pop(request.request_id, None)
                self._scheduler.finish_request(request.request_id)
                if future is not None and not future.done():
                    future.set_exception(exc)
            return

        for request, result in zip(batch, results):
            future = self._futures.pop(request.request_id, None)
            self._scheduler.finish_request(request.request_id)
            if future is not None and not future.done():
                future.set_result(result)
