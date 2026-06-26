import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


logger = logging.getLogger("uvicorn.error")

_T = TypeVar("_T")


class BatchGenerator(Protocol[_T]):
    def generate_batch(self, requests: list["GenerationRequest"]) -> list[_T]: ...


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
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        first_request = self.peek_waiting() if self.num_waiting > 0 else None
        if first_request is None:
            return []

        batch = []
        remaining = deque()
        target_params = first_request.params
        while self._waiting:
            request_id = self._waiting.popleft()
            request = self._requests.get(request_id)
            if request is None:
                continue
            if len(batch) == max_batch_size or request.params != target_params:
                remaining.append(request_id)
                continue
            self._running.add(request_id)
            batch.append(request)
        self._waiting = remaining
        return batch


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
