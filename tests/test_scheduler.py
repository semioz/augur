import asyncio

from augur.scheduler import (
    AsyncBatchScheduler,
    GenerationParams,
    GenerationRequest,
    RequestScheduler,
)


class FakeBatchGenerator:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def generate_batch(self, requests: list[GenerationRequest]) -> list[str]:
        self.batches.append([request.request_id for request in requests])
        return [f"output-{request.request_id}" for request in requests]


def make_request(
    request_id: str,
    prompt: str = "hello",
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=None,
        top_p=None,
        stop=[],
    )


def test_scheduler_tracks_waiting_requests_by_id() -> None:
    scheduler = RequestScheduler()

    scheduler.add_request(make_request("req-1"))
    scheduler.add_request(make_request("req-2"))

    assert scheduler.num_waiting == 2
    assert scheduler.get_request("req-1").prompt == "hello"


def test_scheduler_pops_waiting_requests_in_fcfs_order() -> None:
    scheduler = RequestScheduler()
    scheduler.add_request(make_request("req-1"))
    scheduler.add_request(make_request("req-2"))
    scheduler.add_request(make_request("req-3"))

    batch = scheduler.pop_batch(max_batch_size=2)

    assert [request.request_id for request in batch] == ["req-1", "req-2"]
    assert scheduler.num_waiting == 1
    assert scheduler.num_running == 2


def test_scheduler_cancels_queued_request_by_id() -> None:
    scheduler = RequestScheduler()
    scheduler.add_request(make_request("req-1"))
    scheduler.add_request(make_request("req-2"))
    scheduler.add_request(make_request("req-3"))

    cancelled = scheduler.cancel_request("req-2")
    batch = scheduler.pop_batch(max_batch_size=3)

    assert cancelled is True
    assert [request.request_id for request in batch] == ["req-1", "req-3"]
    assert scheduler.get_request("req-2") is None


def test_scheduler_returns_false_when_cancelling_unknown_request() -> None:
    scheduler = RequestScheduler()

    assert scheduler.cancel_request("missing") is False


def test_scheduler_batches_only_requests_with_matching_generation_params() -> None:
    scheduler = RequestScheduler()
    scheduler.add_request(make_request("req-1", max_new_tokens=8, temperature=0.0))
    scheduler.add_request(make_request("req-2", max_new_tokens=16, temperature=0.0))
    scheduler.add_request(make_request("req-3", max_new_tokens=8, temperature=0.0))

    batch = scheduler.pop_batch(max_batch_size=3)

    assert [request.request_id for request in batch] == ["req-1", "req-3"]
    assert scheduler.num_waiting == 1
    assert scheduler.peek_waiting().request_id == "req-2"


def test_generation_request_exposes_batching_params() -> None:
    request = make_request("req-1", max_new_tokens=8, temperature=0.7)

    assert request.params == GenerationParams(
        max_new_tokens=8,
        temperature=0.7,
        top_k=None,
        top_p=None,
    )


def test_scheduler_finishes_running_request() -> None:
    scheduler = RequestScheduler()
    scheduler.add_request(make_request("req-1"))
    scheduler.pop_batch(max_batch_size=1)

    finished = scheduler.finish_request("req-1")

    assert finished is True
    assert scheduler.num_running == 0
    assert scheduler.get_request("req-1") is None


def test_scheduler_returns_false_when_finishing_unknown_request() -> None:
    scheduler = RequestScheduler()

    assert scheduler.finish_request("missing") is False


def test_async_batch_scheduler_batches_compatible_requests() -> None:
    async def run() -> None:
        generator = FakeBatchGenerator()
        scheduler = AsyncBatchScheduler(
            generator,
            max_batch_size=4,
            batch_window_seconds=0.001,
        )
        scheduler.start()
        try:
            results = await asyncio.gather(
                scheduler.generate(make_request("req-1")),
                scheduler.generate(make_request("req-2")),
            )
        finally:
            await scheduler.shutdown()

        assert results == ["output-req-1", "output-req-2"]
        assert generator.batches == [["req-1", "req-2"]]

    asyncio.run(run())


def test_async_batch_scheduler_keeps_incompatible_requests_in_separate_batches() -> None:
    async def run() -> None:
        generator = FakeBatchGenerator()
        scheduler = AsyncBatchScheduler(
            generator,
            max_batch_size=4,
            batch_window_seconds=0.001,
        )
        scheduler.start()
        try:
            results = await asyncio.gather(
                scheduler.generate(make_request("req-1", max_new_tokens=8)),
                scheduler.generate(make_request("req-2", max_new_tokens=16)),
            )
        finally:
            await scheduler.shutdown()

        assert results == ["output-req-1", "output-req-2"]
        assert generator.batches == [["req-1"], ["req-2"]]

    asyncio.run(run())
