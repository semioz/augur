from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor

from augur.generation import FixedSlotDecoder, PrefillResult
from augur.sampling import sample_next_token


@dataclass(frozen=True)
class PrefillMetrics:
    prefill_seconds: float
    export_seconds: float


@dataclass(frozen=True)
class DecodeMetrics:
    import_seconds: float
    decode_seconds: float


class PrefillService:
    def __init__(self, decoder: FixedSlotDecoder) -> None:
        self.decoder = decoder

    def prefill(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> tuple[PrefillResult, PrefillMetrics]:
        if input_ids.shape[0] != 1:
            raise ValueError("PD prefill currently supports batch size 1")
        started_at = perf_counter()
        logits = self.decoder.prefill(input_ids, torch.tensor([0], device=input_ids.device), attention_mask)
        prefill_seconds = perf_counter() - started_at
        if attention_mask is None:
            next_logits = logits[:, -1]
        else:
            position = attention_mask.to(torch.long).sum(dim=1) - 1
            next_logits = logits[torch.arange(logits.shape[0], device=logits.device), position]
        first_token = int(sample_next_token(next_logits, temperature, top_k, top_p).item())
        started_at = perf_counter()
        result = PrefillResult(
            snapshot=self.decoder.export_slot(0),
            first_token=first_token,
            cache_shape=(self.decoder.cache.keys.shape[0], self.decoder.cache.keys.shape[2], self.decoder.cache.keys.shape[4]),
            dtype=self.decoder.cache.keys.dtype,
        )
        return result, PrefillMetrics(prefill_seconds, perf_counter() - started_at)


class DecodeService:
    def __init__(self, decoder: FixedSlotDecoder) -> None:
        self.decoder = decoder

    def start(self, result: PrefillResult, *, slot: int = 0) -> DecodeMetrics:
        started_at = perf_counter()
        self.decoder.import_prefill_result(slot=slot, result=result)
        return DecodeMetrics(import_seconds=perf_counter() - started_at, decode_seconds=0.0)

    def decode(self, token_id: int, *, slot: int = 0) -> tuple[Tensor, DecodeMetrics]:
        started_at = perf_counter()
        logits = self.decoder.decode(torch.tensor([[token_id]], device=self.decoder.cache.keys.device), torch.tensor([slot], device=self.decoder.cache.keys.device))
        return logits, DecodeMetrics(import_seconds=0.0, decode_seconds=perf_counter() - started_at)
