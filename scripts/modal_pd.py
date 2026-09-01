import time
from pathlib import Path

import modal
import torch

from augur.config import QwenConfig
from augur.generation import FixedSlotDecoder
from augur.pd import DecodeService, PrefillService
from augur.sampling import sample_next_token
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

app = modal.App("augur-pd")
models = modal.Volume.from_name("augur-models", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("einops>=0.8", "packaging>=24.0", "regex>=2024.0", "safetensors>=0.4")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
)


def load_decoder() -> tuple[FixedSlotDecoder, Tokenizer]:
    model_dir = Path("/models/qwen2.5-0.5b")
    cfg = QwenConfig.from_pretrained(model_dir)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    weights = load_weights(
        model_dir / "model.safetensors",
        cfg,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    return FixedSlotDecoder(weights, cfg, max_slots=1, max_seq_len=8192), tokenizer


@app.cls(image=image, gpu="A10G", timeout=300, volumes={"/models": models})
class PrefillWorker:
    @modal.enter()
    def load(self) -> None:
        decoder, self.tokenizer = load_decoder()
        self.service = PrefillService(decoder)

    @modal.method()
    def prefill(self, prompt: str):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=torch.device("cuda"))
        with torch.no_grad():
            return self.service.prefill(input_ids)


@app.cls(image=image, gpu="A10G", timeout=300, volumes={"/models": models})
class DecodeWorker:
    @modal.enter()
    def load(self) -> None:
        decoder, self.tokenizer = load_decoder()
        self.service = DecodeService(decoder)

    @modal.method()
    def decode(self, result, max_new_tokens: int):
        with torch.no_grad():
            import_metrics = self.service.start(result)
            token_ids = [result.first_token]
            decode_seconds = 0.0
            while len(token_ids) < max_new_tokens and token_ids[-1] != self.tokenizer.eos_token_id:
                logits, metrics = self.service.decode(token_ids[-1])
                decode_seconds += metrics.decode_seconds
                token_ids.append(int(sample_next_token(logits[:, -1]).item()))
        return self.tokenizer.decode(token_ids).lstrip(), len(token_ids), import_metrics, decode_seconds


def run_once(prefill_worker, decode_worker, prompt: str, max_new_tokens: int) -> dict[str, float | int | str]:
    prefill_started_at = time.perf_counter()
    result, prefill_metrics = prefill_worker.prefill.remote(prompt)
    prefill_rpc_seconds = time.perf_counter() - prefill_started_at
    decode_started_at = time.perf_counter()
    text, generated_tokens, import_metrics, decode_seconds = decode_worker.decode.remote(result, max_new_tokens)
    decode_rpc_seconds = time.perf_counter() - decode_started_at
    return {
        "text": text,
        "generated_tokens": generated_tokens,
        "prefill_seconds": prefill_metrics.prefill_seconds,
        "export_seconds": prefill_metrics.export_seconds,
        "prefill_rpc_seconds": prefill_rpc_seconds,
        "import_seconds": import_metrics.import_seconds,
        "decode_seconds": decode_seconds,
        "decode_rpc_seconds": decode_rpc_seconds,
    }


@app.local_entrypoint()
def main(
    prompt: str = "What is the capital of France?",
    max_new_tokens: int = 32,
    warmup: int = 1,
    runs: int = 3,
) -> None:
    if warmup < 0 or runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")
    prefill_worker = PrefillWorker()
    decode_worker = DecodeWorker()
    for _ in range(warmup):
        run_once(prefill_worker, decode_worker, prompt, max_new_tokens)
    measurements = [run_once(prefill_worker, decode_worker, prompt, max_new_tokens) for _ in range(runs)]
    keys = [key for key in measurements[0] if key not in {"text", "generated_tokens"}]
    medians = {key: sorted(float(measurement[key]) for measurement in measurements)[runs // 2] for key in keys}
    generated_tokens = int(measurements[0]["generated_tokens"])
    print(
        {
            "text": measurements[0]["text"],
            "generated_tokens": generated_tokens,
            **medians,
            "decode_tokens_per_second": (generated_tokens - 1) / medians["decode_seconds"],
        }
    )
