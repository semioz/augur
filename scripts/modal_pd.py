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


def load_decoder(device: torch.device) -> tuple[FixedSlotDecoder, Tokenizer]:
    model_dir = Path("/models/qwen2.5-0.5b")
    cfg = QwenConfig.from_pretrained(model_dir)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    weights = load_weights(model_dir / "model.safetensors", cfg, device=device, dtype=torch.float16)
    return FixedSlotDecoder(weights, cfg, max_slots=1, max_seq_len=8192), tokenizer


@app.cls(image=image, gpu="A10G:2", timeout=300, volumes={"/models": models})
class PDWorker:
    @modal.enter()
    def load(self) -> None:
        prefill_decoder, self.tokenizer = load_decoder(torch.device("cuda:0"))
        decode_decoder, _ = load_decoder(torch.device("cuda:1"))
        self.peer_access = torch.cuda.can_device_access_peer(0, 1)
        self.prefill = PrefillService(prefill_decoder, export_device=torch.device("cuda:1"))
        self.decode = DecodeService(decode_decoder)

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int, prompt_tokens: int = 0):
        if prompt_tokens > 0:
            token_id = self.tokenizer.encode(" hello")[0]
            input_ids = torch.full((1, prompt_tokens), token_id, device=torch.device("cuda:0"))
        else:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=torch.device("cuda:0"))
        with torch.no_grad():
            result, prefill_metrics = self.prefill.prefill(input_ids)
            import_metrics = self.decode.start(result)
            token_ids = [result.first_token]
            decode_seconds = 0.0
            while len(token_ids) < max_new_tokens and token_ids[-1] != self.tokenizer.eos_token_id:
                logits, metrics = self.decode.decode(token_ids[-1])
                decode_seconds += metrics.decode_seconds
                token_ids.append(int(sample_next_token(logits[:, -1]).item()))
        return {
            "text": self.tokenizer.decode(token_ids).lstrip(),
            "generated_tokens": len(token_ids),
            "peer_access": self.peer_access,
            "prefill_seconds": prefill_metrics.prefill_seconds,
            "handoff_seconds": prefill_metrics.export_seconds,
            "import_seconds": import_metrics.import_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (len(token_ids) - 1) / decode_seconds,
        }


@app.local_entrypoint()
def main(
    prompt: str = "What is the capital of France?",
    max_new_tokens: int = 32,
    warmup: int = 1,
    runs: int = 3,
    prompt_tokens: int = 0,
) -> None:
    if warmup < 0 or runs <= 0 or not 0 <= prompt_tokens <= 8192:
        raise ValueError("invalid warmup, runs, or prompt_tokens")
    worker = PDWorker()
    for _ in range(warmup):
        worker.generate.remote(prompt, max_new_tokens, prompt_tokens)
    measurements = [worker.generate.remote(prompt, max_new_tokens, prompt_tokens) for _ in range(runs)]
    keys = [key for key in measurements[0] if key not in {"text", "peer_access", "generated_tokens"}]
    medians = {key: sorted(float(measurement[key]) for measurement in measurements)[runs // 2] for key in keys}
    print(
        {
            "text": measurements[0]["text"],
            "generated_tokens": measurements[0]["generated_tokens"],
            "peer_access": measurements[0]["peer_access"],
            **medians,
        }
    )
