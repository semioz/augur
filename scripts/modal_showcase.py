from pathlib import Path

import modal
import torch

from augur.config import QwenConfig
from augur.server import AugurEngine
from augur.showcase import create_showcase_app
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

app = modal.App("augur-showcase")
models = modal.Volume.from_name("augur-models", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "einops>=0.8",
        "fastapi>=0.138.1",
        "packaging>=24.0",
        "regex>=2024.0",
        "safetensors>=0.4",
        "uvicorn>=0.49.0",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    max_containers=1,
    volumes={"/models": models},
)
@modal.concurrent(max_inputs=8)
class AugurShowcase:
    @modal.enter()
    def load(self) -> None:
        model_dir = Path("/models/qwen2.5-0.5b")
        checkpoint = model_dir / "model.safetensors"
        if not checkpoint.exists():
            raise RuntimeError(f"Missing model weights: {checkpoint}")
        device = torch.device("cuda")
        cfg = QwenConfig.from_pretrained(model_dir)
        tokenizer = Tokenizer.from_pretrained(model_dir)
        self.engine = AugurEngine(
            model_dir=model_dir,
            cfg=cfg,
            device=device,
            dtype=torch.float16,
            tokenizer=tokenizer,
            weights=load_weights(checkpoint, cfg, device=device, dtype=torch.float16),
        )

    @modal.asgi_app()
    def web(self):
        return create_showcase_app(self.engine)
