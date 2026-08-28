import subprocess
from pathlib import Path

import modal

from augur.cli import main as augur_main
from augur.modal_runner import ModalRunConfig, build_bench_args, parse_runner_args

app = modal.App("augur-gpu")
models = modal.Volume.from_name("augur-models", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "einops>=0.8",
        "fastapi>=0.138.1",
        "packaging>=24.0",
        "regex>=2024.0",
        "safetensors>=0.4",
        "triton>=3.0",
        "uvicorn>=0.49.0",
    )
    .env({"PYTHONPATH": "/root/src"})
)
image = base_image.add_local_dir("src", "/root/src").add_local_dir("scripts", "/root/scripts")
vllm_image = (
    base_image.pip_install("vllm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("scripts", "/root/scripts")
)


def run_remote(config: ModalRunConfig) -> None:
    model_path = Path("/models") / config.model_dir
    if not (model_path / "model.safetensors").exists():
        subprocess.run(
            [
                "python",
                "/root/scripts/download_weights.py",
                "--model",
                config.model,
                "--dest",
                str(model_path),
            ],
            check=True,
        )
        models.commit()
    if config.profile:
        subprocess.run(
            [
                "python",
                "/root/scripts/profile_decode.py",
                "--model-dir",
                str(model_path),
                "--prompt",
                config.prompt,
                "--max-new-tokens",
                str(config.max_new_tokens),
                "--dtype",
                config.dtype,
            ],
            check=True,
        )
        return
    for run in range(config.warmup):
        print(f"warmup {run + 1}/{config.warmup}")
        augur_main(build_bench_args(config))
    for run in range(config.runs):
        print(f"measurement {run + 1}/{config.runs}")
        augur_main(build_bench_args(config))


def make_gpu_function(config: ModalRunConfig):
    return app.function(
        image=vllm_image if config.engine == "vllm" else image,
        gpu=config.gpu,
        timeout=config.timeout,
        volumes={"/models": models},
    )(run_remote)


def main() -> None:
    config = parse_runner_args()
    print(f"using engine={config.engine} gpu={config.gpu}")
    remote_run = make_gpu_function(config)
    with modal.enable_output(), app.run():
        remote_run.remote(config)


if __name__ == "__main__":
    main()
