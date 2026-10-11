"""Measure an offline NVFP4 KV snapshot against Augur's FP16 decode."""

import argparse
from pathlib import Path

import torch
from augur.config import QwenConfig
from augur.kv_cache import KVCache, new_kv_cache
from augur.model import model
from augur.nvfp4 import pack_nvfp4, unpack_nvfp4
from augur.tokenizer import Tokenizer
from augur.weights import load_weights
from scripts.compare_precision import error_metrics


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/qwen2.5-0.5b"))
    parser.add_argument("--prompt-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.prompt_tokens < 1:
        parser.error("--prompt-tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = QwenConfig.from_pretrained(args.model_dir)
    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=torch.float16)
    prompt = tokenizer.encode("Explain why the sky appears blue. " * (args.prompt_tokens + 1))
    ids = torch.tensor([prompt[: args.prompt_tokens]], device=device)
    cache = new_kv_cache(cfg, 1, ids.shape[1] + 1, device, torch.float16)
    first_logits = model(ids, weights, cfg, cache=cache)
    next_token = first_logits[:, -1].argmax(dim=-1, keepdim=True)

    active_k = cache.keys[..., : cache.seq_len, :]
    active_v = cache.values[..., : cache.seq_len, :]
    packed_k = pack_nvfp4(active_k)
    packed_v = pack_nvfp4(active_v)
    fp16_bytes = (active_k.numel() + active_v.numel()) * active_k.element_size()
    packed_bytes = sum(
        data.numel() + scales.numel() + global_scale.element_size()
        for data, scales, global_scale in (packed_k, packed_v)
    )

    decoded_k = unpack_nvfp4(*packed_k, torch.float16)
    decoded_v = unpack_nvfp4(*packed_v, torch.float16)
    plain = KVCache(cache.keys.clone(), cache.values.clone(), cache.seq_lens.clone(), cache.seq_len)
    compressed = KVCache(cache.keys.clone(), cache.values.clone(), cache.seq_lens.clone(), cache.seq_len)
    compressed.keys[..., : cache.seq_len, :] = decoded_k
    compressed.values[..., : cache.seq_len, :] = decoded_v
    reference = model(next_token, weights, cfg, cache=cache)[:, -1]
    plain_logits = model(next_token, weights, cfg, cache=plain)[:, -1]
    actual = model(next_token, weights, cfg, cache=compressed)[:, -1]
    mae, max_abs, relative_l2 = error_metrics(actual, reference)

    print(f"device: {device}; prompt tokens: {ids.shape[1]}; one decode step")
    print(f"active FP16 KV: {fp16_bytes / 1024**2:.2f} MiB")
    print(f"packed NVFP4 KV: {packed_bytes / 1024**2:.2f} MiB ({fp16_bytes / packed_bytes:.2f}x smaller)")
    print(f"plain-copy logit max diff: {(reference - plain_logits).abs().max().item():.6f}")
    print(f"KV relative L2: K={error_metrics(decoded_k, active_k)[2]:.6f}, V={error_metrics(decoded_v, active_v)[2]:.6f}")
    print(f"logit MAE: {mae:.6f}; max abs: {max_abs:.6f}; relative L2: {relative_l2:.6f}")
    print(f"next token: FP16={reference.argmax().item()} NVFP4={actual.argmax().item()}")
    print("Offline snapshot only: decode uses a staged FP16 cache; no runtime memory/speed savings.")


if __name__ == "__main__":
    main()
