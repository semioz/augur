from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class KVCache:
    keys: list[Tensor | None]
    values: list[Tensor | None]

def new_kv_cache(num_layers: int) -> KVCache:
    return KVCache(
        keys=[None] * num_layers,
        values=[None] * num_layers,
    )

def append_kv(cache: KVCache, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    cached_key = cache.keys[layer_idx]
    cached_value = cache.values[layer_idx]

    if cached_key is None:
        cache.keys[layer_idx] = key
        cache.values[layer_idx] = value
    else:
        if cached_value is None:
            raise ValueError(f"cache value missing for layer {layer_idx}")
        cache.keys[layer_idx] = torch.cat((cached_key, key), dim=2)
        cache.values[layer_idx] = torch.cat((cached_value, value), dim=2)

    final_key = cache.keys[layer_idx]
    final_value = cache.values[layer_idx]
    if final_key is None or final_value is None:
        raise ValueError(f"cache missing for layer {layer_idx}")
    return final_key, final_value
