from __future__ import annotations

import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import torch

StateDictPayload = Union[
    Mapping[str, torch.Tensor], Mapping[str, object], torch.nn.Module
]

_TORCH_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def torch_dtype(name: str) -> torch.dtype:
    if name not in _TORCH_DTYPE_NAMES:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return _TORCH_DTYPE_NAMES[name]


def load_state_dict(weights_path: Union[str, Path]) -> dict[str, torch.Tensor]:
    weights_path = Path(weights_path)
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        state_dict = load_safetensors_file(str(weights_path), device="cpu")
    elif weights_path.suffix == ".pkl":
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(payload)
    else:
        raise ValueError(f"Unsupported weights file: {weights_path}")

    return normalize_state_dict(state_dict)


def extract_state_dict(payload: StateDictPayload) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if _is_tensor_state_dict(payload):
            return typing.cast(dict[str, torch.Tensor], dict(payload))
        for key in ("state_dict", "model", "module", "model_state_dict"):
            if key in payload:
                nested_payload = payload[key]
                if isinstance(nested_payload, (Mapping, torch.nn.Module)):
                    return extract_state_dict(nested_payload)
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    raise ValueError("Could not extract a state_dict from the provided weights")


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {
                key.removeprefix(prefix): value
                for key, value in normalized.items()
            }
    return normalized


def _is_tensor_state_dict(payload: Mapping[str, object]) -> bool:
    return bool(payload) and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    )