from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class NeedleModelConfiguation:
    vocabulary_size: int = 8192
    dimensions: int = 512
    hidden_dimensions: int = 512
    attention_heads: int = 8
    kv_heads: int = 4
    encoder_layers: int = 12
    decoder_layers: int = 8
    hidden_layers: int = 8
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    decoder_start_token_id: int = 1
    tie_word_embeddings: bool = True
    max_seq_len: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    dtype: Optional[str] = field(default="bfloat16")

    @property
    def resolved_dtype(self) -> str:
        return self.dtype or "bfloat16"

    @property
    def attention_head_dimensions(self) -> int:
        return self.dimensions // self.attention_heads

    @property
    def kv_dimensions(self) -> int:
        return self.kv_heads * self.attention_head_dimensions

    @property
    def encoder_max_length(self) -> int:
        return self.max_seq_len or self.max_position_embeddings or 1024

    @classmethod
    def from_file(cls, configuration_path: str | Path) -> "NeedleModelConfiguation":
        return cls.from_json_object(_load_json_object(configuration_path))

    @classmethod
    def from_json_object(cls, data: JSONObject) -> "NeedleModelConfiguation":
        defaults = cls()
        return cls(
            vocabulary_size=_int_value(
                data,
                "vocab_size",
                "vocabulary_size",
                default=defaults.vocabulary_size,
            ),
            dimensions=_int_value(
                data,
                "d_model",
                "hidden_size",
                "dimensions",
                default=defaults.dimensions,
            ),
            hidden_dimensions=_int_value(
                data,
                "hidden_size",
                "d_model",
                "hidden_dimensions",
                default=defaults.hidden_dimensions,
            ),
            attention_heads=_int_value(
                data,
                "num_attention_heads",
                "num_heads",
                "attention_heads",
                default=defaults.attention_heads,
            ),
            kv_heads=_int_value(
                data,
                "num_kv_heads",
                "num_key_value_heads",
                "kv_heads",
                default=defaults.kv_heads,
            ),
            encoder_layers=_int_value(
                data,
                "num_encoder_layers",
                "encoder_layers",
                default=defaults.encoder_layers,
            ),
            decoder_layers=_int_value(
                data,
                "num_decoder_layers",
                "decoder_layers",
                default=defaults.decoder_layers,
            ),
            hidden_layers=_int_value(
                data,
                "num_hidden_layers",
                "hidden_layers",
                default=defaults.hidden_layers,
            ),
            rope_theta=_float_value(data, "rope_theta", default=defaults.rope_theta),
            rms_norm_eps=_float_value(
                data, "rms_norm_eps", default=defaults.rms_norm_eps
            ),
            pad_token_id=_int_value(
                data, "pad_token_id", default=defaults.pad_token_id
            ),
            decoder_start_token_id=_int_value(
                data,
                "decoder_start_token_id",
                default=defaults.decoder_start_token_id,
            ),
            tie_word_embeddings=_bool_value(
                data,
                "tie_word_embeddings",
                default=defaults.tie_word_embeddings,
            ),
            max_seq_len=_optional_int_value(data, "max_seq_len"),
            max_position_embeddings=_optional_int_value(data, "max_position_embeddings"),
            dtype=_optional_str_value(
                data, "dtype", "torch_dtype", default=defaults.dtype
            ),
        )


def _load_json_object(path: str | Path) -> JSONObject:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a JSON object: {path}")
    return payload


def _config_value(data: JSONObject, *keys: str, default: JSONValue) -> JSONValue:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _scalar_value(data: JSONObject, *keys: str, default: JSONScalar) -> JSONScalar:
    value = _config_value(data, *keys, default=default)
    if isinstance(value, (dict, list)):
        raise ValueError(f"Expected scalar config value for keys {keys}, got {value!r}")
    return value


def _int_value(data: JSONObject, *keys: str, default: int) -> int:
    value = _scalar_value(data, *keys, default=default)
    return int(value) if value is not None else default


def _float_value(data: JSONObject, *keys: str, default: float) -> float:
    value = _scalar_value(data, *keys, default=default)
    return float(value) if value is not None else default


def _bool_value(data: JSONObject, *keys: str, default: bool) -> bool:
    value = _scalar_value(data, *keys, default=default)
    return bool(value) if value is not None else default


def _optional_int_value(data: JSONObject, key: str) -> int | None:
    value = _scalar_value(data, key, default=None)
    return None if value is None else int(value)


def _optional_str_value(
    data: JSONObject, *keys: str, default: str | None
) -> str | None:
    value = _scalar_value(data, *keys, default=default)
    return None if value is None else str(value)
