from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NeedleModelConfiguation:
    vocabulary_size: int
    dimensions: int
    hidden_dimensions: int
    attention_heads: int
    kv_heads: int
    encoder_layers: int
    decoder_layers: int
    hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    pad_token_id: int
    decoder_start_token_id: int
    tie_word_embeddings: bool
    max_seq_len: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    dtype: Optional[str] = field(default=None)

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
