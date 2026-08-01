from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheLayout(str, Enum):
    EXPLICIT = "explicit"
    PER_LAYER_STATE = "per-layer-state"


class CrossAttentionStorage(str, Enum):
    INPUTS = "inputs"
    STATE = "state"


class CacheOutput(str, Enum):
    FULL = "full"
    DELTA = "delta"
    NONE = "none"


class AttentionImplementation(str, Enum):
    NATIVE = "native"
    DECOMPOSED = "decomposed"


class ActiveCacheStrategy(str, Enum):
    STATIC = "static"
    ADAPTIVE_RUNTIME = "adaptive-runtime"
    DYNAMIC_PREFIX = "dynamic-prefix"


@dataclass(frozen=True, init=False)
class DecoderExportStrategy:
    cache_layout: CacheLayout
    cross_attention_storage: CrossAttentionStorage
    cache_output: CacheOutput
    attention: AttentionImplementation
    active_cache: ActiveCacheStrategy
    use_native_gqa: bool

    def __init__(self) -> None:
        raise TypeError("Use a named DecoderExportStrategy constructor")

    @classmethod
    def reference(
        cls,
        *,
        attention: AttentionImplementation = AttentionImplementation.NATIVE,
        use_native_gqa: bool = True,
    ) -> DecoderExportStrategy:
        return cls._create(
            cache_layout=CacheLayout.EXPLICIT,
            cross_attention_storage=CrossAttentionStorage.INPUTS,
            cache_output=CacheOutput.FULL,
            attention=attention,
            active_cache=ActiveCacheStrategy.STATIC,
            use_native_gqa=use_native_gqa,
        )

    @classmethod
    def onnx(cls) -> DecoderExportStrategy:
        return cls._create(
            cache_layout=CacheLayout.EXPLICIT,
            cross_attention_storage=CrossAttentionStorage.INPUTS,
            cache_output=CacheOutput.DELTA,
            attention=AttentionImplementation.NATIVE,
            active_cache=ActiveCacheStrategy.ADAPTIVE_RUNTIME,
            use_native_gqa=True,
        )

    @classmethod
    def coreml(
        cls,
        *,
        attention: AttentionImplementation = AttentionImplementation.DECOMPOSED,
        dynamic_cache: bool = False,
    ) -> DecoderExportStrategy:
        return cls._create(
            cache_layout=CacheLayout.PER_LAYER_STATE,
            cross_attention_storage=CrossAttentionStorage.STATE,
            cache_output=CacheOutput.NONE,
            attention=attention,
            active_cache=(
                ActiveCacheStrategy.DYNAMIC_PREFIX
                if dynamic_cache
                else ActiveCacheStrategy.STATIC
            ),
            use_native_gqa=False,
        )

    @classmethod
    def coreai(cls) -> DecoderExportStrategy:
        return cls._create(
            cache_layout=CacheLayout.PER_LAYER_STATE,
            cross_attention_storage=CrossAttentionStorage.INPUTS,
            cache_output=CacheOutput.NONE,
            attention=AttentionImplementation.NATIVE,
            active_cache=ActiveCacheStrategy.STATIC,
            use_native_gqa=True,
        )

    @classmethod
    def _create(
        cls,
        *,
        cache_layout: CacheLayout,
        cross_attention_storage: CrossAttentionStorage,
        cache_output: CacheOutput,
        attention: AttentionImplementation,
        active_cache: ActiveCacheStrategy,
        use_native_gqa: bool,
    ) -> DecoderExportStrategy:
        strategy = object.__new__(cls)
        object.__setattr__(strategy, "cache_layout", cache_layout)
        object.__setattr__(
            strategy,
            "cross_attention_storage",
            cross_attention_storage,
        )
        object.__setattr__(strategy, "cache_output", cache_output)
        object.__setattr__(strategy, "attention", attention)
        object.__setattr__(strategy, "active_cache", active_cache)
        object.__setattr__(strategy, "use_native_gqa", use_native_gqa)
        return strategy

    @property
    def stateful(self) -> bool:
        return self.cache_layout == CacheLayout.PER_LAYER_STATE

    @property
    def cross_attention_cache_states(self) -> bool:
        return self.cross_attention_storage == CrossAttentionStorage.STATE

    @property
    def dynamic_cache(self) -> bool:
        return self.active_cache != ActiveCacheStrategy.STATIC

    @property
    def returns_cache_deltas(self) -> bool:
        return self.cache_output == CacheOutput.DELTA

    @property
    def uses_native_attention(self) -> bool:
        return self.attention == AttentionImplementation.NATIVE
