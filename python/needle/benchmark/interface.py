from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import DTypeLike

DTypeSpec = DTypeLike | None
TensorMap = dict[str, np.ndarray]


class BenchmarkRunner(Protocol):
    name: str
    stateful: bool
    cross_attention_stateful: bool
    cache_shape: tuple[object, ...]
    dtypes: Mapping[str, DTypeSpec]
    feed_dtypes: Mapping[str, DTypeSpec]

    @abstractmethod
    def reset_state(self) -> None:
        pass

    @abstractmethod
    def encode(self, input_ids: np.ndarray) -> TensorMap:
        pass

    @abstractmethod
    def initialize_cross_attention_state(
        self,
        encoder_outputs: TensorMap,
    ) -> None:
        pass

    @abstractmethod
    def decode(self, inputs: TensorMap) -> TensorMap:
        pass
