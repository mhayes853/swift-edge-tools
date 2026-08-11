"""Needle model core.

Backend exporters live under `needle.export` and are imported explicitly so
that importing the core model never pulls in an optional runtime dependency.
"""

from .cache_layout import (
    empty_decoder_caches,
    explicit_cache_shape,
)
from .json import JSONObject, JSONScalar, JSONValue
from .needle_configuration import NeedleModelConfiguation
from .needle_torch import Needle, NeedleDecoder, NeedleEncoder
from .torch_utils import (
    StateDictPayload,
    extract_state_dict,
    load_state_dict,
    normalize_state_dict,
    torch_dtype,
)

__all__ = [
    "JSONObject",
    "JSONScalar",
    "JSONValue",
    "Needle",
    "NeedleDecoder",
    "NeedleEncoder",
    "NeedleModelConfiguation",
    "StateDictPayload",
    "empty_decoder_caches",
    "explicit_cache_shape",
    "extract_state_dict",
    "load_state_dict",
    "normalize_state_dict",
    "torch_dtype",
]
