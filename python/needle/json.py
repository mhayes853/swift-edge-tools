from __future__ import annotations

import json
from pathlib import Path
from typing import Union

JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, list["JSONValue"], dict[str, "JSONValue"]]
JSONObject = dict[str, JSONValue]


def load_json_object(path: Union[str, Path]) -> JSONObject:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload
