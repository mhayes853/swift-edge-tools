from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..cache_layout import (  # pyright: ignore[reportMissingImports]
    cross_attention_source,
)
from .interface import (  # pyright: ignore[reportMissingImports]
    BenchmarkRunner,
    TensorMap,
)

_ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(bfloat16)": None,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class OnnxRunner(BenchmarkRunner):
    name = "onnx"
    stateful = False
    cross_attention_stateful = False

    def __init__(self, bundle: Path) -> None:
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.encoder = ort.InferenceSession(
            str(bundle / "encoder.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(bundle / "decoder.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.dtypes = {
            feature.name: _ONNX_DTYPES[feature.type]
            for feature in (*self.decoder.get_inputs(), *self.decoder.get_outputs())
        }
        self.feed_dtypes = self.dtypes
        self.decoder_output_names = tuple(
            output.name for output in self.decoder.get_outputs()
        )
        self.cache_shape = tuple(
            next(
                feature.shape
                for feature in self.decoder.get_inputs()
                if feature.name == "key_cache"
            )
        )

    def reset_state(self) -> None:
        pass

    def encode(self, input_ids: np.ndarray) -> TensorMap:
        outputs = self.encoder.run(None, {"input_ids": input_ids.astype(np.int64)})
        names = [output.name for output in self.encoder.get_outputs()]
        return {
            name: np.asarray(value) for name, value in zip(names, outputs, strict=True)
        }

    def initialize_cross_attention_state(
        self,
        encoder_outputs: TensorMap,
    ) -> None:
        pass

    def decode(self, inputs: TensorMap) -> TensorMap:
        outputs = self.decoder.run(list(self.decoder_output_names), inputs)
        return {
            name: np.asarray(value)
            for name, value in zip(self.decoder_output_names, outputs, strict=True)
        }


class CoreMLRunner(BenchmarkRunner):
    name = "coreml"

    def __init__(self, bundle: Path) -> None:
        import coremltools as ct

        self.encoder = ct.models.MLModel(
            str(bundle / "encoder.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            optimization_hints={
                "reshapeFrequency": ct.ReshapeFrequency.Infrequent,
            },
        )
        self.decoder = ct.models.MLModel(
            str(bundle / "decoder.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        spec = self.decoder.get_spec()
        encoder_spec = self.encoder.get_spec()
        self.dtypes = {}
        for feature in (
            *encoder_spec.description.output,
            *spec.description.input,
            *spec.description.output,
            *spec.description.state,
        ):
            kind = feature.type.multiArrayType.dataType
            self.dtypes[feature.name] = {
                65568: np.float32,
                65552: np.float16,
                65600: np.float64,
                131104: np.int32,
                131080: np.int8,
            }.get(kind, np.float32)
        # coremltools.predict upconverts float16 numpy inputs on every call. Feed
        # float32 so the bridge passes them through without another allocation.
        self.feed_dtypes = {
            name: (np.float32 if dtype == np.float16 else dtype)
            for name, dtype in self.dtypes.items()
        }
        self.decoder_input_names = tuple(
            feature.name for feature in spec.description.input
        )
        state_features = {feature.name: feature for feature in spec.description.state}
        if "key_cache" in state_features:
            self.cache_shape = tuple(
                state_features["key_cache"].type.multiArrayType.shape
            )
        else:
            layer_key_states = sorted(
                name
                for name in state_features
                if name.startswith("self_attention_key_cache_")
            )
            self.cache_shape = (
                (
                    len(layer_key_states),
                    *state_features[layer_key_states[0]].type.multiArrayType.shape,
                )
                if layer_key_states
                else ()
            )
        self.cross_attention_stateful = any(
            name.startswith("cross_attention_key_cache_") for name in state_features
        )
        self.stateful = len(state_features) > 0
        self.state = self.decoder.make_state() if self.stateful else None

    def reset_state(self) -> None:
        if self.stateful:
            self.state = self.decoder.make_state()

    def encode(self, input_ids: np.ndarray) -> TensorMap:
        outputs = self.encoder.predict({"input_ids": input_ids.astype(np.int32)})
        # Core ML returns views into buffers that subsequent predictions recycle.
        return {name: np.array(value, copy=True) for name, value in outputs.items()}

    def initialize_cross_attention_state(
        self,
        encoder_outputs: TensorMap,
    ) -> None:
        if not self.cross_attention_stateful or self.state is None:
            return
        for name in self.dtypes:
            if not name.startswith(
                ("cross_attention_key_cache_", "cross_attention_value_cache_")
            ):
                continue
            source_name, layer_index = cross_attention_source(name)
            state_value = self.state.read_state(name)
            initialized = np.zeros(state_value.shape, dtype=self.dtypes[name])
            source = encoder_outputs[source_name][layer_index]
            initialized[..., : source.shape[-2], :] = source
            self.state.write_state(name, initialized)

    def decode(self, inputs: TensorMap) -> TensorMap:
        feed = {name: inputs[name] for name in self.decoder_input_names}
        outputs = self.decoder.predict(feed, state=self.state)
        return {name: np.array(value, copy=True) for name, value in outputs.items()}


async def _load_coreai_functions(bundle: Path) -> tuple[Any, Any]:
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

    encoder_options = SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.neural_engine()
    )
    decoder_options = SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.gpu()
    )
    encoder = await AIModel.load(
        bundle / "encoder.aimodel",
        specialization_options=encoder_options,
    )
    decoder = await AIModel.load(
        bundle / "decoder.aimodel",
        specialization_options=decoder_options,
    )
    return encoder.load_function("main"), decoder.load_function("main")


class CoreAIRunner(BenchmarkRunner):
    name = "coreai"

    def __init__(self, bundle: Path) -> None:
        import asyncio

        self._loop = asyncio.new_event_loop()
        self.encoder, self.decoder = self._loop.run_until_complete(
            _load_coreai_functions(bundle)
        )
        self.stateful = len(self.decoder.desc.state_names) > 0
        self.state = self._new_state()
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "int32": np.int32,
            "int64": np.int64,
        }
        self.dtypes = {
            name: dtype_map[str(self.decoder.desc.input_descriptor(name).dtype)]
            for name in self.decoder.desc.input_names
        }
        self.dtypes.update(
            {
                name: dtype_map[str(self.decoder.desc.output_descriptor(name).dtype)]
                for name in self.decoder.desc.output_names
            }
        )
        self.dtypes.update(
            {
                name: dtype_map[str(self.decoder.desc.state_descriptor(name).dtype)]
                for name in self.decoder.desc.state_names
            }
        )
        self.dtypes.update(
            {
                name: dtype_map[str(self.encoder.desc.output_descriptor(name).dtype)]
                for name in self.encoder.desc.output_names
            }
        )
        self.feed_dtypes = self.dtypes
        self.decoder_input_names = tuple(self.decoder.desc.input_names)
        if "key_cache" in self.decoder.desc.state_names:
            self.cache_shape = tuple(
                self.decoder.desc.state_descriptor("key_cache").shape
            )
        else:
            layer_key_states = sorted(
                name
                for name in self.decoder.desc.state_names
                if name.startswith("self_attention_key_cache_")
            )
            self.cache_shape = (
                (
                    len(layer_key_states),
                    *self.decoder.desc.state_descriptor(layer_key_states[0]).shape,
                )
                if layer_key_states
                else ()
            )
        self.cross_attention_stateful = any(
            name.startswith("cross_attention_key_cache_")
            for name in self.decoder.desc.state_names
        )

    def _new_state(self) -> dict[str, Any] | None:
        if not self.stateful:
            return None
        from coreai.runtime import NDArray

        return {
            name: NDArray.from_descriptor(self.decoder.desc.state_descriptor(name))
            for name in self.decoder.desc.state_names
        }

    def reset_state(self) -> None:
        self.state = self._new_state()

    def encode(self, input_ids: np.ndarray) -> TensorMap:
        from coreai.runtime import NDArray

        outputs = self._loop.run_until_complete(
            self.encoder({"input_ids": NDArray(input_ids.astype(np.int32))})
        )
        return {name: value.numpy() for name, value in outputs.items()}

    def initialize_cross_attention_state(
        self,
        encoder_outputs: TensorMap,
    ) -> None:
        if not self.cross_attention_stateful or self.state is None:
            return
        from coreai.runtime import NDArray

        for name in self.decoder.desc.state_names:
            if not name.startswith(
                ("cross_attention_key_cache_", "cross_attention_value_cache_")
            ):
                continue
            source_name, layer_index = cross_attention_source(name)
            shape = tuple(self.decoder.desc.state_descriptor(name).shape)
            initialized = np.zeros(shape, dtype=self.dtypes[name])
            source = encoder_outputs[source_name][layer_index]
            initialized[..., : source.shape[-2], :] = source
            self.state[name] = NDArray(initialized)

    def decode(self, inputs: TensorMap) -> TensorMap:
        from coreai.runtime import NDArray

        feed = {name: NDArray(inputs[name]) for name in self.decoder_input_names}
        outputs = self._loop.run_until_complete(self.decoder(feed, state=self.state))
        return {name: value.numpy() for name, value in outputs.items()}


RUNNERS = {
    "onnx": OnnxRunner,
    "coreml": CoreMLRunner,
    "coreai": CoreAIRunner,
}
