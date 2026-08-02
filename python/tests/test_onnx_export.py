import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import onnx  # type: ignore[import-not-found]
import onnxruntime as ort  # type: ignore[import-not-found]
import torch

import cli
from needle import (
    DecoderExportStrategy,
    JSONObject,
    Needle,
    NeedleModelConfiguation,
    empty_decoder_caches,
)
from needle.export.helpers import (
    sample_decoder_inputs_from_encoder_outputs,
    sample_encoder_input,
)
from needle.export.onnx import (
    export_needle_onnx,  # pyright: ignore[reportMissingImports]
)
from needle.export.onnx_compression import (  # pyright: ignore[reportMissingImports]
    ONNXModelComponent,
)


class RecordingCompressor:
    def __init__(self) -> None:
        self.components = list[ONNXModelComponent]()

    def compress(
        self,
        source: Path,
        destination: Path,
        *,
        component: ONNXModelComponent,
    ) -> None:
        self.components.append(component)
        onnx.save(onnx.load(source), destination)


class ONNXExportTests(unittest.TestCase):
    def test_cli_exports_int8_quantized_bundle_end_to_end(self) -> None:
        with (
            tempfile.TemporaryDirectory() as source_name,
            tempfile.TemporaryDirectory() as output_name,
        ):
            source_directory = Path(source_name)
            output_directory = Path(output_name)
            configuration_data: JSONObject = {
                "vocab_size": 16,
                "d_model": 8,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_hidden_layers": 1,
                "max_seq_len": 4,
                "pad_token_id": 0,
                "decoder_start_token_id": 1,
                "tie_word_embeddings": True,
                "torch_dtype": "float32",
            }
            configuration_contents = json.dumps(configuration_data)
            tokenizer_contents = '{"type":"test-tokenizer"}'
            (source_directory / "configuration.json").write_text(configuration_contents)
            (source_directory / "tokenizer.json").write_text(tokenizer_contents)

            configuration = NeedleModelConfiguation.from_json_object(configuration_data)
            needle = Needle(
                configuration,
                decoder_strategy=DecoderExportStrategy.reference(),
            )
            torch.save(needle.state_dict(), source_directory / "weights.pkl")

            result = cli.main(
                [
                    "--backend",
                    "onnx",
                    "--onnx-quantization",
                    "int8",
                    "--source",
                    str(source_directory),
                    "--output",
                    str(output_directory),
                ]
            )

            self.assertEqual(result, 0)
            exported_configuration = json.loads(
                (output_directory / "configuration.json").read_text()
            )
            self.assertEqual(
                exported_configuration,
                {
                    **configuration_data,
                    "decoder_max_length": 4,
                    "dtype": "float32",
                },
            )
            self.assertEqual(
                (output_directory / "tokenizer.json").read_text(),
                tokenizer_contents,
            )

            encoder_path = output_directory / "encoder.onnx"
            decoder_path = output_directory / "decoder.onnx"
            for model_path in (encoder_path, decoder_path):
                model = onnx.load(model_path)
                quantized_nodes = [
                    node for node in model.graph.node if node.op_type == "MatMulNBits"
                ]
                self.assertGreater(len(quantized_nodes), 0)
                self.assertTrue(
                    all(
                        next(
                            attribute.i
                            for attribute in node.attribute
                            if attribute.name == "bits"
                        )
                        == 8
                        for node in quantized_nodes
                    )
                )

            encoder_session = ort.InferenceSession(
                str(encoder_path), providers=["CPUExecutionProvider"]
            )
            encoder_input = sample_encoder_input(
                configuration, configuration.encoder_max_length
            )
            encoder_outputs = encoder_session.run(
                None,
                {"input_ids": encoder_input.numpy()},
            )
            encoder_output_tensors = (
                torch.from_numpy(encoder_outputs[0]),
                torch.from_numpy(encoder_outputs[1]),
                torch.from_numpy(encoder_outputs[2]),
            )
            decoder_prefix = sample_decoder_inputs_from_encoder_outputs(
                configuration,
                encoder_output_tensors,
            )
            decoder_inputs = (
                *decoder_prefix,
                *empty_decoder_caches(configuration, dtype=torch.float32),
            )
            decoder_session = ort.InferenceSession(
                str(decoder_path), providers=["CPUExecutionProvider"]
            )
            decoder_outputs = decoder_session.run(
                None,
                {
                    model_input.name: value.numpy()
                    for model_input, value in zip(
                        decoder_session.get_inputs(), decoder_inputs, strict=True
                    )
                },
            )

            self.assertEqual(
                [output.name for output in decoder_session.get_outputs()],
                ["logits", "key_cache_delta", "value_cache_delta"],
            )
            self.assertTrue(
                all(
                    np.isfinite(cast(np.ndarray, value)).all()
                    for value in decoder_outputs
                )
            )

    def test_export_defaults_to_float32_for_portable_runtime_execution(self) -> None:
        with (
            tempfile.TemporaryDirectory() as source_name,
            tempfile.TemporaryDirectory() as output_name,
        ):
            source_directory = Path(source_name)
            output_directory = Path(output_name)
            configuration_data: JSONObject = {
                "vocab_size": 16,
                "d_model": 8,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_hidden_layers": 1,
                "max_seq_len": 4,
                "torch_dtype": "bfloat16",
            }
            (source_directory / "configuration.json").write_text(
                json.dumps(configuration_data)
            )
            (source_directory / "tokenizer.json").write_text("{}")
            configuration = NeedleModelConfiguation.from_json_object(configuration_data)
            torch.save(
                Needle(
                    configuration,
                    decoder_strategy=DecoderExportStrategy.reference(),
                ).state_dict(),
                source_directory / "weights.pkl",
            )
            compressor = RecordingCompressor()

            result = export_needle_onnx(
                str(source_directory),
                output_directory,
                compressor=compressor,
            )

            self.assertEqual(compressor.components, ["encoder", "decoder"])
            self.assertEqual(result, output_directory.resolve())
            exported_configuration = json.loads(
                (result / "configuration.json").read_text()
            )
            self.assertEqual(exported_configuration["dtype"], "float32")
            self.assertEqual(exported_configuration["torch_dtype"], "float32")
            self.assertEqual(exported_configuration["decoder_max_length"], 4)

            encoder_model = onnx.load(result / "encoder.onnx")
            decoder_model = onnx.load(result / "decoder.onnx")
            # ORT registers SimplifiedLayerNormalization in the default domain,
            # but the ONNX checker does not know that schema at opset 21. Loading
            # sessions validates the final runtime graphs instead.
            ort.InferenceSession(
                str(result / "encoder.onnx"), providers=["CPUExecutionProvider"]
            )
            ort.InferenceSession(
                str(result / "decoder.onnx"), providers=["CPUExecutionProvider"]
            )
            decoder_operations = [node.op_type for node in decoder_model.graph.node]
            self.assertIn("SimplifiedLayerNormalization", decoder_operations)
            self.assertNotIn("ReduceMean", decoder_operations)
            self.assertEqual(
                encoder_model.graph.input[0].type.tensor_type.shape.dim[1].dim_param,
                "encoder_sequence_length",
            )
            decoder_input_shapes = {
                model_input.name: model_input.type.tensor_type.shape
                for model_input in decoder_model.graph.input
            }
            self.assertEqual(
                decoder_input_shapes["cross_attention_mask"].dim[3].dim_param,
                "encoder_sequence_length",
            )
            self.assertEqual(
                decoder_input_shapes["key_cache"].dim[1].dim_param,
                "decoder_cache_length",
            )
            self.assertFalse(
                any(
                    initializer.data_type == onnx.TensorProto.BFLOAT16
                    for model in (encoder_model, decoder_model)
                    for initializer in model.graph.initializer
                )
            )


if __name__ == "__main__":
    unittest.main()
