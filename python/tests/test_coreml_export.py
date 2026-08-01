import json
import tempfile
import unittest
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import coremltools as ct
import torch
from coremltools.models import MLModel

import needle.export.coreml as coreml_export
from needle import DecoderExportStrategy, Needle, NeedleModelConfiguation
from needle.export.coreml import CoreMLComputeUnits, export_needle_coreml


def coreml_operation_histogram(model: MLModel) -> Counter[str]:
    """Counts operations in a converted model, including operations in nested blocks."""
    histogram = Counter[str]()

    def count_operations(operations: Sequence[Any]) -> None:
        for operation in operations:
            histogram[operation.op_type] += 1
            for block in operation.blocks:
                count_operations(block.operations)

    program = model._mil_program
    if program is None:
        raise ValueError("CoreML model does not retain its MIL program")
    for function in program.functions.values():
        count_operations(function.operations)
    return histogram


class CoreMLExportTests(unittest.TestCase):
    def test_coreml_compile_platforms_are_normalized_and_validated(self) -> None:
        self.assertEqual(
            coreml_export._validated_compile_platforms(["watchos", "iOS"]),
            ("watchOS", "iOS"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported CoreML compile platform"):
            coreml_export._validated_compile_platforms(["Linux"])
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            coreml_export._validated_compile_platforms(["watchOS", "watchos"])

    def test_export_needle_coreml_runs_end_to_end_on_local_bundle(self) -> None:
        self._assert_export_runs_end_to_end()

    def test_coreml_export_decomposes_decoder_attention_by_default(self) -> None:
        histograms: list[Counter[str]] = []
        persist_model = coreml_export._persist_model

        def persist_model_with_histogram(model: MLModel, **kwargs: Any) -> None:
            histograms.append(coreml_operation_histogram(model))
            persist_model(model, **kwargs)

        with patch.object(
            coreml_export,
            "_persist_model",
            side_effect=persist_model_with_histogram,
        ):
            self._assert_export_runs_end_to_end(
                compute_units=CoreMLComputeUnits.CPU_AND_GPU
            )

        encoder_histogram, decoder_histogram = histograms
        self.assertGreater(encoder_histogram["scaled_dot_product_attention"], 0)
        self.assertEqual(decoder_histogram["scaled_dot_product_attention"], 0)
        self.assertGreater(decoder_histogram["matmul"], 0)
        self.assertGreater(decoder_histogram["softmax"], 0)

    def _assert_export_runs_end_to_end(
        self,
        compute_units: CoreMLComputeUnits = CoreMLComputeUnits.ALL,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as source_name,
            tempfile.TemporaryDirectory() as output_name,
        ):
            source_directory = Path(source_name)
            output_directory = Path(output_name)

            (source_directory / "configuration.json").write_text(
                json.dumps(
                    {
                        "vocab_size": 16,
                        "d_model": 8,
                        "hidden_size": 8,
                        "num_attention_heads": 2,
                        "num_kv_heads": 1,
                        "num_encoder_layers": 1,
                        "num_decoder_layers": 1,
                        "num_hidden_layers": 1,
                        "pad_token_id": 0,
                        "decoder_start_token_id": 1,
                        "tie_word_embeddings": True,
                        "torch_dtype": "float32",
                    }
                )
            )
            (source_directory / "tokenizer.json").write_text("{}")

            configuration = NeedleModelConfiguation.from_file(
                source_directory / "configuration.json"
            )
            torch.save(
                Needle(
                    configuration,
                    decoder_strategy=DecoderExportStrategy.reference(),
                ).state_dict(),
                source_directory / "weights.pkl",
            )

            result = export_needle_coreml(
                str(source_directory),
                output_directory,
                compute_units=compute_units,
            )

            self.assertEqual(result, output_directory.resolve())
            self.assertTrue((result / "configuration.json").exists())
            self.assertTrue((result / "tokenizer.json").exists())

            encoder_model = ct.models.MLModel(
                str(result / "encoder.mlpackage"),
                skip_model_load=True,
            )
            decoder_model = ct.models.MLModel(
                str(result / "decoder.mlpackage"),
                skip_model_load=True,
            )
            self.assertEqual(
                [field.name for field in encoder_model.get_spec().description.input],
                ["input_ids"],
            )
            self.assertEqual(
                [field.name for field in encoder_model.get_spec().description.output],
                [
                    "cross_attention_mask",
                    "encoder_projected_k",
                    "encoder_projected_v",
                ],
            )
            decoder_model_spec = decoder_model.get_spec()
            self.assertEqual(
                [field.name for field in decoder_model_spec.description.input],
                [
                    "input_ids",
                    "cache_position",
                    "self_attention_mask",
                    "cross_attention_mask",
                ],
            )
            self.assertEqual(
                [field.name for field in decoder_model_spec.description.output],
                ["logits"],
            )
            expected_state_names = [
                "self_attention_key_cache_0",
                "self_attention_value_cache_0",
                "cross_attention_key_cache_0",
                "cross_attention_value_cache_0",
            ]
            self.assertEqual(
                [field.name for field in decoder_model_spec.description.state],
                expected_state_names,
            )
            program_input_names = {
                value.name
                for function in decoder_model_spec.mlProgram.functions.values()
                for value in function.inputs
            }
            self.assertTrue(set(expected_state_names).issubset(program_input_names))


if __name__ == "__main__":
    unittest.main()
