import json
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import coremltools as ct
import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_opt.palettization import KMeansPalettizerConfig
from coreai_opt.quantization import QuantizerConfig

import needle.coreml_export as coreml_export
from needle import (
    AttentionImplementation,
    DecoderExportStrategy,
    Needle,
    NeedleModelConfiguation,
)
from needle.coreml_export import (
    CoreMLComputeUnits,
    coreml_operation_histogram,
    export_needle_coreml,
)
from needle.needle_compression import CoreMLQuantizerCompressor


class CoreMLExportTests(unittest.TestCase):
    def test_coreml_compiler_writes_platform_specific_artifacts(self) -> None:
        output_ctx = tempfile.TemporaryDirectory()
        try:
            output_directory = Path(output_ctx.name)
            source_path = Path("/tmp/encoder.mlpackage")
            with patch.object(coreml_export.subprocess, "run") as run_mock:
                coreml_export._compile_model(
                    source_path,
                    output_directory=output_directory,
                    platforms=("watchOS",),
                )

            platform_directory = output_directory / "compiled" / "watchOS"
            self.assertTrue(platform_directory.is_dir())
            run_mock.assert_called_once_with(
                [
                    "xcrun",
                    "coremlcompiler",
                    "compile",
                    str(source_path),
                    str(platform_directory),
                    "--platform",
                    "watchOS",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            output_ctx.cleanup()

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

    def test_stateful_coreml_decoder_supports_dynamic_cache_prefix(self) -> None:
        def assert_dynamic_cache(output_directory: Path) -> None:
            model = ct.models.MLModel(
                str(output_directory / "decoder.mlpackage"),
                skip_model_load=True,
            )
            mask = next(
                feature
                for feature in model.get_spec().description.input
                if feature.name == "self_attention_mask"
            )
            cache_range = mask.type.multiArrayType.shapeRange.sizeRanges[-1]
            self.assertEqual(cache_range.lowerBound, 1)
            self.assertEqual(cache_range.upperBound, 512)

        self._assert_export_runs_end_to_end(
            dynamic_decoder_cache=True,
            post_export_assertions=assert_dynamic_cache,
        )

    def test_export_needle_coreml_supports_quantizer_compressor(self) -> None:
        compressor = CoreMLQuantizerCompressor(QuantizerConfig.presets.w8())
        self._assert_export_runs_end_to_end(compressor=compressor)

    def test_export_needle_coreml_quantizer_supports_dynamic_head_dimensions(
        self,
    ) -> None:
        compressor = CoreMLQuantizerCompressor(QuantizerConfig.presets.w8())
        self._assert_export_runs_end_to_end(
            compressor=compressor,
            dimensions=128,
        )

    def test_export_needle_coreml_supports_quantizer_and_palettizer_compressor(
        self,
    ) -> None:
        compressor = CoreMLQuantizerCompressor(
            QuantizerConfig.presets.w8(),
            palettizer_config=KMeansPalettizerConfig(),
        )
        self._assert_export_runs_end_to_end(compressor=compressor)

    def test_coreml_export_defaults_to_decomposed_decoder_attention(self) -> None:
        histograms = self._export_operation_histograms(
            compute_units=CoreMLComputeUnits.CPU_AND_GPU,
        )

        self.assertGreater(histograms[0]["scaled_dot_product_attention"], 0)
        self.assertEqual(histograms[1]["scaled_dot_product_attention"], 0)

    def test_coreml_export_avoids_native_sdpa_with_ane(self) -> None:
        histograms = self._export_operation_histograms(
            compute_units=CoreMLComputeUnits.ALL,
        )

        for histogram in histograms:
            self.assertEqual(histogram["scaled_dot_product_attention"], 0)

    def test_coreml_decoder_attention_strategy_can_be_overridden(self) -> None:
        histograms = self._export_operation_histograms(
            compute_units=CoreMLComputeUnits.CPU_AND_GPU,
            decoder_use_native_sdpa=True,
        )

        for histogram in histograms:
            self.assertGreater(histogram["scaled_dot_product_attention"], 0)

    def test_export_needle_coreml_supports_bfloat16_source_weights(self) -> None:
        self._assert_export_runs_end_to_end(source_dtype="bfloat16")

    def test_export_needle_coreml_persists_model_metadata(self) -> None:
        metadata = AIModelAssetMetadata()
        metadata.author = "Needle Tests"
        metadata.model_description = "Needle CoreML export metadata test"
        metadata.license = "A totally legal license"
        metadata.set_custom("suite", "coreml_export")

        def assert_metadata(result: Path) -> None:
            encoder_model = ct.models.MLModel(
                str(result / "encoder.mlpackage"),
                skip_model_load=True,
            )
            decoder_model = ct.models.MLModel(
                str(result / "decoder.mlpackage"),
                skip_model_load=True,
            )
            self._assert_metadata_matches(encoder_model, metadata)
            self._assert_metadata_matches(decoder_model, metadata)

        self._assert_export_runs_end_to_end(
            model_metadata=metadata,
            post_export_assertions=assert_metadata,
        )

    def _export_operation_histograms(
        self,
        *,
        compute_units: CoreMLComputeUnits,
        decoder_use_native_sdpa: bool | None = None,
    ) -> list[Counter[str]]:
        histograms: list[Counter[str]] = []
        persist_model = coreml_export._persist_model

        def persist_model_with_histogram(
            model: ct.models.MLModel,
            **kwargs: Any,
        ) -> None:
            histograms.append(coreml_operation_histogram(model))
            persist_model(model, **kwargs)

        with patch.object(
            coreml_export,
            "_persist_model",
            side_effect=persist_model_with_histogram,
        ):
            self._assert_export_runs_end_to_end(
                compute_units=compute_units,
                decoder_use_native_sdpa=decoder_use_native_sdpa,
            )

        return histograms

    def _assert_export_runs_end_to_end(
        self,
        compressor=None,
        model_metadata: AIModelAssetMetadata | None = None,
        post_export_assertions: Callable[[Path], None] | None = None,
        source_dtype: str = "float32",
        compute_units: CoreMLComputeUnits = CoreMLComputeUnits.ALL,
        decoder_use_native_sdpa: bool | None = None,
        dynamic_decoder_cache: bool = False,
        dimensions: int = 8,
    ) -> None:
        source_ctx = tempfile.TemporaryDirectory()
        output_ctx = tempfile.TemporaryDirectory()
        try:
            source_directory = Path(source_ctx.name)
            output_directory = Path(output_ctx.name)

            (source_directory / "configuration.json").write_text(
                json.dumps(
                    {
                        "vocab_size": 16,
                        "d_model": dimensions,
                        "hidden_size": dimensions,
                        "num_attention_heads": 2,
                        "num_kv_heads": 1,
                        "num_encoder_layers": 1,
                        "num_decoder_layers": 1,
                        "num_hidden_layers": 1,
                        "pad_token_id": 0,
                        "decoder_start_token_id": 1,
                        "tie_word_embeddings": True,
                        "torch_dtype": source_dtype,
                    }
                )
            )
            (source_directory / "tokenizer.json").write_text("{}")

            configuration = NeedleModelConfiguation.from_file(
                source_directory / "configuration.json"
            )
            needle_instance = Needle(
                configuration,
                decoder_strategy=DecoderExportStrategy.reference(),
            )
            torch.save(needle_instance.state_dict(), source_directory / "weights.pkl")

            decoder_strategy = DecoderExportStrategy.coreml(
                attention=(
                    AttentionImplementation.NATIVE
                    if decoder_use_native_sdpa
                    else AttentionImplementation.DECOMPOSED
                ),
                dynamic_cache=dynamic_decoder_cache,
            )
            result = export_needle_coreml(
                str(source_directory),
                output_directory,
                compressor=compressor,
                model_metadata=model_metadata,
                compute_units=compute_units,
                decoder_strategy=decoder_strategy,
            )

            self.assertEqual(result, output_directory.resolve())
            self.assertTrue((result / "encoder.mlpackage").exists())
            self.assertTrue((result / "decoder.mlpackage").exists())
            self.assertFalse((result / "encoder.aimodel").exists())
            self.assertFalse((result / "decoder.aimodel").exists())
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
            self.assertEqual(
                [field.name for field in decoder_model.get_spec().description.input],
                [
                    "input_ids",
                    "cache_position",
                    "self_attention_mask",
                    "cross_attention_mask",
                ],
            )
            expected_state_names = [
                "self_attention_key_cache_0",
                "self_attention_value_cache_0",
                "cross_attention_key_cache_0",
                "cross_attention_value_cache_0",
            ]
            decoder_model_spec = decoder_model.get_spec()
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
            self.assertEqual(
                [field.name for field in decoder_model.get_spec().description.output],
                ["logits"],
            )
            if post_export_assertions is not None:
                post_export_assertions(result)
        finally:
            output_ctx.cleanup()
            source_ctx.cleanup()

    def _assert_metadata_matches(
        self,
        model: ct.models.MLModel,
        expected: AIModelAssetMetadata,
    ) -> None:
        self.assertEqual(model.author, expected.author)
        self.assertEqual(model.short_description, expected.model_description)
        self.assertEqual(model.license, expected.license)
        actual_metadata = cast(dict[str, str], model.user_defined_metadata)
        for key, value in self._custom_metadata(expected).items():
            self.assertEqual(actual_metadata[key], value)

    def _custom_metadata(self, metadata: AIModelAssetMetadata) -> dict[str, str]:
        return cast(
            dict[str, str],
            object.__getattribute__(metadata, "creator_defined_metadata"),
        )


if __name__ == "__main__":
    unittest.main()
