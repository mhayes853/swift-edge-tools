import tempfile
import unittest
from pathlib import Path

import torch

from needle.torch_utils import extract_state_dict, load_state_dict, torch_dtype


class TorchUtilsTests(unittest.TestCase):
    def test_load_state_dict_accepts_pickled_state_dict_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "weights.pkl"
            expected = {"model.embed_tokens.weight": torch.zeros((2, 2))}
            torch.save({"state_dict": expected}, weights_path)

            state_dict = load_state_dict(weights_path)

        self.assertEqual(tuple(state_dict["model.embed_tokens.weight"].shape), (2, 2))

    def test_torch_dtype_lookup(self) -> None:
        self.assertEqual(torch_dtype("bfloat16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            torch_dtype("bogus")

    def test_extract_state_dict_from_module(self) -> None:
        module = torch.nn.Linear(3, 3)
        state_dict = extract_state_dict(module)
        self.assertIn("weight", state_dict)

    def test_extract_state_dict_from_nested_mapping(self) -> None:
        nested = {"state_dict": {"a": torch.ones(1)}}
        self.assertEqual(extract_state_dict(nested)["a"].shape, (1,))


if __name__ == "__main__":
    unittest.main()