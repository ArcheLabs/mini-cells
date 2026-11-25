from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from minicells_native.backend import validate_loaded_checkpoint
from minicells_native.errors import ModelLoadError
from minicells_native.loader import load_backbone
from minicells_native.model_assets import ASSETS_13B, MODEL_13B, REVISION_13B, expected_shapes
from minicells_native.model_registry import MODEL, MODELS, REVISION, TOKENIZER, TOKENIZER_REV
from test_model_assets import CONFIG


class ModelRegistryTest(unittest.TestCase):
    @patch("minicells_native.backend.Backbone")
    @patch("huggingface_hub.snapshot_download")
    @patch("torch.cuda.is_available", return_value=True)
    def test_asset_selection(self, _cuda, download, backbone):
        def resolve(model, *, revision, **_):
            return Path(cache) / revision

        download.side_effect = resolve
        with TemporaryDirectory() as cache:
            load_backbone(MODEL_13B, "cuda:0", "fp16", cache, source="pinned")
            download.assert_called_once()
            args, kwargs = download.call_args
            self.assertEqual(args, (MODEL_13B,))
            self.assertEqual(kwargs["revision"], REVISION_13B)
            self.assertEqual(set(kwargs["allow_patterns"]), set(ASSETS_13B))
            snapshot = Path(cache) / REVISION_13B
            backbone.assert_called_once_with("pinned", snapshot, snapshot, dtype="float16", device="cuda:0")

            download.reset_mock()
            backbone.reset_mock()
            load_backbone(MODEL, "cuda", "auto", cache, source="pinned")
            self.assertEqual([call.args for call in download.call_args_list], [(MODEL,), (TOKENIZER,)])
            self.assertEqual([call.kwargs["revision"] for call in download.call_args_list],
                             [REVISION, TOKENIZER_REV])
            backbone.assert_called_once_with("pinned", Path(cache) / REVISION, Path(cache) / TOKENIZER_REV,
                                             dtype="float16", device="cuda")
            download.reset_mock()
            with self.assertRaisesRegex(ModelLoadError, "registered MoM model"):
                load_backbone("unregistered/model", "cuda", "auto", cache, source="pinned")
            download.assert_not_called()
        self.assertEqual(MODELS[MODEL_13B].hidden_size, 2048)

    def test_post_load_checks(self):
        shapes = expected_shapes(CONFIG)
        params = {name: torch.nn.Parameter(torch.empty(shape, device="meta", dtype=torch.float16))
                  for name, shape in shapes.items()}
        tied = SimpleNamespace(weight=params["model.embeddings.weight"])
        params["lm_head.weight"] = tied.weight
        values = tuple(dict.fromkeys(params.values()))
        model = SimpleNamespace(
            named_parameters=lambda **kwargs: params.items(),
            parameters=lambda: iter(values),
            get_input_embeddings=lambda: tied,
            get_output_embeddings=lambda: SimpleNamespace(weight=params["lm_head.weight"]),
        )
        audit = dict(config=CONFIG, parameter_count=sum(value.numel() for value in values))
        validate_loaded_checkpoint(model, {}, audit, torch.float16, torch.device("meta"))
        with self.assertRaisesRegex(ValueError, "Weight load mismatch"):
            validate_loaded_checkpoint(model, {"missing_keys": ["x"]}, audit, torch.float16,
                                       torch.device("meta"))
        with self.assertRaisesRegex(ValueError, "device or dtype"):
            validate_loaded_checkpoint(model, {}, audit, torch.float32, torch.device("meta"))
        params["lm_head.weight"] = torch.nn.Parameter(torch.empty_like(tied.weight))
        with self.assertRaisesRegex(ValueError, "must share"):
            validate_loaded_checkpoint(model, {}, audit, torch.float16, torch.device("meta"))
