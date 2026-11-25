import json
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from minicells_native.model_assets import expected_shapes, inspect_weights, validate_shapes, weight_header

CONFIG = dict(hidden_size=4, num_heads=1, head_dim=2, expand_v=1, intermediate_size=4,
              vocab_size=3, num_memories=2, conv_size=2, num_hidden_layers=1)


def fixture(path, *, unequal_alias=False, trailing_data=b""):
    offset = 0
    header, tensors = {}, []
    for name, shape in expected_shapes(CONFIG).items():
        fill = 1 if unequal_alias and name == "lm_head.weight" else 0
        data = bytes([fill]) * (2 * prod(shape))
        header[name] = dict(dtype="BF16", shape=shape, data_offsets=[offset, offset + len(data)])
        tensors.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(tensors) + trailing_data)


class ModelAssetsTest(unittest.TestCase):
    def test_layout_counts_tied_parameter_once(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            fixture(path)
            report = inspect_weights(path, CONFIG)
        self.assertEqual(report["tensor_count"], len(expected_shapes(CONFIG)))
        self.assertEqual(report["stored_parameter_count"] - report["parameter_count"], 12)

    def test_unequal_tied_values_and_payload_size_are_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            fixture(path, unequal_alias=True)
            with self.assertRaisesRegex(ValueError, "unequal"):
                inspect_weights(path, CONFIG)
            fixture(path, trailing_data=b"\0")
            with self.assertRaisesRegex(ValueError, "payload size"):
                inspect_weights(path, CONFIG)

    def test_exact_names_shapes_and_duplicate_keys(self):
        shapes = expected_shapes(CONFIG)
        observed = dict(shapes)
        observed.pop("lm_head.weight")
        observed["unexpected"] = (1,)
        observed["model.norm.weight"] = (1,)
        with self.assertRaisesRegex(ValueError, "missing=.*unexpected=.*mismatches="):
            validate_shapes(observed, shapes)
        from io import BytesIO

        raw = b'{"x": {}, "x": {}}'
        with self.assertRaisesRegex(ValueError, "Duplicate Safetensors entry"):
            weight_header(BytesIO(len(raw).to_bytes(8, "little") + raw))
