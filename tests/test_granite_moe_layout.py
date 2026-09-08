from __future__ import annotations

from types import SimpleNamespace

from minicells.granite_moe_layout import identify_packed_expert_tensors
from minicells.moe_subexpert import validate_group_shapes


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)


class _FakeGranite:
    def __init__(self):
        # Deliberately mirrors the failing hosted run: model-level
        # intermediate_size is 1024 while each expert is width 512.
        self.config = SimpleNamespace(num_local_experts=32, intermediate_size=1024)
        self._parameters = [
            (
                "model.layers.23.block_sparse_moe.input_linear.weight",
                _FakeTensor((32, 1024, 1024)),
            ),
            (
                "model.layers.23.block_sparse_moe.output_linear.weight",
                _FakeTensor((32, 1024, 512)),
            ),
        ]

    def named_parameters(self):
        return iter(self._parameters)


def test_identifies_real_granite_expert_width_from_tensor_geometry():
    model = _FakeGranite()
    gate_up, down = identify_packed_expert_tensors(model, 23)

    assert gate_up[0].endswith("input_linear.weight")
    assert down[0].endswith("output_linear.weight")
    assert validate_group_shapes(gate_up[1], down[1]) == 512
