import json
import struct
from pathlib import Path

import pytest

from minicells.moe_conversion import (
    MANIFEST_NAME,
    MoeConversionError,
    create_clm_moe_bundle,
    load_manifest,
    materialize_hf_checkpoint,
    verify_clm_moe_bundle,
)


def _write_fake_safetensors(
    path: Path,
    specs: dict[str, tuple[str, list[int], bytes]],
) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in specs.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _write_tiny_checkpoint(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "granitemoe",
                "architectures": ["GraniteMoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "intermediate_size": 2,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_fake_safetensors(
        root / "model.safetensors",
        {
            "model.layers.0.block_sparse_moe.input_linear.weight": (
                "F32",
                [2, 4, 4],
                bytes(range(128)),
            ),
            "model.layers.0.block_sparse_moe.output_linear.weight": (
                "F32",
                [2, 4, 2],
                bytes(range(64)),
            ),
            "model.layers.0.block_sparse_moe.router.layer.weight": (
                "F32",
                [2, 4],
                bytes(range(32)),
            ),
            "model.embed_tokens.weight": (
                "F32",
                [4, 4],
                bytes(range(64, 128)),
            ),
        },
    )


@pytest.mark.unit
def test_moe_substrate_round_trip_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tiny_checkpoint(source)

    bundle = tmp_path / "bundle"
    manifest = create_clm_moe_bundle(
        source,
        bundle,
        source_model_id="local/test-granitemoe",
        source_revision="fixture-v1",
    )

    assert manifest["conversion"] == {
        "kind": "substrate_wrap",
        "execution_semantics": "preserve_hf_moe",
        "expert_is_cell": False,
        "mutation_state": "none",
    }
    assert len(manifest["expert_addresses"]) == 4
    assert {record["role"] for record in manifest["tensors"]} >= {
        "shared_backbone",
        "moe_router",
        "moe_packed_experts",
    }

    verification = verify_clm_moe_bundle(bundle)
    assert verification["status"] == "PASS"
    assert load_manifest(bundle)["identity_sha256"] == manifest["identity_sha256"]

    output = tmp_path / "materialized"
    materialize_hf_checkpoint(bundle, output)
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (output / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()
    assert (bundle / MANIFEST_NAME).is_file()


@pytest.mark.unit
def test_conversion_001_rejects_non_granite_moe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MoeConversionError, match="model_type=granitemoe"):
        create_clm_moe_bundle(source, tmp_path / "bundle", source_model_id="invalid")
