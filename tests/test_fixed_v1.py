import json
from pathlib import Path

from minicells.fixed_v1 import model_hash, predict

ROOT = Path(__file__).resolve().parents[1]


def test_python_mirror_matches_rust_native_fixture():
    fixture = json.loads((ROOT / "fixtures/v1/fixed-parity.json").read_text())
    packed = (ROOT / "service/generated/genesis_model.bin").read_bytes()
    assert "0x" + model_hash(packed).hex() == fixture["model_hash"]
    assert predict(packed, fixture["input"]) == fixture["prediction"]
