import json
from pathlib import Path

from audit_qwen import REVISION, WEIGHTS
from phase3a_templates import SPLIT_FAMILIES


def test_formal_draft_preserves_frozen_inputs():
    root = Path(__file__).resolve().parent
    draft = json.loads((root / 'phase3a_v21_config.json').read_text())
    assert draft['status'] == 'NOT_FROZEN'
    assert draft['model']['revision'] == REVISION
    assert all(draft['model']['hashes'][k] == v for k, v in WEIGHTS.items())
    assert draft['dataset']['families'] == {k: list(v) for k, v in SPLIT_FAMILIES.items()}
    assert draft['dataset']['seeds'] == [42, 43, 44, 45, 46]
    assert draft['evaluation']['candidates'] == [32, 128, 512]
    assert draft['evaluation']['test'] == 'SEALED'
