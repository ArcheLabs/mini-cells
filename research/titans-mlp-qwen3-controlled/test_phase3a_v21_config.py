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


def test_dry_run_rejects_unsealed_test(tmp_path):
    import pytest

    from phase3a_v21_dry_run import load_config

    root = Path(__file__).resolve().parent
    draft = load_config(root / 'phase3a_v21_config.json')
    draft['evaluation']['test'] = 'OPEN'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match='sealed draft'):
        load_config(path)


def test_calibration_handles_ties_conservatively():
    import torch

    from phase3a_v21_dry_run import calibrate

    negatives = torch.tensor([.8, .8, .5, .1])
    cutoff = calibrate(negatives, .25)
    assert (negatives >= cutoff).float().mean() <= .25
    assert cutoff > .8
