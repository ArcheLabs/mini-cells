"""Single new-behavior preflight; no repetition of M2/M3 gate matrices."""
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('m4_entry',BASE/'run.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_new_fusion_causality_ablation_streaming_and_checkpoint(tmp_path):
    entry.verify_frozen();protocol=json.loads((BASE/'configs/smoke.json').read_text())
    out=Path(os.environ.get('M4_PREFLIGHT_OUTPUT',str(tmp_path)))
    result=entry.preflight(out,protocol)
    assert result['status']=='PASS' and result['training']['ledger']['committed_updates']==2
    assert len(result['ablations'])==6 and result['checkpoint_restore_error']==0
