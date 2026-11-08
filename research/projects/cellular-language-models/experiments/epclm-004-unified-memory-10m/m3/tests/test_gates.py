"""One ordered suite, also used directly by the single Kaggle entry point."""
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('m3_entry',BASE/'run.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_ordered_math_causality_state_mag_and_recall(tmp_path):
    entry.verify_frozen()
    protocol=json.loads((BASE/'configs/smoke.json').read_text())
    out=Path(os.environ.get('M3_GATE_OUTPUT',str(tmp_path)))
    records=entry.cpu_gates(out,protocol)
    assert list(records)==[f'gate{i}' for i in range(1,6)]
    assert all(r['status']=='PASS' for r in records.values())
