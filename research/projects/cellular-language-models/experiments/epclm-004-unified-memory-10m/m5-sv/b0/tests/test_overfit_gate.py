"""Scientific pass never follows from publication success alone."""
from pathlib import Path
import importlib.util


def test_three_arms_and_verified_publication_are_required():
    path=Path(__file__).resolve().parents[1]/'overfit.py'
    spec=importlib.util.spec_from_file_location('b0_overfit_gate_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arms=['E0','E6','E7-Innovation']
    results={a:{'status':'OVERFIT_ESTABLISHED','publication_status':'VERIFIED'} for a in arms}
    assert module.overfit_gate(results,arms)
    results['E6']['status']='OVERFIT_NOT_ESTABLISHED';assert not module.overfit_gate(results,arms)
    results['E6']['status']='OVERFIT_ESTABLISHED';results['E0']['publication_status']='FAILED'
    assert not module.overfit_gate(results,arms)
    del results['E0'];assert not module.overfit_gate(results,arms)
