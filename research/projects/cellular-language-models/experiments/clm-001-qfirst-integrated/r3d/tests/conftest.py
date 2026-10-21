from pathlib import Path
import runpy
import sys
import pytest

BASE = Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE / "run.py"), run_name="r3d_test_bootstrap")
sys.path.insert(0, str(BASE / "tests"))


@pytest.fixture
def config():
    from minicells_r3d.common import configuration
    from minicells_r3c_moe_bridge.common import synthetic_configuration

    c = configuration()
    c["synthetic"] = True
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["engine"]["balance"]["coefficient"] = 0.0
    c["training"].update(
        a_steps=2,
        b_steps=4,
        policy_every=1,
        states_per_policy_update=4,
        checkpoint_every=2,
        print_every=4,
        train_tokens_per_step=8,
    )
    c["statistics"].update(bootstrap_replicates=30, zero_exposure_window=2)
    c["seeds"] = [11, 12, 13]
    return c
