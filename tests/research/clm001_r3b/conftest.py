from pathlib import Path
import runpy
import sys
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b"
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"),
    run_name="test_r3b_bootstrap",
)
sys.path.insert(0, str(BASE / "implementation"))


@pytest.fixture
def case(tmp_path):
    from minicells_runtime.smoke import configuration, prepare
    from minicells_runtime.store import digest
    from clm001_r3.common import source_sha
    from clm001_r3b.common import load, runtime
    from clm001_r3b.runner import new_run

    proposed = load()
    engine = configuration()
    proposed["engine"] = engine
    proposed["diagnostics"].update(
        steps=[0, 3, 6, 8, 10],
        functional_steps=[0, 10],
        action_type_states=2,
        action_type_mc_samples=2,
    )
    assets = prepare(tmp_path / "assets", engine)
    config = runtime(
        proposed,
        source_sha(),
        beta=0.1,
        pilot_hash="CPU_ONLY",
        run_id="r3b-cpu",
        assets_hash=digest(assets / "manifest.json"),
    )
    root = tmp_path / "run"
    new_run(proposed, config, assets, root)
    return proposed, config, root


# Reuse the already tested offline Hub backend; never touch a real repository.
sys.path.insert(0, str(ROOT / "tests/engineering"))
