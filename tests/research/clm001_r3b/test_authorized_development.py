from copy import deepcopy
import pytest
from minicells_runtime.store import read, ContinuationError
from clm001_r3.common import source_sha
from clm001_r3b.common import BASE, load, require_training_release
from clm001_r3b.development_release import choose_auto_mode, MODE


def test_explicit_user_release_keeps_budget_honest_and_checks_identity():
    approval = read(BASE / "config/authorization.json")
    evidence = read(BASE / "reports/development-release-evidence.json")
    candidate = load()
    require_training_release(candidate, source_sha(), {}, approval, evidence)
    assert approval["estimated_budget_accepted"] and approval["formal_500_step_authorized"]
    assert approval["two_hour_guarantee"] is False
    assert evidence["performance_gate_status"] == "NOT_VERIFIED"
    bad = deepcopy(approval)
    bad["estimated_budget_accepted"] = False
    with pytest.raises(ContinuationError, match="NOT_AUTHORIZED"):
        require_training_release(candidate, source_sha(), {}, bad, evidence)
    bad = deepcopy(evidence)
    bad["beta"] += 1
    with pytest.raises(ContinuationError, match="IDENTITY_MISMATCH"):
        require_training_release(candidate, source_sha(), {}, approval, bad)


def test_auto_selects_resume_and_never_treats_network_failure_as_missing_run():
    approval = {"release_mode": MODE, "formal_500_step_authorized": True}

    class Existing:
        def locate(self, *args):
            return {"hf_revision": "a" * 40}

    assert choose_auto_mode(Existing(), "a" * 40, "run", approval)[0] == "resume"

    class Missing:
        def locate(self, *args):
            raise ContinuationError("HF_RUN_NOT_FOUND")

    assert choose_auto_mode(Missing(), "a" * 40, "run", approval)[0] == "train"

    class Blocked:
        def locate(self, *args):
            raise ContinuationError("STORAGE_BLOCKED")

    with pytest.raises(ContinuationError, match="STORAGE_BLOCKED"):
        choose_auto_mode(Blocked(), "a" * 40, "run", approval)
