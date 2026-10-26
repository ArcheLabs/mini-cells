import subprocess

import pytest
from minicells_clm002.common import atomic
from minicells_clm002.seeds import refresh_registry, validate_registry


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def source(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Unit")
    git(tmp_path, "config", "user.email", "unit@example.com")
    (tmp_path / "prior.txt").write_text("old metadata\n")
    (tmp_path / "experiment").mkdir()
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "baseline")
    return tmp_path


def previous():
    return {
        "development": [810001, 810002, 810003],
        "stage_a_confirmation": [810004, 810005, 810006, 810007, 810008],
        "stage_b_confirmation": [810009, 810010, 810011, 810012, 810013],
        "scientific_training_started": False,
        "task_or_model_quality_used": False,
    }


def test_seed_scan_binds_execution_scope_without_self_referential_commit(source):
    registry = refresh_registry(previous(), source, "experiment")
    scanned = registry["scan_source_sha"]
    atomic(source / "experiment/reservations.json", registry)
    git(source, "add", ".")
    git(source, "commit", "-m", "retain evidence")
    verified = validate_registry(registry, source, "experiment")
    assert verified["execution_source_sha"] != scanned
    assert verified["source_snapshot_verified"]
    assert verified["current_tracked_file_count"] > registry["tracked_file_count_at_scan"]


def test_changed_external_scope_rejects_cached_seeds_and_collision_blocks_refresh(source):
    registry = refresh_registry(previous(), source, "experiment")
    (source / "prior.txt").write_text("historical seed " + str(previous()["development"][0]))
    git(source, "add", ".")
    git(source, "commit", "-m", "new historical use")
    with pytest.raises(RuntimeError, match="SOURCE_SCOPE_CHANGED"):
        validate_registry(registry, source, "experiment")
    with pytest.raises(RuntimeError, match="COLLISION"):
        refresh_registry(registry, source, "experiment")


def test_legacy_cache_is_not_silently_reused(source):
    with pytest.raises(RuntimeError, match="LEGACY_SEED_CACHE"):
        validate_registry(previous(), source, "experiment")


def test_changed_seed_ids_do_not_inherit_old_scan_proof(source):
    registry = refresh_registry(previous(), source, "experiment")
    registry["development"][0] += 100
    with pytest.raises(RuntimeError, match="SEED_IDS_NOT_BOUND|RESERVATION_INVALID"):
        validate_registry(registry, source, "experiment")


def test_checked_in_registry_is_verified_against_execution_tree():
    from minicells_clm002.seeds import load_registry

    registry = load_registry()
    assert registry["source_snapshot_verified"]
    assert registry["execution_source_sha"]
    assert registry["scan_scope"]["scope_sha256"]
    assert registry["scientific_training_started"] is False
