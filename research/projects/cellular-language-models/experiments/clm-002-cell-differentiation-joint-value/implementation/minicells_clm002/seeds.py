"""Bind reserved Seeds to a committed repository-wide scan scope, not a stale file cache."""

import copy
import subprocess
from pathlib import Path

from .common import BASE, ROOT, read, sha

SCHEMA = "minicells.clm002.seed_reservation.v2"
GROUPS = {"development": 3, "stage_a_confirmation": 5, "stage_b_confirmation": 5}


def _git(root, *args, check=True):
    return subprocess.run(["git", *map(str, args)], cwd=root, capture_output=True, check=check)


def snapshot(root=ROOT, excluded_prefix=None):
    root = Path(root)
    excluded_prefix = excluded_prefix or BASE.relative_to(ROOT).as_posix()
    prefix = Path(excluded_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or str(prefix) in (".", ""):
        raise ValueError("CLM002_INVALID_SEED_SCAN_EXCLUSION")
    rows = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD").stdout.split(b"\0")
    entries = []
    total = 0
    for row in rows:
        if not row:
            continue
        total += 1
        header, path = row.split(b"\t", 1)
        name = path.decode("utf-8")
        mode, kind, object_sha = header.decode("ascii").split()
        if name == str(prefix) or name.startswith(str(prefix) + "/"):
            continue
        entries.append({"path": name, "mode": mode, "kind": kind, "object_sha": object_sha})
    entries.sort(key=lambda row: row["path"])
    return {
        "source_sha": _git(root, "rev-parse", "HEAD").stdout.decode().strip(),
        "repository_tree_sha": _git(root, "rev-parse", "HEAD^{tree}").stdout.decode().strip(),
        "excluded_prefix": str(prefix),
        "scope_sha256": sha(entries),
        "scope_entry_count": len(entries),
        "tracked_file_count": total,
        "definition": "ALL_COMMITTED_REPOSITORY_ENTRIES_EXCEPT_THIS_EXPERIMENT; MODE_PATH_GIT_OBJECT_ID",
    }


def refresh_registry(previous, root=ROOT, excluded_prefix=None):
    """Rescan the same reserved IDs; a collision stops instead of selecting replacement Seeds."""
    selected = []
    for group, count in GROUPS.items():
        values = previous[group]
        if len(values) != count or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("CLM002_SEED_GROUP_INVALID")
        selected.extend(values)
    if (
        len(set(selected)) != 13
        or previous.get("scientific_training_started") is not False
        or previous.get("task_or_model_quality_used") is not False
    ):
        raise ValueError("CLM002_SEEDS_NOT_UNEXECUTED_DISJOINT_RESERVATIONS")
    before = snapshot(root, excluded_prefix)
    pattern = "(^|[^0-9])(" + "|".join(map(str, selected)) + ")([^0-9]|$)"
    hits = _git(
        root,
        "grep",
        "-I",
        "-l",
        "-E",
        pattern,
        "HEAD",
        "--",
        ".",
        ":(exclude)" + before["excluded_prefix"] + "/**",
        check=False,
    )
    if hits.returncode not in (0, 1):
        raise RuntimeError("CLM002_COMMITTED_SEED_SCAN_FAILED")
    if hits.returncode == 0:
        raise RuntimeError("CLM002_RESERVED_SEED_COLLISION_REQUIRES_REVIEW")
    after = snapshot(root, before["excluded_prefix"])
    if before != after:
        raise RuntimeError("CLM002_SOURCE_CHANGED_DURING_SEED_SCAN")
    return {
        "schema": SCHEMA,
        "status": "RESERVED_UNEXECUTED_METADATA_SEEDS_NOT_CONFIRMATION_REGISTRATION",
        "scan_source_sha": before["source_sha"],
        "scan_repository_tree_sha": before["repository_tree_sha"],
        "scan_scope": {
            k: before[k]
            for k in ("excluded_prefix", "scope_sha256", "scope_entry_count", "definition")
        },
        "tracked_file_count_at_scan": before["tracked_file_count"],
        **{group: list(previous[group]) for group in GROUPS},
        "previous_reservation_sha256": sha(previous),
        "reserved_ids_sha256": sha(selected),
        "collisions": [],
        "all_groups_disjoint": True,
        "scientific_training_started": False,
        "task_or_model_quality_used": False,
        "coverage_limits": "TRACKED_TEXT_ONLY; BINARY_CHECKPOINT_INTERNALS_AND_UNPUBLISHED_RUNS_NOT_CERTIFIED",
        "self_reference_rule": "THE_SCAN_COMMIT_PRECEDES_ITS_OWN_EVIDENCE; VALIDATE_IDENTICAL_EXTERNAL_SCOPE_AT_EACH_EXECUTION",
    }


def validate_registry(registry, root=ROOT, excluded_prefix=None):
    if registry.get("schema") != SCHEMA:
        raise RuntimeError("CLM002_LEGACY_SEED_CACHE_REQUIRES_FRESH_SCAN")
    current = snapshot(root, excluded_prefix)
    if registry["scan_scope"] != {
        k: current[k]
        for k in ("excluded_prefix", "scope_sha256", "scope_entry_count", "definition")
    }:
        raise RuntimeError("CLM002_SEED_SOURCE_SCOPE_CHANGED_RESCAN_REQUIRED")
    ids = [value for group, count in GROUPS.items() for value in registry[group]]
    if (
        any(len(registry[group]) != count for group, count in GROUPS.items())
        or len(set(ids)) != 13
        or registry.get("scientific_training_started") is not False
        or registry.get("task_or_model_quality_used") is not False
        or registry.get("collisions") != []
        or registry.get("all_groups_disjoint") is not True
    ):
        raise RuntimeError("CLM002_SEED_RESERVATION_INVALID")
    if registry.get("reserved_ids_sha256") != sha(ids):
        raise RuntimeError("CLM002_SEED_IDS_NOT_BOUND_TO_SCAN")
    value = copy.deepcopy(registry)
    value.update(
        {
            "source_snapshot_verified": True,
            "execution_source_sha": current["source_sha"],
            "execution_repository_tree_sha": current["repository_tree_sha"],
            "current_tracked_file_count": current["tracked_file_count"],
            "registration_scope": "SEED_RESERVATION_ONLY; NO_CONFIRMATION_DATA_APPROVAL",
        }
    )
    return value


def load_registry():
    return validate_registry(read(BASE / "config/SEED_REGISTRY.json"))
