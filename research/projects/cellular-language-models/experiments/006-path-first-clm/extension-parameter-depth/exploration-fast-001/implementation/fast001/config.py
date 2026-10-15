"""Exactly three arms; no historical PD config is rewritten or passed to its validator."""

import copy
import json
from pathlib import Path
import subprocess
from exp006.reuse import canonical_sha256
from .io import write_json

BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ARMS = ("M0-TINY", "CLM-S8", "CLM-S16")
BASELINE_SOURCE = "91b1340"


def load(path=None):
    c = json.loads(Path(path or BASE / "config/proposed.json").read_text())
    validate(c)
    return c


def validate(c):
    if c["experiment_id"] != "006-PD1-FAST-001" or set(c["arms"]) != set(ARMS):
        raise ValueError("FAST identity/exact three-arm matrix required")
    m, t = c["model"], c["training"]
    if (
        m["memory_mode"] != "static_input_readonly"
        or m["step_capacity"] < 16
        or m["alpha"] != 0.125
        or m["moe_layers"] != 6
        or m["moe_experts"] != 4
        or m["moe_top_k"] != 2
    ):
        raise ValueError("frozen causal context/operator/baseline contract differs")
    if any(c["arms"][a]["mechanism"] != "J0" for a in ARMS):
        raise ValueError("only joint J0 is allowed")
    if [c["arms"][a]["max_actions"] for a in ARMS] != [6, 8, 16]:
        raise ValueError("fixed 6-layer MoE / S8 / S16 budgets required")
    if t["mechanism"] != "J0" or t["cost_lambda"] != 0 or t["precision"] != "float32":
        raise ValueError("no STOP, J1, cost penalty or mixed precision")
    if t["steps"] not in (50, 500) and not c.get("engineering_fixture", False):
        raise ValueError("only declared smoke or development budget")
    if t["tokens"] != t["steps"] * t["effective_batch"] * t["sequence_length"]:
        raise ValueError("target token budget mismatch")
    if t["microbatch"] not in (1, 2) or t["effective_batch"] % t["microbatch"]:
        raise ValueError("invalid frozen microbatch")
    d = c["data"]
    a, b = d["train_range"]
    x, y = d["development_range"]
    if d["split"] != "train" or not 0 <= a < b <= x < y <= min(60000, d["reserved_train_boundary"]):
        raise ValueError("reserved confirmation/raw split overlap")
    if c["approval"].get("confirmation") is not False:
        raise ValueError("confirmation is sealed")
    if c["seed_namespace"] != "006-PD1-FAST-001/development/v1":
        raise ValueError("independent FAST seed namespace required")
    if c["data"]["near_duplicate"] != "normalized-word-5gram-simhash64-hamming<=3":
        raise ValueError("unknown near-duplicate criterion")
    if not c.get("engineering_fixture") and (t["sequence_length"] != 64 or t["effective_batch"] != 8):
        raise ValueError("explicit FAST 64-token / batch8 profile required")
    governance(c)
    if c["performance"]["implementation"] != "pd001-light-v1":
        raise ValueError("light executor required")


def governance(c):
    """Read reservation metadata only; never read confirmation story/array content."""
    paths = [
        BASE.parent / "config/proposed.json",
        BASE.parent.parent / "config/proposed.json",
        BASE.parent.parent / "config/proposed-61m.json",
        BASE.parent.parent.parent / "005-compositional-cell-graph/config/registered.json",
    ]
    reserved = set()
    boundaries = []
    for path in paths:
        metadata = json.loads(path.read_text())
        reserved.update(metadata.get("seeds", {}).get("confirmation", []))
        reserved.update(metadata.get("seeds", {}).get("development", []))
        if "confirmation_range" in metadata.get("data", {}):
            boundaries.append(metadata["data"]["confirmation_range"][0])
    ledger = json.loads((ROOT / "research/governance/formal_seed_registry.json").read_text())
    reserved.update(row["seed"] for row in ledger["seeds"])
    if c["seed"] in reserved:
        raise ValueError("FAST seed collides with historical or reserved namespace")
    if c["data"]["development_range"][1] > min(boundaries):
        raise ValueError("FAST raw rows reach a historical confirmation reservation")
    return {
        "checked_seed": c["seed"],
        "historical_reserved_seed_collision": False,
        "raw_confirmation_boundary": min(boundaries),
        "reads": "configuration and seed ledger metadata only",
    }


def model_config(c, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return {
        **c["model"],
        "steps": 6,
        "budget": c["arms"][arm]["max_actions"],
        "adaptive": False,
        "self_only": False,
        "state_only": False,
    }


def engine_config(c):
    # Native MoE uses its original internal arm identifier; public identities stay FAST-only.
    return {**c, "arms": {**c["arms"], "M0": c["arms"]["M0-TINY"]}}


def shape_hash(model):
    return canonical_sha256({n: list(p.shape) for n, p in model.named_parameters()})


def counts(c):
    import torch
    from .models import make_model

    rows = {}
    for arm in ARMS:
        with torch.device("meta"):
            model = make_model(c, arm, c["seed"])
        groups = {}
        for n, p in model.named_parameters():
            key = (
                "critic"
                if ".critic." in n
                else "connection"
                if n.startswith("controller.")
                else "cell_pool"
                if n.startswith("cells.")
                else "context"
                if n.startswith(("context.", "memory."))
                else "embedding_head"
                if n.startswith("embedding.")
                else "moe_backbone"
                if n.startswith("blocks.")
                else "shared"
            )
            groups[key] = groups.get(key, 0) + p.numel()
        rows[arm] = {
            "total_params": sum(groups.values()),
            "groups": groups,
            "shape_sha256": shape_hash(model),
            "resident_weight_bytes": sum(groups.values()) * 4,
        }
    baseline = rows["M0-TINY"]["total_params"]
    for row in rows.values():
        row["ratio_to_moe"] = row["total_params"] / baseline
    if (
        rows["CLM-S8"]["total_params"] != rows["CLM-S16"]["total_params"]
        or rows["CLM-S8"]["shape_sha256"] != rows["CLM-S16"]["shape_sha256"]
    ):
        raise ValueError("CLM shape/parameter mismatch")
    if not c.get("engineering_fixture") and not 0.45 <= rows["CLM-S8"]["ratio_to_moe"] <= 0.55:
        raise ValueError("CLM resident ratio outside [0.45,0.55]")
    return rows


def search(c):
    from .models import make_model
    import torch

    result = copy.deepcopy(c)

    def number(arm):
        with torch.device("meta"):
            m = make_model(result, arm, result["seed"])
        return sum(p.numel() for p in m.parameters())

    candidates = []
    for hidden in range(64, 321, 8):
        result["model"]["moe_hidden"] = hidden
        candidates.append((abs(number("M0-TINY") - 2000000), hidden))
    result["model"]["moe_hidden"] = min(candidates)[1]
    target = number("M0-TINY") * 0.5
    candidates = []
    for hidden in range(32, 385, 8):
        result["model"]["cell_hidden"] = hidden
        candidates.append((abs(number("CLM-S8") - target), hidden))
    result["model"]["cell_hidden"] = min(candidates)[1]
    return result, counts(result)


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def authorize(c, device, mode):
    validate(c)
    if c.get("engineering_fixture"):
        if str(device) != "cpu":
            raise RuntimeError("synthetic fixture is CPU-only")
        return
    if mode == "benchmark":
        return  # CLI separately requires visible user pilot flag on CUDA.
    if not c["approval"].get("development") or c["status"] != "DEVELOPMENT_FROZEN":
        raise RuntimeError("User review required: freeze the measured development config before training")
    if c["approval"].get("configuration_sha256") != canonical_sha256(
        {k: v for k, v in c.items() if k != "approval"}
    ):
        raise RuntimeError("approved frozen configuration was changed")
    if c["approval"].get("source_sha") != source_sha():
        raise RuntimeError("approved source differs from running source")
    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            source_sha(),
            "--",
            str(BASE.relative_to(ROOT)),
            str((BASE.parent / "implementation").relative_to(ROOT)),
            str((BASE.parent.parent / "implementation").relative_to(ROOT)),
            "shared/python",
            "src",
        ],
        cwd=ROOT,
    ).returncode:
        raise RuntimeError("dirty protected source; do not train a different implementation")


def freeze(c, pilot, path, *, approve=False, accept_slow=False):
    if not approve:
        raise RuntimeError("explicit --approve-development is required")
    from .benchmark import validate_pilot

    validate_pilot(c, pilot)
    if pilot["projected_total_seconds"] > c["performance"]["target_wall_seconds"] and not accept_slow:
        raise RuntimeError("over 30-minute target; review report and explicitly accept slower execution")
    result = copy.deepcopy(c)
    result["status"] = "DEVELOPMENT_FROZEN"
    result["approval"].update(development=True, source_sha=source_sha(), pilot_sha256=canonical_sha256(pilot))
    result["approval"]["base_config_sha256"] = canonical_sha256(c)
    result["approval"]["configuration_sha256"] = canonical_sha256(
        {k: v for k, v in result.items() if k != "approval"}
    )
    path = Path(path)
    if path.exists():
        raise FileExistsError("do not replace frozen config")
    write_json(path, result)
    return result
