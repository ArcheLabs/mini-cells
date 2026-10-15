"""Reuse PD portable transport without claiming /working is cloud-persistent."""

from pathlib import Path
import json
from exp006.reuse import canonical_sha256
from .io import write_json
from pd001.persistence import export_bundle, import_bundle
from .config import source_sha


def export(work, destination):
    work = Path(work)
    config = json.loads((work / "execution-config.json").read_text())
    identity = {
        "experiment_id": config["experiment_id"],
        "source_sha": source_sha(),
        "config_sha256": canonical_sha256(config),
        "scientific_decision": False,
    }
    write_json(work / "execution-identity.json", identity)
    result = export_bundle(work, destination, source_sha())
    return {**result, "experiment_id": config["experiment_id"], "retain_output_required": True}


def restore(bundle, destination):
    declared = json.loads((Path(bundle) / "execution-config.json").read_text())
    if declared.get("experiment_id") != "006-PD1-FAST-001":
        raise ValueError("not FAST portable output")
    config = import_bundle(bundle, destination, source_sha())
    if config["experiment_id"] != "006-PD1-FAST-001":
        raise ValueError("not FAST portable output")
    return config
