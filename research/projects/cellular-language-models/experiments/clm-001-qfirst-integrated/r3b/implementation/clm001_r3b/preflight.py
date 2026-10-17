"""One preflight; cache assets/Pilot by scientific identity, never rebuild on resume."""

from pathlib import Path
import shutil
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from .common import BASE, validate


def cpu_evidence(candidate):
    report = read(BASE / "reports/cpu-gate.json")
    if report.get("status") != "PASS" or report.get("candidate_sha256") != canonical(candidate):
        raise ContinuationError("CPU_GATE_NOT_VERIFIED")
    from clm001_r3.common import ROOT

    for relative, expected in report["source_files"].items():
        if digest(ROOT / relative) != expected:
            raise ContinuationError("CPU_GATE_SOURCE_CHANGED", relative)
    return report


def engineering_evidence(path=None):
    value = read(path or BASE / "reports/engineering-acceptance.json")
    return value


def assets(candidate, destination, store, *, assets_ref=None, cached_ref=None):
    from clm001_r3.data import load_assets, prepare

    validate(candidate)
    c = candidate["engine"]
    destination = Path(destination)
    if not destination.exists():
        ref = assets_ref or cached_ref
        if ref:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                fetched = Path(tmp) / "package"
                store.fetch_verified_checkpoint(ref, fetched)
                load_assets(c, fetched / "assets")
                shutil.copytree(fetched / "assets", destination)
        else:
            prepare(c, destination)
    data = load_assets(c, destination)
    required = c["data"]["required_r1_tokenizer_sha256"]
    if data["tokenizer_sha256"] != required:
        raise ContinuationError(
            "FROZEN_R3_TOKENIZER_MISMATCH",
            "use the trusted original R3 asset bundle; never silently replace the tokenizer",
        )
    return data


def run(
    candidate,
    source,
    workspace,
    store,
    *,
    acceptance_path=None,
    assets_ref=None,
    cached_ref=None,
    publish_github=True,
):
    validate(candidate)
    cpu = cpu_evidence(candidate)
    # Formal real-data preflight cannot fabricate release from a synthetic receipt.
    accepted = engineering_evidence(acceptance_path)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    hf = store.preflight()
    run_id = "r3b-" + source[:12] + "-" + canonical(candidate)[:8]
    if publish_github:
        from .publication import preflight

        preflight(workspace, source, run_id)
    data = assets(
        candidate, workspace / "assets", store, assets_ref=assets_ref, cached_ref=cached_ref
    )
    result = {
        "status": "PREFLIGHT_PASS",
        "source_sha": source,
        "run_id": run_id,
        "candidate_sha256": canonical(candidate),
        "cpu_gate_sha256": canonical(cpu),
        "acceptance": accepted,
        "hf": hf,
        "assets_manifest_sha256": data["asset_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "formal_training_authorized": False,
        "scientific_decision": False,
    }
    atomic(workspace / "preflight.json", result)
    return result
