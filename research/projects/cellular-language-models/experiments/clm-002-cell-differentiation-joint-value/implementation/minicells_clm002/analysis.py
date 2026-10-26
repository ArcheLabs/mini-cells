"""Independent retained-report verification; no model load, no new optimizer updates."""

from pathlib import Path
from .common import read, digest, atomic


def analyze_existing(input_root, output):
    root = Path(input_root)
    manifest = read(root / "RUN_MANIFEST.json")
    for name, row in manifest["artifact_files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (root / relative).is_file()
            or (root / relative).is_symlink()
        ):
            raise RuntimeError("CLM002_ANALYSIS_UNSAFE_ARTIFACT")
        if (
            digest(root / relative) != row["sha256"]
            or (root / relative).stat().st_size != row["bytes"]
        ):
            raise RuntimeError("CLM002_ANALYSIS_ARTIFACT_HASH_CHANGED")
    if (
        manifest.get("scope") != "GATE0_SYNTHETIC_CPU_ONLY"
        or manifest.get("scientific_optimizer_updates") != 0
    ):
        raise RuntimeError("CLM002_ANALYSIS_UNREGISTERED_SCIENTIFIC_SCOPE")
    result = dict(
        identity=manifest["identity"],
        status="VERIFIED_GATE0_EVIDENCE_NO_SCIENTIFIC_RESULTS",
        source_sha=manifest["source_sha"],
        verified_artifact_count=len(manifest["artifact_files"]),
        decision=read(root / "DECISION.json"),
        stage_a="NOT_RUN",
        stage_b="NOT_RUN",
        optimizer_updates=0,
        model_loaded=False,
    )
    atomic(Path(output) / "READONLY_ANALYSIS.json", result)
    return result
