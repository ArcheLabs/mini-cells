"""Portable Kaggle Output checkpoint bundles; no network, secrets or code upload.

Only user-selected, hash-verified output is imported. The directory itself is
not a persistent store: Save Version must actually retain it as Kaggle Output.
"""

import json
import shutil
from pathlib import Path
from exp006.reuse import canonical_sha256, file_sha256, write_json

FILES = ("execution-config.json", "execution-identity.json", "assets-manifest-original.json")


def _allowed(name):
    p = Path(name)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and (name in FILES or (p.parts and p.parts[0] in ("assets", "runs")))
        and all(part not in (".git", ".env") for part in p.parts)
        and (p.suffix in (".json", ".pt", ".npy") or p.name == "tokenizer.json")
    )


def export_bundle(work, destination, source):
    work, destination = Path(work), Path(destination)
    if destination.exists():
        raise FileExistsError("immutable restore bundle already exists")
    config = json.loads((work / "execution-config.json").read_text())
    identity = json.loads((work / "execution-identity.json").read_text())
    if identity["source_sha"] != source or identity["config_sha256"] != canonical_sha256(config):
        raise ValueError("portable bundle source/config identity mismatch")
    candidates = [work / name for name in FILES if (work / name).exists()]
    for folder in ("assets", "runs"):
        candidates.extend(p for p in (work / folder).rglob("*") if p.is_file())
    selected = []
    for path in candidates:
        name = str(path.relative_to(work))
        if _allowed(name):
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != work):
                raise ValueError("symlink in portable evidence")
            selected.append((path, name))
    if not any(name.endswith("checkpoint.pt") for _, name in selected):
        raise ValueError("no resumable checkpoint to export")
    destination.mkdir(parents=True)
    hashes = {}
    for path, name in selected:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = file_sha256(target)
    manifest = {
        "format": "pd001-portable-output-v1",
        "source_sha": source,
        "config_sha256": canonical_sha256(config),
        "files": hashes,
        "scientific_decision": False,
        "persistence_scope": "local portable files; Kaggle Save Version retention must be verified by user",
    }
    write_json(destination / "restore-manifest.json", manifest)
    return manifest


def import_bundle(bundle, destination, source):
    bundle, destination = Path(bundle), Path(destination)
    if destination.exists():
        raise FileExistsError("restore must target a fresh output directory")
    manifest = json.loads((bundle / "restore-manifest.json").read_text())
    if manifest.get("format") != "pd001-portable-output-v1" or manifest["source_sha"] != source:
        raise ValueError(
            "portable output format/source mismatch; old source checkpoints cannot continue new science"
        )
    if not all(name in manifest["files"] for name in FILES[:2]):
        raise ValueError("portable config/identity missing")
    for name, digest in manifest["files"].items():
        path = bundle / name
        if (
            not _allowed(name)
            or path.is_symlink()
            or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents if parent != bundle)
            or file_sha256(path) != digest
        ):
            raise ValueError("portable file invalid or corrupt: " + name)
    config = json.loads((bundle / "execution-config.json").read_text())
    identity = json.loads((bundle / "execution-identity.json").read_text())
    if (
        canonical_sha256(config) != manifest["config_sha256"]
        or identity["config_sha256"] != manifest["config_sha256"]
        or identity["source_sha"] != source
    ):
        raise ValueError("portable config identity mismatch")
    if not any(name.endswith("checkpoint.pt") for name in manifest["files"]):
        raise ValueError("portable checkpoint missing")
    destination.mkdir(parents=True)
    for name in manifest["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / name, target)
    write_json(
        destination / "restore-receipt.json",
        {**manifest, "restored": True, "actual_kaggle_persistence_verified": False},
    )
    return config
