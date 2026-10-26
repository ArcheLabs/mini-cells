import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
ID = "JCT-001"
ARMS = ("A", "B", "C", "D")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("NONFINITE_JSON_" + value)

    return json.loads(Path(path).read_text(), parse_constant=reject)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def configuration():
    c = read(BASE / "CONFIG.json")
    if c["experiment_id"] != ID or sha(c) != (BASE / "config/CONFIG.sha256").read_text().strip():
        raise RuntimeError("JCT_CONFIGURATION_IDENTITY_CHANGED")
    if digest(BASE / "JCT001_PROTOCOL.md") != c["protocol_sha256"]:
        raise RuntimeError("JCT_ORIGINAL_PROTOCOL_CHANGED")
    for name, expected in c["source_references"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("JCT_BOUND_SOURCE_CHANGED:" + name)
    return c


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_identity(c, registry=None):
    from minicells_r3mf_d1.source_identity import verify_source_tree

    source = verify_source_tree(ROOT)
    code = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((BASE / "implementation").rglob("*.py"))
    }
    for entry in (BASE / "run.py", BASE / "gpu_run.py"):
        if entry.exists():
            code[str(entry.relative_to(ROOT))] = digest(entry)
    identity = dict(
        experiment_id=ID,
        source_sha=source,
        config_sha256=sha(c),
        source_code_sha256=sha(code),
        scope="SYNTHETIC_CPU_ENGINEERING_ONLY",
        seed=c["fixture"]["seed"],
    )
    import platform
    import importlib.metadata

    identity["runtime"] = dict(
        python=platform.python_version(),
        numpy=importlib.metadata.version("numpy"),
        torch=importlib.metadata.version("torch"),
    )
    identity["data_registry_sha256"] = sha(registry) if registry is not None else None
    identity["run_id"] = sha(identity)[:32]
    return identity
