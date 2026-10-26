import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
ID = "CLM-002-DIFF-JOINT-VALUE-001"
LINEAGE = BASE.parent / "clm-001-qfirst-integrated"


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("CLM002_NONFINITE_JSON_" + value)

    return json.loads(Path(path).read_text(), parse_constant=reject)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def configuration():
    c = read(BASE / "config/draft.json")
    if (
        c["experiment_id"] != ID
        or sha(c) != (BASE / "config/draft.sha256").read_text().strip()
        or digest(BASE / "PROTOCOL.zh-CN.md") != c["protocol_sha256"]
    ):
        raise RuntimeError("CLM002_CONFIG_OR_ORIGINAL_PROTOCOL_CHANGED")
    for file, expected in c["source_references"].items():
        if digest(ROOT / file) != expected:
            raise RuntimeError("CLM002_HISTORICAL_SOURCE_CHANGED:" + file)
    return c


def identity(c):
    from minicells_r3mf_d1.source_identity import verify_source_tree

    source = verify_source_tree(ROOT)
    code = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((BASE / "implementation").rglob("*.py"))
    }
    code["run.py"] = digest(BASE / "run.py")
    value = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": sha(c),
        "code_sha256": sha(code),
        "scope": "GATE0_SYNTHETIC_CPU_ONLY",
        "scientific_training_authorized": False,
    }
    value["run_id"] = sha(value)[:32]
    return value


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], cwd=ROOT, text=True).strip()
