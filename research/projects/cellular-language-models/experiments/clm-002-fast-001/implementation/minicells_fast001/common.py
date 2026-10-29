import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
ID = "CLM-002-FAST-001"


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def configuration():
    c = read(BASE / "config/registered.json")
    if (
        c["experiment_id"] != ID
        or sha(c) != (BASE / "config/registered.sha256").read_text().strip()
    ):
        raise RuntimeError("FAST_CONFIG_IDENTITY_CHANGED")
    if digest(BASE / "PROTOCOL.zh-CN.md") != c["protocol_sha256"]:
        raise RuntimeError("FAST_PROTOCOL_IDENTITY_CHANGED")
    return c


def identity(c, dataset_sha):
    from minicells_r3mf_d1.source_identity import verify_source_tree

    source = verify_source_tree(ROOT)
    value = {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": sha(c),
        "dataset_sha256": dataset_sha,
    }
    value["run_id"] = sha(value)[:32]
    return value


def release(c, dataset_sha):
    review = read(BASE / "release/SCIENTIFIC_REVIEW.json")
    expected = {
        "config_sha256": sha(c),
        "dataset_sha256": dataset_sha,
        "generator_sha256": digest(BASE / "implementation/minicells_fast001/data.py"),
        "implementation_sha256": implementation_identity(),
    }
    if (
        review.get("status") != "PROTOCOL_REVIEW_COMPLETE"
        or review.get("explicit_user_launch_required") is not True
        or not review.get("reviewer")
        or any(review.get(k) != v for k, v in expected.items())
    ):
        raise RuntimeError("FAST_ONE_PROTOCOL_REVIEW_REQUIRED_NO_SCIENTIFIC_TRAINING_STARTED")
    return review


@functools.lru_cache(maxsize=1)
def implementation_identity():
    """Bind executable source objects without scanning historical text or fetching blobs."""
    prefixes = [
        BASE.relative_to(ROOT).as_posix(),
        (BASE.parent / "clm-002-cell-differentiation-joint-value").relative_to(ROOT).as_posix(),
        (BASE.parent / "clm-001-qfirst-integrated").relative_to(ROOT).as_posix(),
        (BASE.parent / "jct-001").relative_to(ROOT).as_posix(),
        "src",
        "shared/python",
        (BASE.parent / "005-compositional-cell-graph/implementation").relative_to(ROOT).as_posix(),
        (BASE.parent / "006-path-first-clm").relative_to(ROOT).as_posix(),
        "research/projects/cellular-language-models/engineering/kaggle-continuation/implementation",
        "research/projects/continual-learning/context/04-continual-learning/scripts/research",
        "research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research",
    ]
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", "HEAD", "--", *prefixes], cwd=ROOT
    )
    rows = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        path = name.decode()
        if (path.endswith(".py") and "/tests/" not in path and "/registration/" not in path) or (
            "/config/" in path and path.endswith((".json", ".sha256"))
        ):
            rows.append([path, header.decode()])
    return sha(sorted(rows))


def require_ci(source):
    import urllib.request

    url = (
        "https://api.github.com/repos/ArcheLabs/mini-cells/actions/workflows/clm002-fast001.yml/runs?head_sha="
        + source
        + "&per_page=20"
    )
    headers = {"User-Agent": "MiniCells-FAST-001", "Accept": "application/vnd.github+json"}
    # Public repository: reading CI does not require broader Secret permissions.
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=20
        ) as response:
            runs = json.load(response)["workflow_runs"]
    except Exception as error:
        raise RuntimeError("FAST_EXACT_SOURCE_CI_UNAVAILABLE_NO_GPU_STARTED") from error
    matching = [r for r in runs if r["head_sha"] == source]
    if not matching or matching[0]["conclusion"] != "success":
        raise RuntimeError("FAST_EXACT_SOURCE_CI_NOT_GREEN_NO_GPU_STARTED")
    return {
        "source_sha": source,
        "run_id": matching[0]["id"],
        "url": matching[0]["html_url"],
        "conclusion": "success",
    }


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
