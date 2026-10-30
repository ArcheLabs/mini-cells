import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
NATIVE = BASE.parent / "native-clm-v0/implementation"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def source():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def native():
    import sys

    if str(NATIVE) not in sys.path:
        sys.path.insert(0, str(NATIVE))
    import native_clm_phase_b as phase_b
    import native_clm_phase_c as phase_c
    import native_clm_runtime as runtime

    return runtime, phase_b, phase_c


def tensor_hash(tensor):
    import torch

    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str((list(t.shape), str(t.dtype))).encode() + t.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def model_hash(model):
    return sha({k: tensor_hash(v) for k, v in model.state_dict().items()})


def original_language_models():
    import importlib.util
    import sys

    name = "epclm_original_language_models"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "src/minicells/language_models.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def full_logits(model, tokens):
    output = model(tokens)
    return output.logits if hasattr(output, "logits") else output
