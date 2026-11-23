"""Resolve pinned MoM model assets."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from .errors import ModelLoadError

UPSTREAM = "https://github.com/OpenSparseLLMs/MoM.git"


def load_backbone(model, device, dtype, cache_dir=None, source=None, tokenizer=None):
    import torch

    from .backend import Backbone, MODEL, MOM, REVISION, TOKENIZER, TOKENIZER_REV

    if device != "cuda" or not torch.cuda.is_available():
        raise ModelLoadError("The current MoM runtime requires a CUDA-enabled PyTorch and CUDA GPU.")
    types = {"auto": "float16", "fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    if dtype not in types:
        raise ModelLoadError("dtype must be auto, fp16, bf16, or fp32")
    dtype = types[dtype]
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ModelLoadError("This GPU has no native BF16. Use dtype='fp16'.")
    try:
        from filelock import FileLock
        from huggingface_hub import snapshot_download

        cache = Path(cache_dir or Path.home() / ".cache" / "minicells").expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        if str(model) == MODEL:
            snapshot = snapshot_download(MODEL, revision=REVISION,
                                         allow_patterns=["config.json", "model.safetensors"],
                                         cache_dir=cache / "hub")
        else:
            snapshot = Path(model).expanduser().resolve()
            if not snapshot.is_dir() or snapshot.name != REVISION:
                raise ModelLoadError(f"Use {MODEL} or its snapshot directory named {REVISION}")
        if tokenizer is None:
            tokenizer = snapshot_download(TOKENIZER, revision=TOKENIZER_REV,
                                          allow_patterns=["*.json", "tokenizer.model"],
                                          cache_dir=cache / "hub")
        if source is None:
            source = cache / MOM
            with FileLock(str(source) + ".lock", timeout=180):
                if not source.exists():
                    with tempfile.TemporaryDirectory(dir=cache) as tmp:
                        checkout = Path(tmp)
                        for args in (("init",), ("fetch", "--depth", "1", UPSTREAM, MOM),
                                     ("checkout", "--detach", "FETCH_HEAD")):
                            subprocess.run(["git", *args], cwd=checkout, timeout=180, check=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        shutil.move(checkout, source)
        return Backbone(source, snapshot, tokenizer, dtype=dtype)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not open pinned MoM model: {getattr(exc, 'stderr', None) or exc}") from exc
