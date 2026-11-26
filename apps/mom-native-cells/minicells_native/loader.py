"""Resolve pinned MoM model assets."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from .errors import ModelLoadError

UPSTREAM = "https://github.com/OpenSparseLLMs/MoM.git"


def load_backbone(model, device, dtype, cache_dir=None, source=None, tokenizer=None):
    import torch

    from .backend import Backbone, MOM
    from .model_registry import MODELS, snapshot_model

    if device not in ("cuda", "cuda:0") or not torch.cuda.is_available():
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
        spec = MODELS.get(str(model))
        if spec is None:
            snapshot = Path(model).expanduser().resolve()
            spec = snapshot_model(snapshot)
        else:
            snapshot = snapshot_download(spec.id, revision=spec.revision,
                                         allow_patterns=spec.checkpoint_files, cache_dir=cache / "hub")
        if tokenizer is None:
            tokenizer = snapshot if spec.tokenizer_in_snapshot else snapshot_download(
                spec.tokenizer, revision=spec.tokenizer_revision,
                allow_patterns=["*.json", "tokenizer.model"], cache_dir=cache / "hub")
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
        return Backbone(source, snapshot, tokenizer, dtype=dtype, device=device)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not open pinned MoM model: {getattr(exc, 'stderr', None) or exc}") from exc
