"""Frozen MoM inference with independent native cache branches."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import subprocess

import torch

MODEL = "linear-moe-hub/MoM-Gated-Deltanet-340M"
REVISION = "f4081422ea2cf01b1b5d646cac05cad62a5d5438"
TOKENIZER = "fla-hub/gla-1.3B-100B"
TOKENIZER_REV = "46b15820a4df269e99aed9d709e017677c15d24b"
MOM = "125038087b71b635c171cec857bd0f21d10955b6"
LAYERS, BANKS, HEADS, WIDTH = 24, 5, 4, 256


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@contextmanager
def fused_attention(blocks, caches, weights):
    """Mix full native attention contributions before the decoder residual."""
    installed = []
    try:
        for block in blocks:
            attn = block.attn
            original = attn.forward

            def wrap(original):
                @wraps(original)
                def forward(*args, **kwargs):
                    shared = kwargs["past_key_values"]
                    mixed = None
                    for cache, weight in zip(caches, weights, strict=True):
                        values, _, _, _ = original(*args, **{**kwargs, "past_key_values": cache})
                        term = values.float() * weight
                        mixed = term if mixed is None else mixed + term
                    return mixed.to(values.dtype), None, shared, None
                return forward

            installed.append((attn, "forward" in attn.__dict__, original))
            attn.forward = wrap(original)
        yield
    finally:
        for attn, had_override, original in installed:
            if had_override:
                attn.forward = original
            else:
                del attn.forward


class Backbone:
    def __init__(self, source, snapshot, tokenizer_snapshot, dtype="bfloat16"):
        import sys
        from importlib.metadata import version
        from transformers import AutoTokenizer

        expected = {"transformers": "4.51.3", "flash-linear-attention": "0.2.2"}
        for package, pin in expected.items():
            if version(package) != pin:
                raise ValueError(f"{package} must be {pin}")
        source = Path(source).resolve()
        if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip() != MOM:
            raise ValueError("Unexpected MoM source revision")
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                    "--untracked-files=no"], text=True).strip():
            raise ValueError("MoM source checkout has tracked modifications")
        snapshot, tokenizer_snapshot = Path(snapshot), Path(tokenizer_snapshot)
        if snapshot.name != REVISION or tokenizer_snapshot.name != TOKENIZER_REV:
            raise ValueError("Use the pinned snapshot directories")
        sys.path.insert(0, str(source))
        from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM
        from mom.layers import mom_gated_deltanet
        from fla.models.utils import Cache

        kernel = mom_gated_deltanet.fused_recurrent_gated_delta_rule
        if "head_first" not in inspect.signature(kernel).parameters:
            @wraps(kernel)
            def time_major(*args, head_first=False, **kwargs):
                if head_first:
                    raise ValueError("Time-major layout required")
                return kernel(*args, **kwargs)
            mom_gated_deltanet.fused_recurrent_gated_delta_rule = time_major
        self.cache_type = Cache
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=True)
        self.native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        if dtype == "bfloat16" and not self.native_bf16:
            raise RuntimeError("Select float16 explicitly for a GPU without native BF16")
        self.model, info = MomGatedDeltaNetForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=getattr(torch, dtype), output_loading_info=True)
        if any(info.values()):
            raise ValueError(f"Weight load mismatch: {info}")
        config = self.model.config
        for key, value in dict(hidden_size=1024, num_hidden_layers=24, num_memories=4,
                               shared_mem=True, topk=2, num_heads=4, head_dim=256,
                               use_short_conv=True, conv_size=4, attn=None).items():
            if getattr(config, key) != value:
                raise ValueError(f"Unexpected config: {key}")
        if len(self.tokenizer) != config.vocab_size:
            raise ValueError("Tokenizer vocabulary mismatch")
        self.model.cuda().eval().requires_grad_(False)
        files = {p.name: sha(p) for p in sorted(tokenizer_snapshot.iterdir())
                 if p.is_file() and p.name != "config.json"}
        self.identity = dict(model_id=MODEL, model_revision=REVISION, architecture="mom-gated-deltanet",
                             mom_revision=MOM, tokenizer_id=TOKENIZER, tokenizer_revision=TOKENIZER_REV,
                             tokenizer_hash=hashlib.sha256(repr(files).encode()).hexdigest(),
                             tokenizer_files=files, config_hash=sha(snapshot/"config.json"),
                             weights_sha256=sha(snapshot/"model.safetensors"), dtype=dtype,
                             num_hidden_layers=LAYERS, num_memories=4, shared_memory=True, topk=2,
                             torch=str(torch.__version__), triton=version("triton"), **expected)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def stream(self, tensors):
        if not tensors:
            return self.cache_type()
        layers = []
        for layer in range(LAYERS):
            banks = [tensors.get(f"{layer}.{bank}") for bank in range(BANKS)]
            layers.append(dict(recurrent_state=[v.cuda().clone() if v is not None else None for v in banks],
                               conv_state=tuple([None, None] for _ in range(3)),
                               attn_state=None, ffn_state=None))
        return self.cache_type.from_legacy_cache(layers, seen_tokens=0)

    def persistent(self, cache):
        if len(cache) != LAYERS:
            raise RuntimeError("Unexpected native cache size")
        tensors = {}
        for layer, state in enumerate(cache.to_legacy_cache()):
            banks = state["recurrent_state"]
            if len(banks) != BANKS:
                raise RuntimeError("Unexpected native bank count")
            for bank, tensor in enumerate(banks):
                if tensor is not None:
                    tensors[f"{layer}.{bank}"] = tensor.detach().cpu().clone()
        return tensors

    @torch.inference_mode()
    def step(self, tokens, caches, weights):
        if len(tokens) > 64 and not self.native_bf16:
            raise ValueError("The upstream Chunk kernel uses BF16. Use a BF16-capable GPU for inputs over 64 tokens.")
        ids = torch.tensor([tokens], device="cuda")
        backbone = self.model.model
        if len(caches) <= 1:
            result = backbone(input_ids=ids, past_key_values=caches[0] if caches else None,
                              use_cache=True, return_dict=True)
        else:
            with fused_attention(backbone.layers, caches, weights):
                result = backbone(input_ids=ids, past_key_values=caches[0], use_cache=True, return_dict=True)
        return result.last_hidden_state
