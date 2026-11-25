"""Frozen MoM inference with independent native cache branches."""
from contextlib import contextmanager
from functools import wraps
import inspect
from pathlib import Path
import subprocess

import torch

from .model_assets import MOM, audit_checkpoint, expected_shapes, sha, tokenizer_hash, validate_shapes
from .model_registry import MODEL, REVISION, TOKENIZER, TOKENIZER_REV, load_tokenizer, snapshot_model

LAYERS, BANKS, HEADS, WIDTH = 24, 5, 4, 256


def validate_loaded_checkpoint(model, info, audit, dtype, device):
    if any(info.values()):
        raise ValueError(f"Weight load mismatch: {info}")
    if audit is None:
        return
    named = dict(model.named_parameters(remove_duplicate=False))
    validate_shapes({name: tuple(p.shape) for name, p in named.items()}, expected_shapes(audit["config"]))
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("The loaded embedding and LM head must share one parameter")
    parameters = list(model.parameters())
    if sum(param.numel() for param in parameters) != audit["parameter_count"]:
        raise ValueError("Loaded parameter count differs from the verified checkpoint")
    if any(p.dtype != dtype or p.device != device for p in parameters):
        raise ValueError("Loaded parameter device or dtype mismatch")


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
    def __init__(self, source, snapshot, tokenizer_snapshot, dtype="bfloat16", device="cuda"):
        import sys
        from importlib.metadata import version

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
        spec = snapshot_model(snapshot)
        if tokenizer_snapshot.name != spec.tokenizer_revision:
            raise ValueError("Use the pinned snapshot directories")
        audit = audit_checkpoint(snapshot, tokenizer_snapshot) if spec.tokenizer_in_snapshot else None
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
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(spec, tokenizer_snapshot)
        self.native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        if dtype == "bfloat16" and not self.native_bf16:
            raise RuntimeError("Select float16 explicitly for a GPU without native BF16")
        self.model, info = MomGatedDeltaNetForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=getattr(torch, dtype), output_loading_info=True)
        validate_loaded_checkpoint(self.model, info, audit, getattr(torch, dtype), torch.device("cpu"))
        config = self.model.config
        for key, value in dict(hidden_size=spec.hidden_size, num_hidden_layers=24, num_memories=4,
                               shared_mem=True, topk=2, num_heads=4, head_dim=256,
                               use_short_conv=True, conv_size=4, attn=None).items():
            if getattr(config, key) != value:
                raise ValueError(f"Unexpected config: {key}")
        if len(self.tokenizer) != config.vocab_size:
            raise ValueError("Tokenizer vocabulary mismatch")
        self.layer_count = config.num_hidden_layers
        self.bank_count = config.num_memories + int(config.shared_mem)
        self.recurrent_shape = (1, config.num_heads, config.head_dim,
                                int(config.head_dim * config.expand_v))
        self.model.to(self.device).eval().requires_grad_(False)
        files = (audit["tokenizer_files"] if audit else
                 {p.name: sha(p) for p in sorted(tokenizer_snapshot.iterdir())
                  if p.is_file() and p.name != "config.json"})
        self.identity = dict(model_id=spec.id, model_revision=spec.revision, architecture="mom-gated-deltanet",
                             mom_revision=MOM, tokenizer_id=spec.tokenizer, tokenizer_revision=spec.tokenizer_revision,
                             tokenizer_hash=tokenizer_hash(files), tokenizer_files=files,
                             config_hash=audit["config_hash"] if audit else sha(snapshot / "config.json"),
                             weights_sha256=(audit["shard_sha256_observed"]["model.safetensors"] if audit
                                             else sha(snapshot / "model.safetensors")), dtype=dtype,
                             num_hidden_layers=self.layer_count, num_memories=config.num_memories,
                             shared_memory=config.shared_mem, topk=config.topk,
                             torch=str(torch.__version__), triton=version("triton"), **expected)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def stream(self, tensors):
        if not tensors:
            return self.cache_type()
        layers = []
        for layer in range(self.layer_count):
            banks = [tensors.get(f"{layer}.{bank}") for bank in range(self.bank_count)]
            layers.append(dict(recurrent_state=[v.to(device=self.device, copy=True) if v is not None else None for v in banks],
                               conv_state=tuple([None, None] for _ in range(3)),
                               attn_state=None, ffn_state=None))
        return self.cache_type.from_legacy_cache(layers, seen_tokens=0)

    def persistent(self, cache):
        if len(cache) != self.layer_count:
            raise RuntimeError("Unexpected native cache size")
        tensors = {}
        for layer, state in enumerate(cache.to_legacy_cache()):
            banks = state["recurrent_state"]
            if len(banks) != self.bank_count:
                raise RuntimeError("Unexpected native bank count")
            for bank, tensor in enumerate(banks):
                if tensor is not None:
                    tensors[f"{layer}.{bank}"] = tensor.detach().cpu().clone()
        return tensors

    @torch.inference_mode()
    def step(self, tokens, caches, weights):
        if len(tokens) > 64 and not self.native_bf16:
            raise ValueError("The upstream Chunk kernel uses BF16. Use a BF16-capable GPU for inputs over 64 tokens.")
        ids = torch.tensor([tokens], device=self.device)
        backbone = self.model.model
        if len(caches) <= 1:
            result = backbone(input_ids=ids, past_key_values=caches[0] if caches else None,
                              use_cache=True, return_dict=True)
        else:
            with fused_attention(backbone.layers, caches, weights):
                result = backbone(input_ids=ids, past_key_values=caches[0], use_cache=True, return_dict=True)
        return result.last_hidden_state
