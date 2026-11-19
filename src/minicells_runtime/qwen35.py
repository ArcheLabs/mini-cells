"""Pinned Qwen text adapter with recurrent-only snapshots."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import torch

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
TRANSFORMERS = '5.3.0'


@dataclass
class Session:
    cache: object
    processed: int


class Qwen35Adapter:
    def __init__(self, model, identity):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
        self.model = model.eval()
        self.identity = identity
        self.cache_type = Qwen3_5DynamicCache
        self.config = model.config

    @classmethod
    def load(cls, device='cuda', dtype=torch.float16):
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import Qwen3_5ForCausalLM
        if transformers.__version__ != TRANSFORMERS:
            raise RuntimeError(f'Install transformers=={TRANSFORMERS}')
        config = Path(hf_hub_download(MODEL_ID, 'config.json', revision=REVISION))
        model = Qwen3_5ForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=dtype, use_safetensors=True,
        ).to(device)
        return cls(model, dict(model=MODEL_ID, revision=REVISION, dtype=str(dtype),
                               config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                               transformers=TRANSFORMERS))

    @torch.no_grad()
    def encode(self, ids, state=None, *, absolute_positions=False):
        cache = self.cache_type(self.config)
        processed = 0
        if state is not None:
            processed = state.processed
            # Upstream only consumes an existing recurrent state on single-token calls.
            for name in ('conv_states', 'recurrent_states', 'key_cache', 'value_cache'):
                values = getattr(state.cache, name)
                setattr(cache, name, [None if v is None else v.clone() for v in values])
        chunks = ids.split(1, dim=1) if processed else (ids,)
        hidden = []
        position = processed
        for chunk in chunks:
            kwargs = {}
            if absolute_positions:
                kwargs['position_ids'] = torch.arange(
                    position, position + chunk.shape[1], device=ids.device)[None]
            hidden.append(self.model.model(input_ids=chunk, past_key_values=cache,
                                          use_cache=True, **kwargs).last_hidden_state)
            position += chunk.shape[1]
        return torch.cat(hidden, dim=1), Session(cache, processed + ids.shape[1])

    def logits(self, hidden):
        return self.model.lm_head(hidden)

    def export_runtime_state(self, state, path):
        from safetensors.torch import save_file
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f'{name}.{i}': tensor.detach().cpu().contiguous()
                   for name in ('conv_states', 'recurrent_states')
                   for i, tensor in enumerate(getattr(state.cache, name)) if tensor is not None}
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            stage = Path(tmp) / 'snapshot'
            stage.mkdir()
            save_file(tensors, stage / 'state.safetensors')
            manifest = dict(schema=1, identity=self.identity, processed=state.processed,
                            continuity_mode='recurrent_only', attention_cache='reset',
                            sha256=hashlib.sha256((stage / 'state.safetensors').read_bytes()).hexdigest())
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            stage.rename(path)

    def import_runtime_state(self, path):
        from safetensors.torch import load
        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        data = (path / 'state.safetensors').read_bytes()
        if (manifest['schema'] != 1 or manifest['identity'] != self.identity
                or manifest['sha256'] != hashlib.sha256(data).hexdigest()
                or type(manifest['processed']) is not int or manifest['processed'] < 1):
            raise ValueError('Invalid or incompatible runtime snapshot')
        tensors = load(data)
        # Probe the actual pinned kernel and configuration for shape and dtype.
        device = next(self.model.parameters()).device
        _, probe = self.encode(torch.zeros((1, 1), dtype=torch.long, device=device))
        cache = self.cache_type(self.config)
        names = set()
        for name in ('conv_states', 'recurrent_states'):
            for i, reference in enumerate(getattr(probe.cache, name)):
                if reference is None:
                    continue
                key = f'{name}.{i}'
                names.add(key)
                tensor = tensors.get(key)
                if (tensor is None or tensor.shape != reference.shape or tensor.dtype != reference.dtype
                        or not tensor.isfinite().all()):
                    raise ValueError(f'Invalid runtime tensor {key}')
                getattr(cache, name)[i] = tensor.to(device)
        if tensors.keys() != names:
            raise ValueError('Unexpected runtime tensors')
        return Session(cache, manifest['processed'])
