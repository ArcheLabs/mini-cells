"""Atomic local revisions and portable Safetensors bundles."""
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import uuid
import zipfile

import torch
from safetensors.torch import load, save

from .backend import LAYERS, BANKS, HEADS, WIDTH
import hashlib

from .generation import generate


class Cells:
    def __init__(self, root, backbone):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backbone = backbone

    def path(self, cell_id):
        if str(uuid.UUID(cell_id)) != cell_id:
            raise ValueError("Invalid Cell ID")
        return self.root / (cell_id + ".zip")

    def store(self, metadata, tensors):
        data = save(tensors)
        metadata = {**metadata, "tensor_sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("manifest.json", json.dumps(metadata))
                bundle.writestr("recurrent_states.safetensors", data)
            path.replace(self.path(metadata["cell_id"]))
        finally:
            path.unlink(missing_ok=True)
        return metadata

    def decode(self, file):
        if not zipfile.is_zipfile(file):
            raise ValueError("Invalid Cell bundle")
        with zipfile.ZipFile(file) as bundle:
            if sorted(bundle.namelist()) != ["manifest.json", "recurrent_states.safetensors"]:
                raise ValueError("Unexpected Cell bundle files")
            if (bundle.getinfo("manifest.json").file_size > 100_000 or
                    bundle.getinfo("recurrent_states.safetensors").file_size > 150_000_000):
                raise ValueError("Cell bundle too large")
            meta = json.loads(bundle.read("manifest.json"))
            data = bundle.read("recurrent_states.safetensors")
        if (meta["format"], meta["format_version"], meta["state_schema_version"]) != ("minicells-cell", 1, 1):
            raise ValueError("Unsupported Cell format")
        if meta["backbone"] != self.backbone.identity:
            raise ValueError("Incompatible backbone or tokenizer")
        if hashlib.sha256(data).hexdigest() != meta["tensor_sha256"]:
            raise ValueError("Cell checksum mismatch")
        tensors = load(data)
        allowed = {f"{i}.{j}" for i in range(LAYERS) for j in range(BANKS)}
        if (set(tensors) - allowed or bool(tensors) != (meta["tokens_written"] > 0)
                or (tensors and any(f"{i}.4" not in tensors for i in range(LAYERS)))):
            raise ValueError("Invalid layer or bank coverage")
        for tensor in tensors.values():
            if (tensor.shape != (1, HEADS, WIDTH, WIDTH) or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid native recurrent tensor")
        return meta, tensors

    def read(self, cell_id):
        return self.decode(self.path(cell_id))

    def create(self, name):
        now = datetime.now(timezone.utc).isoformat()
        return self.store(dict(format="minicells-cell", format_version=1, state_schema_version=1,
            cell_id=str(uuid.uuid4()), name=name, revision=0, parent_revision=None, tokens_written=0,
            created_at=now, updated_at=now, backbone=self.backbone.identity), {})

    def fork(self, cell_id, name=None):
        meta, tensors = self.read(cell_id)
        return self.store({**meta, "cell_id": str(uuid.uuid4()), "parent_cell_id": cell_id,
                           "name": meta["name"] if name is None else name,
                           "parent_revision": meta["revision"], "revision": 0}, tensors)

    def import_bundle(self, file):
        meta, tensors = self.decode(BytesIO(file))
        self.path(meta["cell_id"])
        if self.path(meta["cell_id"]).exists():
            raise ValueError("Cell already exists. Fork before importing a second copy.")
        return self.store(meta, tensors)

    def save_stream(self, path, identity, cache, offset):
        tensors = {}
        def encode(item):
            if torch.is_tensor(item):
                key = str(len(tensors))
                tensors[key] = item.detach().cpu().contiguous().clone()
                return {"tensor": key}
            if item is None:
                return None
            if isinstance(item, (list, tuple)):
                return [encode(value) for value in item]
            raise TypeError(type(item))

        states = []
        for layer in cache.to_legacy_cache():
            if layer["attn_state"] is not None or layer["ffn_state"] is not None:
                raise ValueError("Unexpected native stream fields")
            states.append({key: encode(layer[key]) for key in ("recurrent_state", "conv_state")})
        data = save(tensors)
        receipt = dict(identity=identity, position=cache.get_seq_length(), offset=offset,
                       states=states, sha256=hashlib.sha256(data).hexdigest())
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as temp:
            staged = Path(temp.name)
        try:
            with zipfile.ZipFile(staged, "w") as bundle:
                bundle.writestr("stream.json", json.dumps(receipt))
                bundle.writestr("stream.safetensors", data)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    def load_stream(self, path, identity):
        with zipfile.ZipFile(path) as bundle:
            receipt = json.loads(bundle.read("stream.json"))
            if receipt["identity"] != identity:
                return None
            data = bundle.read("stream.safetensors")
        if hashlib.sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError("Stream checksum mismatch")
        tensors = load(data)
        def decode(item):
            if isinstance(item, dict):
                return tensors[item["tensor"]].to(self.backbone.device)
            if isinstance(item, list):
                return [decode(value) for value in item]
            return item
        layers = [{**{k:decode(v) for k,v in layer.items()}, "attn_state": None, "ffn_state": None}
                  for layer in receipt["states"]]
        return self.backbone.cache_type.from_legacy_cache(layers, seen_tokens=receipt["position"]), receipt["offset"]

    def write(self, cell_id, text, chunk_size, progress, cancelled, resume=False):
        meta, tensors = self.read(cell_id)
        tokens = self.backbone.tokenizer.encode(text, add_special_tokens=False)
        if not tokens or not 1 <= chunk_size <= 2048:
            raise ValueError("Provide nonempty text and a chunk size between 1 and 2048")
        stream_path = self.path(cell_id).with_suffix(".stream")
        identity = dict(revision=meta["revision"], cell=cell_id, chunk_size=chunk_size,
                        text=hashlib.sha256(text.encode()).hexdigest(),
                        state=meta["tensor_sha256"], backbone=self.backbone.identity)
        prior = self.load_stream(stream_path, identity) if resume and stream_path.exists() else None
        cache, offset = prior if prior else (self.backbone.stream(tensors), 0)
        for start in range(offset, len(tokens), chunk_size):
            if cancelled():
                return None
            end = min(start + chunk_size, len(tokens))
            self.backbone.step(tokens[start:end], [cache], [1.0])
            if resume:
                self.save_stream(stream_path, identity, cache, end)
            progress(end, len(tokens))
        if cancelled():
            return None
        result = self.store({**meta, "parent_revision": meta["revision"], "revision": meta["revision"] + 1,
                           "updated_at": datetime.now(timezone.utc).isoformat(),
                           "tokens_written": meta["tokens_written"] + len(tokens)},
                          self.backbone.persistent(cache))
        stream_path.unlink(missing_ok=True)
        return result

    @torch.inference_mode()
    def query(self, question, choices, max_new_tokens, temperature):
        weights = [float(c.get("weight", 1)) for c in choices]
        if (len(choices) > 4 or any(not math.isfinite(w) or w < 0 for w in weights)
                or (weights and (not math.isfinite(sum(weights)) or sum(weights) <= 0))
                or len({c["cell_id"] for c in choices}) != len(choices)):
            raise ValueError("Use up to four distinct Cells and finite nonnegative weights")
        active = [(c["cell_id"], w) for c, w in zip(choices, weights) if w > 0]
        weights = [w / sum(weights) for _, w in active]
        caches = [self.backbone.stream(self.read(c)[1]) for c, _ in active]
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not caches:
            caches = [self.backbone.stream({})]
        tok = self.backbone.tokenizer
        prompt = tok.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
        if not 1 <= max_new_tokens <= 512 or len(prompt) + max_new_tokens > 2048:
            raise ValueError("Query and generation exceed token limits")
        import time
        start = time.monotonic()
        text, answer = generate(self.backbone, prompt, lambda tokens: self.backbone.step(tokens, caches, weights),
                                max_new_tokens, temperature)
        return dict(answer=text,
                    cells_used=[c for c, _ in active], normalized_weights=weights,
                    generation_tokens=len(answer), latency_ms=(time.monotonic()-start)*1000)
