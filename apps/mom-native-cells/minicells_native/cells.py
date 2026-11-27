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
from .full_state import FullState, MAX_STRUCTURE_BYTES, MAX_TENSOR_BYTES


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
        full = isinstance(tensors, FullState)
        structure = tensors.structure_bytes() if full else None
        data = save(tensors.tensors if full else tensors)
        if full:
            if len(structure) > MAX_STRUCTURE_BYTES or len(data) > MAX_TENSOR_BYTES:
                raise ValueError("Full-state Cell exceeds its schema size limit")
            metadata = {**metadata, "structure_sha256": hashlib.sha256(structure).hexdigest()}
        metadata = {**metadata, "tensor_sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("manifest.json", json.dumps(metadata))
                if full:
                    bundle.writestr("cache_structure.json", structure)
                    bundle.writestr("cache_state.safetensors", data)
                else:
                    bundle.writestr("recurrent_states.safetensors", data)
            path.replace(self.path(metadata["cell_id"]))
        finally:
            path.unlink(missing_ok=True)
        return metadata

    def decode(self, file):
        if not zipfile.is_zipfile(file):
            raise ValueError("Invalid Cell bundle")
        with zipfile.ZipFile(file) as bundle:
            if bundle.getinfo("manifest.json").file_size > 100_000:
                raise ValueError("Cell manifest too large")
            meta = json.loads(bundle.read("manifest.json"))
            version = meta["format_version"]
            if (meta["format"] != "minicells-cell" or version not in (1, 2)
                    or meta["state_schema_version"] != version):
                raise ValueError("Unsupported Cell format")
            tensor_name = "cache_state.safetensors" if version == 2 else "recurrent_states.safetensors"
            expected = ["manifest.json", tensor_name]
            if version == 2:
                expected.append("cache_structure.json")
            if sorted(bundle.namelist()) != sorted(expected):
                raise ValueError("Unexpected Cell bundle files")
            if bundle.getinfo(tensor_name).file_size > MAX_TENSOR_BYTES:
                raise ValueError("Cell bundle too large")
            if meta["backbone"] != self.backbone.identity:
                raise ValueError("Incompatible backbone or tokenizer")
            data = bundle.read(tensor_name)
            if hashlib.sha256(data).hexdigest() != meta["tensor_sha256"]:
                raise ValueError("Cell checksum mismatch")
            tensors = load(data)
            if version == 2:
                if (meta["state_completeness"] != "full_state"
                        or bundle.getinfo("cache_structure.json").file_size > MAX_STRUCTURE_BYTES):
                    raise ValueError("Invalid full-state Cell manifest")
                structure = bundle.read("cache_structure.json")
                if hashlib.sha256(structure).hexdigest() != meta["structure_sha256"]:
                    raise ValueError("Cell structure checksum mismatch")
                state = FullState(json.loads(structure), tensors)
                state.validate(self.backbone)
                # MoM advances the Cache cursor by packed routed length.
                if bool(state.structure["layer_count"]) != bool(meta["tokens_written"]):
                    raise ValueError("Cell Cache coverage conflicts with its write count")
                return meta, state
        meta["state_completeness"] = "recurrent_only"
        layers = getattr(self.backbone, "layer_count", LAYERS)
        banks = getattr(self.backbone, "bank_count", BANKS)
        shape = getattr(self.backbone, "recurrent_shape", (1, HEADS, WIDTH, WIDTH))
        allowed = {f"{i}.{j}" for i in range(layers) for j in range(banks)}
        if (set(tensors) - allowed or bool(tensors) != (meta["tokens_written"] > 0)
                or (tensors and any(f"{i}.{banks - 1}" not in tensors for i in range(layers)))):
            raise ValueError("Invalid layer or bank coverage")
        for tensor in tensors.values():
            if (tensor.shape != shape or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()):
                raise ValueError("Invalid native recurrent tensor")
        return meta, tensors

    def read(self, cell_id):
        return self.decode(self.path(cell_id))

    def create(self, name, *, full_state=False):
        now = datetime.now(timezone.utc).isoformat()
        version = 2 if full_state else 1
        state = FullState.capture(self.backbone, self.backbone.cache_type()) if full_state else {}
        return self.store(dict(format="minicells-cell", format_version=version, state_schema_version=version,
            state_completeness="full_state" if full_state else "recurrent_only",
            cell_id=str(uuid.uuid4()), name=name, revision=0, parent_revision=None, tokens_written=0,
            created_at=now, updated_at=now, backbone=self.backbone.identity), state)

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

    def save_stream(self, path, identity, cache, offset, *, full_state=False):
        full = FullState.capture(self.backbone, cache) if full_state else None
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
        for layer in (() if full else cache.to_legacy_cache()):
            if layer["attn_state"] is not None or layer["ffn_state"] is not None:
                raise ValueError("Unexpected native stream fields")
            states.append({key: encode(layer[key]) for key in ("recurrent_state", "conv_state")})
        data = save(full.tensors if full else tensors)
        receipt = dict(identity=identity, position=cache.get_seq_length(), offset=offset,
                       states=states, full_structure=full.structure if full else None,
                       structure_sha256=hashlib.sha256(full.structure_bytes()).hexdigest() if full else None,
                       sha256=hashlib.sha256(data).hexdigest())
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
        if receipt.get("full_structure") is not None:
            state = FullState(receipt["full_structure"], tensors)
            if hashlib.sha256(state.structure_bytes()).hexdigest() != receipt["structure_sha256"]:
                raise ValueError("Stream structure checksum mismatch")
            state.validate(self.backbone)
            if state.structure["sequence_length"] != receipt["position"]:
                raise ValueError("Stream cursor mismatch")
            return state.restore(self.backbone), receipt["offset"]
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
        full_state = meta["format_version"] == 2
        prior = self.load_stream(stream_path, identity) if resume and stream_path.exists() else None
        cache, offset = prior if prior else (self.backbone.stream(tensors), 0)
        if type(offset) is not int or not 0 <= offset <= len(tokens):
            raise ValueError("Invalid write checkpoint offset")
        for start in range(offset, len(tokens), chunk_size):
            if cancelled():
                return None
            end = min(start + chunk_size, len(tokens))
            self.backbone.step(tokens[start:end], [cache], [1.0])
            if resume:
                self.save_stream(stream_path, identity, cache, end, full_state=full_state)
            progress(end, len(tokens))
        if cancelled():
            return None
        state = FullState.capture(self.backbone, cache) if full_state else self.backbone.persistent(cache)
        result = self.store({**meta, "parent_revision": meta["revision"], "revision": meta["revision"] + 1,
                           "updated_at": datetime.now(timezone.utc).isoformat(),
                           "tokens_written": meta["tokens_written"] + len(tokens)},
                          state)
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
