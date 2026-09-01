"""Real-data and frozen-foundation I/O for Core Validation 006."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .real_representation_006_config import CoreValidation006Config


@dataclass(frozen=True)
class TokenSequenceRecord:
    partition: str
    source: str
    input_ids: torch.Tensor
    document_sha256: str
    token_sha256: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "source": self.source,
            "document_sha256": self.document_sha256,
            "token_sha256": self.token_sha256,
            "tokens": int(self.input_ids.numel()),
        }


@dataclass(frozen=True)
class FrozenSequence:
    partition: str
    source: str
    hidden: torch.Tensor
    labels: torch.Tensor
    document_sha256: str
    token_sha256: str

    @property
    def tokens(self) -> int:
        return int(self.labels.numel())


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_hash(ids: list[int]) -> str:
    payload = ",".join(str(int(x)) for x in ids).encode("ascii")
    return _sha_bytes(payload)


def select_real_sequences(
    cfg: CoreValidation006Config,
    tokenizer: Any,
) -> tuple[list[TokenSequenceRecord], dict[str, Any]]:
    """Materialize a tiny deterministic, source-balanced real-text stream.

    Raw text is never written to repository/results. Reproducibility is recorded
    using the pinned dataset revision plus document/token hashes.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Core 006 real-data run requires mini-cells[lm]") from exc

    total_per_source = (
        cfg.router_bootstrap_sequences_per_source
        + cfg.train_sequences_per_source
        + cfg.eval_sequences_per_source
    )
    counts = {source: 0 for source in cfg.sources}
    records: list[TokenSequenceRecord] = []
    stream = load_dataset(
        cfg.dataset_id,
        split=cfg.dataset_split,
        streaming=True,
        revision=cfg.dataset_revision,
    )

    for row in stream:
        meta = row.get("meta") or {}
        source = meta.get("redpajama_set_name")
        if source not in counts or counts[source] >= total_per_source:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        doc_hash = _sha_bytes(text.encode("utf-8", errors="replace"))
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.sequence_length * 8,
        )["input_ids"]
        if len(encoded) < cfg.sequence_length:
            continue
        max_start = len(encoded) - cfg.sequence_length
        start = int(doc_hash[:16], 16) % (max_start + 1) if max_start else 0
        ids = [int(x) for x in encoded[start : start + cfg.sequence_length]]
        index = counts[source]
        if index < cfg.router_bootstrap_sequences_per_source:
            partition = "router"
        elif index < cfg.router_bootstrap_sequences_per_source + cfg.train_sequences_per_source:
            partition = "train"
        else:
            partition = "eval"
        records.append(
            TokenSequenceRecord(
                partition=partition,
                source=source,
                input_ids=torch.tensor(ids, dtype=torch.long),
                document_sha256=doc_hash,
                token_sha256=_token_hash(ids),
            )
        )
        counts[source] += 1
        if all(v >= total_per_source for v in counts.values()):
            break

    missing = {k: total_per_source - v for k, v in counts.items() if v < total_per_source}
    if missing:
        raise RuntimeError(f"dataset stream exhausted before quotas were filled: {missing}")

    rows = [r.manifest_row() for r in records]
    manifest_digest = _sha_bytes(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    manifest = {
        "dataset_id": cfg.dataset_id,
        "dataset_revision": cfg.dataset_revision,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "sequence_length": cfg.sequence_length,
        "sources": list(cfg.sources),
        "records": rows,
        "manifest_sha256": manifest_digest,
    }
    return records, manifest


def write_data_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_foundation(cfg: CoreValidation006Config, *, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Core 006 requires transformers; install mini-cells[lm]") from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if not hasattr(model, "gpt_neox") or not hasattr(model, "embed_out"):
        raise TypeError("Core 006 v1 expects GPTNeoXForCausalLM-compatible Pythia")
    return tokenizer, model


def extract_frozen_sequences(
    records: Iterable[TokenSequenceRecord],
    model: Any,
    *,
    device: torch.device,
    batch_size: int = 8,
) -> list[FrozenSequence]:
    records = list(records)
    out: list[FrozenSequence] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            ids = torch.stack([r.input_ids for r in batch]).to(device)
            hidden = model.gpt_neox(ids, use_cache=False, return_dict=True).last_hidden_state
            hidden = hidden[:, :-1].detach().to(device="cpu", dtype=torch.float16)
            labels = ids[:, 1:].detach().to(device="cpu", dtype=torch.long)
            for i, record in enumerate(batch):
                out.append(
                    FrozenSequence(
                        partition=record.partition,
                        source=record.source,
                        hidden=hidden[i].contiguous(),
                        labels=labels[i].contiguous(),
                        document_sha256=record.document_sha256,
                        token_sha256=record.token_sha256,
                    )
                )
    return out


def save_frozen_cache(
    sequences: list[FrozenSequence],
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest_sha256": manifest["manifest_sha256"], "sequences": sequences}, p)


def load_frozen_cache(
    manifest: dict[str, Any],
    path: str | Path,
) -> list[FrozenSequence] | None:
    p = Path(path)
    if not p.is_file():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        return None
    return payload["sequences"]
