"""Deterministic tokenizer build/load helpers for CLM-0.4-mini M1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .protocol import canonical_json_hash, file_sha256


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
TOKENIZER_VERSION = "clm-0.4-mini-byte-bpe-v1"


class TokenizerBundle:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
        if any(value is None for value in ids.values()):
            raise ValueError("tokenizer is missing required special tokens")
        self.pad_id = int(ids[PAD])
        self.bos_id = int(ids[BOS])
        self.eos_id = int(ids[EOS])
        self.unk_id = int(ids[UNK])

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = list(self.tokenizer.encode(str(text)).ids)
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return [int(value) for value in ids]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            [int(value) for value in ids], skip_special_tokens=skip_special_tokens
        )

    def sequence(self, text: str, *, max_tokens: int) -> tuple[list[int], int]:
        """Return fixed `max_tokens` sequence and scored non-pad length."""
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2")
        ids = self.encode(text, add_special_tokens=True)[:max_tokens]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        length = len(ids)
        ids.extend([self.pad_id] * (max_tokens - len(ids)))
        return ids, length

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TokenizerBundle":
        return cls(Tokenizer.from_file(str(path)))


def _training_source_hash(texts: Iterable[str]) -> tuple[list[str], str, int]:
    values: list[str] = []
    hasher = hashlib.sha256()
    byte_count = 0
    for text in texts:
        value = str(text).replace("\r\n", "\n")
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
        byte_count += len(encoded)
        values.append(value)
    return values, hasher.hexdigest(), byte_count


def train_tokenizer(
    texts: Iterable[str],
    *,
    out_dir: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    source_manifest: dict | None = None,
) -> dict:
    """Train and persist a deterministic byte-level BPE tokenizer.

    Determinism depends on the input iterator order, which is hashed into the
    tokenizer manifest and later copied into the protocol lock.
    """
    values, source_hash, byte_count = _training_source_hash(texts)
    if not values:
        raise ValueError("tokenizer training text is empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(values, trainer=trainer, length=len(values))
    bundle = TokenizerBundle(tokenizer)
    tokenizer_path = out_dir / "tokenizer.json"
    bundle.save(tokenizer_path)
    manifest = {
        "format": "minicells.clm-0.4-mini.tokenizer-manifest.v1",
        "tokenizer_version": TOKENIZER_VERSION,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": bundle.vocab_size,
        "min_frequency": int(min_frequency),
        "special_tokens": list(SPECIAL_TOKENS),
        "special_token_ids": {
            PAD: bundle.pad_id,
            BOS: bundle.bos_id,
            EOS: bundle.eos_id,
            UNK: bundle.unk_id,
        },
        "training_text_count": len(values),
        "training_utf8_bytes": int(byte_count),
        "training_source_sha256": source_hash,
        "source_manifest": source_manifest or {},
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    (out_dir / "tokenizer-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
