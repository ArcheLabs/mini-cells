from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass(frozen=True)
class TokenizedCorpus:
    train: torch.Tensor
    validation: torch.Tensor
    tokenizer_path: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class BatchSchedule:
    starts: tuple[tuple[int, ...], ...]
    batch_size: int
    sequence_length: int
    tokens_per_step: int

    @property
    def steps(self) -> int:
        return len(self.starts)

    @property
    def consumed_tokens(self) -> int:
        return self.steps * self.tokens_per_step


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 005 requires the optional 'lm' dependencies. "
            "Install with: pip install -e '.[lm]'"
        ) from exc
    return load_dataset


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 005 requires the optional 'lm' dependencies. "
            "Install with: pip install -e '.[lm]'"
        ) from exc
    return Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer


def iter_tinystories(
    split: str,
    *,
    max_stories: int | None = None,
) -> Iterator[str]:
    load_dataset = _require_datasets()
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    yielded = 0
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        yield text.strip()
        yielded += 1
        if max_stories is not None and yielded >= max_stories:
            break


def train_bpe_tokenizer(
    output_path: Path,
    *,
    vocab_size: int = 2048,
    max_stories: int = 20_000,
) -> object:
    Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer = _require_tokenizers()
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_tinystories("train", max_stories=max_stories),
        trainer=trainer,
        length=max_stories,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def load_tokenizer(path: Path) -> object:
    Tokenizer, _, _, _, _ = _require_tokenizers()
    return Tokenizer.from_file(str(path))


def encode_story_stream(
    tokenizer: object,
    texts: Iterable[str],
    *,
    target_tokens: int,
) -> tuple[torch.Tensor, int]:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain EOS token")
    ids: list[int] = []
    stories = 0
    for text in texts:
        encoded = tokenizer.encode(text).ids
        if encoded:
            ids.extend(encoded)
            ids.append(eos_id)
            stories += 1
        if len(ids) >= target_tokens:
            break
    if len(ids) < target_tokens:
        raise RuntimeError(
            f"TinyStories stream ended at {len(ids)} tokens; required {target_tokens}"
        )
    return torch.tensor(ids[:target_tokens], dtype=torch.long), stories


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_tinystories_corpus(
    root: Path,
    *,
    vocab_size: int = 2048,
    train_stream_tokens: int = 800_000,
    validation_stream_tokens: int = 100_000,
    tokenizer_stories: int = 20_000,
) -> TokenizedCorpus:
    cache = root / "results" / "consumer-language-bridge-v1" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"

    expected_cache = {
        "vocab_size_requested": vocab_size,
        "tokenizer_training_stories": tokenizer_stories,
        "train_stream_tokens": train_stream_tokens,
        "validation_stream_tokens": validation_stream_tokens,
    }
    if tokenizer_path.exists() and train_path.exists() and validation_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected_cache.items()):
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            if (
                _tensor_sha256(train) == manifest.get("train_token_sha256")
                and _tensor_sha256(validation) == manifest.get("validation_token_sha256")
            ):
                return TokenizedCorpus(train, validation, tokenizer_path, manifest)
        for path in (tokenizer_path, train_path, validation_path, manifest_path):
            path.unlink(missing_ok=True)

    tokenizer = train_bpe_tokenizer(
        tokenizer_path,
        vocab_size=vocab_size,
        max_stories=tokenizer_stories,
    )
    train, train_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("train"),
        target_tokens=train_stream_tokens,
    )
    validation, validation_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("validation"),
        target_tokens=validation_stream_tokens,
    )
    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        "format": "minicells.tinystories-corpus.v1",
        "dataset": "roneneldan/TinyStories",
        "streaming": True,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "tokenizer_training_stories": tokenizer_stories,
        "train_stream_tokens": int(train.numel()),
        "validation_stream_tokens": int(validation.numel()),
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_token_sha256": _tensor_sha256(train),
        "validation_token_sha256": _tensor_sha256(validation),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TokenizedCorpus(train, validation, tokenizer_path, manifest)


def make_training_schedule(
    stream_length: int,
    *,
    seed: int = 5005,
    budget_tokens: int = 500_000,
    batch_size: int = 8,
    sequence_length: int = 125,
) -> BatchSchedule:
    tokens_per_step = batch_size * sequence_length
    if budget_tokens % tokens_per_step != 0:
        raise ValueError("budget_tokens must be exactly divisible by batch_size * sequence_length")
    if stream_length <= sequence_length + 1:
        raise ValueError("token stream is too short")
    steps = budget_tokens // tokens_per_step
    rng = random.Random(seed)
    high = stream_length - sequence_length - 1
    starts = tuple(
        tuple(rng.randrange(high) for _ in range(batch_size))
        for _ in range(steps)
    )
    return BatchSchedule(starts, batch_size, sequence_length, tokens_per_step)


def batch_from_starts(
    token_stream: torch.Tensor,
    starts: tuple[int, ...],
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        token_stream[start : start + sequence_length + 1]
        for start in starts
    ]
    packed = torch.stack(rows, dim=0).to(device, non_blocking=True)
    return packed[:, :-1], packed[:, 1:]


def fixed_validation_starts(
    stream_length: int,
    *,
    batches: int = 24,
    batch_size: int = 8,
    sequence_length: int = 128,
    seed: int = 5105,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    high = stream_length - sequence_length - 1
    if high <= 0:
        raise ValueError("validation stream is too short")
    return tuple(
        tuple(rng.randrange(high) for _ in range(batch_size))
        for _ in range(batches)
    )
