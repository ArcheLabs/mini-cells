"""Manifest-driven base-corpus generation and sharding for CLM-0.4-mini M1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np

from .curriculum import TextExample
from .model import MiniCLMConfig, StableAddressRouter
from .protocol import canonical_json_hash, file_sha256
from .tokenizer import TokenizerBundle


BASE_CORPUS_VERSION = "clm-0.4-mini-base-corpus-v1"
CATEGORY_IDS = {"language_carrier": 0, "controlled_base_math": 1, "controlled_base_story": 2}


@dataclass(frozen=True)
class BaseRecord:
    record_id: str
    category: str
    address_id: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _coverage_address_pool(router: StableAddressRouter, *, minimum_hits: int = 1) -> list[str]:
    """Find deterministic input-side addresses that cover every base Cell in layers 3/4."""
    counts = {layer: [0] * router.num_cells for layer in (3, 4)}
    addresses: list[str] = []
    candidate = 0
    while min(min(values) for values in counts.values()) < int(minimum_hits):
        address = f"base/address-{candidate:06d}"
        candidate += 1
        routes = {layer: router.route(layer, address) for layer in (3, 4)}
        useful = any(
            counts[layer][cell] < minimum_hits
            for layer in (3, 4)
            for cell in routes[layer]
        )
        if not useful:
            continue
        addresses.append(address)
        for layer in (3, 4):
            for cell in routes[layer]:
                counts[layer][cell] += 1
        if candidate > 1_000_000:
            raise RuntimeError("unable to build base-route coverage address pool")
    return addresses


def base_math_stream(seed: int = 4101) -> Iterator[str]:
    rng = random.Random(int(seed))
    index = 0
    while True:
        mode = index % 4
        if mode == 0:
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            yield f"Question: What is {a} plus {b}? Answer: {a + b}."
        elif mode == 1:
            b = rng.randint(0, 20)
            a = rng.randint(b, 40)
            yield f"Question: What is {a} minus {b}? Answer: {a - b}."
        elif mode == 2:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            relation = "greater" if a > b else "less" if a < b else "equal"
            yield f"Compare {a} and {b}. The first number is {relation} than the second."
        else:
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            yield f"Nia has {a} stones and receives {b} more. She has {a + b} stones now."
        index += 1


def base_story_stream(seed: int = 4201) -> Iterator[str]:
    rng = random.Random(int(seed))
    names = ["Ada", "Bram", "Cleo", "Dion", "Esme", "Finn", "Gia", "Hugo"]
    cities = ["Luma", "Sora", "Vela", "Neris", "Orin", "Tera"]
    jobs = ["baker", "teacher", "gardener", "painter", "librarian", "cook"]
    index = 0
    while True:
        name = names[index % len(names)]
        city = cities[rng.randrange(len(cities))]
        job = jobs[rng.randrange(len(jobs))]
        if index % 2 == 0:
            yield f"{name} lives in {city}. {name} works as a {job}."
        else:
            yield f"Question: Where does {name} live? Answer: {city}. Question: What is {name}'s job? Answer: {job}."
        index += 1


def smoke_carrier_texts() -> list[str]:
    return [
        "A small lantern glowed beside the window while the rain crossed the garden.",
        "The child packed a red book, a cup, and a folded map before walking home.",
        "At sunrise the baker opened the shop and placed warm bread on the wooden shelf.",
        "Two friends built a paper bridge and tested it with smooth stones from the river.",
        "The quiet robot learned to carry boxes from the workshop to the blue storage room.",
        "After lunch the class read a short story and talked about why the hero changed plans.",
        "A green bird landed on the fence, watched the street, and flew toward the old tower.",
        "Mila found a key under the chair and returned it to the neighbor who had lost it.",
    ]


def base_math_eval_examples(count: int = 64, seed: int = 4301) -> list[TextExample]:
    rng = random.Random(int(seed))
    result: list[TextExample] = []
    for index in range(int(count)):
        if index % 2 == 0:
            a, b = rng.randint(0, 30), rng.randint(0, 30)
            prompt, answer = f"Question: What is {a} plus {b}? Answer:", f" {a + b}."
        else:
            b = rng.randint(0, 20)
            a = rng.randint(b, 40)
            prompt, answer = f"Question: What is {a} minus {b}? Answer:", f" {a - b}."
        result.append(TextExample(f"base-math-eval:{index:03d}", "base/eval-math", prompt, answer))
    return result


def base_story_eval_examples(count: int = 64, seed: int = 4401) -> list[TextExample]:
    rng = random.Random(int(seed))
    names = ["Ada", "Bram", "Cleo", "Dion", "Esme", "Finn", "Gia", "Hugo"]
    cities = ["Luma", "Sora", "Vela", "Neris", "Orin", "Tera"]
    result: list[TextExample] = []
    for index in range(int(count)):
        name = names[index % len(names)]
        city = cities[rng.randrange(len(cities))]
        result.append(
            TextExample(
                f"base-story-eval:{index:03d}",
                "base/eval-story",
                f"Context: {name} lives in {city}. Question: Where does {name} live? Answer:",
                f" {city}.",
                knowledge_key=f"base:{name.lower()}:location:{index}",
            )
        )
    return result


def _token_segments(tokenizer: TokenizerBundle, text: str, *, max_seq_len: int) -> Iterator[list[int]]:
    raw = tokenizer.encode(text, add_special_tokens=False)
    payload = max(1, int(max_seq_len) - 1)
    if not raw:
        raw = [tokenizer.unk_id]
    for start in range(0, len(raw), payload):
        chunk = raw[start : start + payload]
        yield [tokenizer.bos_id, *chunk, tokenizer.eos_id]


class BaseShardWriter:
    """Build bounded-memory uint16 `.npy` shards plus a cryptographic manifest."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerBundle,
        model_config: MiniCLMConfig,
        out_dir: str | Path,
        target_tokens: int,
        mixture: dict[str, float],
        shard_sequences: int = 2048,
    ) -> None:
        if tokenizer.vocab_size > np.iinfo(np.uint16).max:
            raise ValueError("uint16 shard format cannot represent this vocabulary")
        self.tokenizer = tokenizer
        self.cfg = model_config
        self.out_dir = Path(out_dir)
        self.target_tokens = int(target_tokens)
        self.mixture = {str(k): float(v) for k, v in mixture.items()}
        if set(self.mixture) != set(CATEGORY_IDS):
            raise ValueError("base mixture categories do not match protocol")
        if abs(sum(self.mixture.values()) - 1.0) > 1e-9:
            raise ValueError("base mixture fractions must sum to 1")
        self.shard_sequences = int(shard_sequences)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.router = StableAddressRouter(num_cells=self.cfg.base_cells, salt=self.cfg.routing_salt)
        self.address_pool = _coverage_address_pool(self.router, minimum_hits=1)
        self.address_to_index = {value: index for index, value in enumerate(self.address_pool)}
        self._rows: list[list[int]] = []
        self._lengths: list[int] = []
        self._addresses: list[int] = []
        self._categories: list[int] = []
        self._shards: list[dict] = []
        self._shard_index = 0
        self._address_cursor = 0
        self.category_tokens = {key: 0 for key in CATEGORY_IDS}

    def _next_address(self) -> tuple[str, int]:
        address = self.address_pool[self._address_cursor % len(self.address_pool)]
        self._address_cursor += 1
        return address, self.address_to_index[address]

    def _append_sequence(self, sequence: list[int], category: str) -> None:
        if len(sequence) > self.cfg.max_seq_len + 1:
            raise ValueError("base sequence exceeds model maximum")
        array = list(sequence)
        length = len(array)
        array.extend([self.tokenizer.pad_id] * (self.cfg.max_seq_len + 1 - length))
        _, address_index = self._next_address()
        self._rows.append(array)
        self._lengths.append(length)
        self._addresses.append(address_index)
        self._categories.append(CATEGORY_IDS[category])
        self.category_tokens[category] += max(0, length - 1)
        if len(self._rows) >= self.shard_sequences:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        prefix = f"shard-{self._shard_index:05d}"
        paths = {
            "tokens": self.out_dir / f"{prefix}-tokens.npy",
            "lengths": self.out_dir / f"{prefix}-lengths.npy",
            "addresses": self.out_dir / f"{prefix}-addresses.npy",
            "categories": self.out_dir / f"{prefix}-categories.npy",
        }
        np.save(paths["tokens"], np.asarray(self._rows, dtype=np.uint16), allow_pickle=False)
        np.save(paths["lengths"], np.asarray(self._lengths, dtype=np.uint16), allow_pickle=False)
        np.save(paths["addresses"], np.asarray(self._addresses, dtype=np.uint16), allow_pickle=False)
        np.save(paths["categories"], np.asarray(self._categories, dtype=np.uint8), allow_pickle=False)
        self._shards.append(
            {
                "index": self._shard_index,
                "sequences": len(self._rows),
                "files": {
                    key: {"path": path.name, "sha256": file_sha256(path)}
                    for key, path in paths.items()
                },
            }
        )
        self._shard_index += 1
        self._rows, self._lengths, self._addresses, self._categories = [], [], [], []

    def _fill_category(self, category: str, texts: Iterable[str], target: int) -> None:
        for text in texts:
            for sequence in _token_segments(self.tokenizer, str(text), max_seq_len=self.cfg.max_seq_len):
                self._append_sequence(sequence, category)
                if self.category_tokens[category] >= target:
                    return
        raise RuntimeError(f"source exhausted before reaching token target for {category}")

    def build(
        self,
        *,
        carrier_texts: Iterable[str],
        carrier_source: dict,
        math_seed: int = 4101,
        story_seed: int = 4201,
    ) -> dict:
        targets = {
            category: int(round(self.target_tokens * fraction))
            for category, fraction in self.mixture.items()
        }
        # Infinite controlled generators are intentionally sliced by token quota.
        self._fill_category("language_carrier", carrier_texts, targets["language_carrier"])
        self._fill_category("controlled_base_math", base_math_stream(math_seed), targets["controlled_base_math"])
        self._fill_category("controlled_base_story", base_story_stream(story_seed), targets["controlled_base_story"])
        self._flush()
        address_path = self.out_dir / "address-table.json"
        address_path.write_text(json.dumps(self.address_pool, indent=2) + "\n", encoding="utf-8")
        actual_total = sum(self.category_tokens.values())
        manifest = {
            "format": "minicells.clm-0.4-mini.base-corpus-manifest.v1",
            "generator_version": BASE_CORPUS_VERSION,
            "target_tokens": self.target_tokens,
            "actual_tokens": actual_total,
            "mixture_target": self.mixture,
            "category_tokens": self.category_tokens,
            "category_fractions": {
                key: value / float(max(1, actual_total))
                for key, value in self.category_tokens.items()
            },
            "model_sequence_length": self.cfg.max_seq_len,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "routing_salt": self.cfg.routing_salt,
            "base_address_pool_size": len(self.address_pool),
            "address_table": {"path": address_path.name, "sha256": file_sha256(address_path)},
            "carrier_source": dict(carrier_source),
            "controlled_seeds": {"math": int(math_seed), "story": int(story_seed)},
            "shards": self._shards,
        }
        manifest["manifest_sha256"] = canonical_json_hash(manifest)
        (self.out_dir / "base-corpus-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def iter_hf_tinystories(
    *,
    revision: str,
    split: str = "train",
    max_examples: int | None = None,
) -> Iterator[str]:
    """Stream a pinned TinyStories revision. `datasets` is an optional dependency."""
    if not revision:
        raise ValueError("formal carrier requires an explicit dataset revision")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install mini-cells[lm] to load TinyStories") from exc
    dataset = load_dataset(
        "roneneldan/TinyStories", split=split, revision=revision, streaming=True
    )
    for index, row in enumerate(dataset):
        if max_examples is not None and index >= int(max_examples):
            break
        text = str(row.get("text", "")).strip()
        if text:
            yield text
