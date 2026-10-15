from pathlib import Path
import sys
import copy
import hashlib
import json
import random
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth/exploration-fast-001"
)
for path in (
    ROOT / "src",
    ROOT / "shared/python",
    BASE.parent.parent / "implementation",
    BASE.parent / "implementation",
    BASE / "implementation",
):
    sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


@pytest.fixture
def proposed():
    from fast001.config import load

    return load()


@pytest.fixture
def tiny(proposed):
    c = copy.deepcopy(proposed)
    c["engineering_fixture"] = True
    c["model"].update(
        width=8,
        heads=2,
        cell_count=2,
        cell_hidden=24,
        moe_hidden=8,
        vocab_size=64,
        max_sequence_length=8,
        router_hidden=8,
        step_width=4,
        edge_rank=4,
    )
    c["training"].update(
        steps=4, tokens=32, effective_batch=2, microbatch=1, sequence_length=4, warmup_steps=2
    )
    c["data"].update(train_range=[0, 6], development_range=[6, 10])
    c["evaluation"].update(
        steps=[2, 4],
        intermediate_windows=2,
        final_windows=3,
        intervention_windows=1,
        bootstrap_replicates=40,
        decode_repetitions=2,
    )
    c["performance"].update(diagnostics_interval=2, checkpoint_interval=2)
    return c


class ToyTokenizer:
    def get_vocab_size(self):
        return 64

    def token_to_id(self, name):
        return 1 if name == "<eos>" else None

    def encode(self, text):
        return SimpleNamespace(
            ids=[
                int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little") % 62 + 2
                for w in text.split()
            ]
        )

    def save(self, path):
        Path(path).write_text(json.dumps({"engineering_fixture": True, "vocab_size": 64}))


@pytest.fixture
def stories():
    rng = random.Random(1510001)
    return [
        {"text": " ".join("word" + str(rng.randrange(10000)) for _ in range(40))} for _ in range(10)
    ]


@pytest.fixture
def assets(tiny, stories, tmp_path):
    from fast001.data import select_rows, build

    selected, audit = select_rows(iter(stories), tiny)
    path = tmp_path / "assets"
    build(tiny, selected, audit, path, "e" * 64, tokenizer=ToyTokenizer())
    return path
