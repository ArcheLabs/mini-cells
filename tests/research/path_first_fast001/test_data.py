import copy
import json
import pytest
from fast001.data import Deduplicator, select_rows, load_assets, contract, grouped_windows, build
from exp006.reuse import file_sha256
from fast001.io import write_json


def test_near_and_exact_dedup():
    d = Deduplicator()
    assert d.accept("One two three four five six seven eight nine ten")
    assert not d.accept("ONE, two three four five six seven eight nine ten!")
    assert d.accept("Other distinct animals visit forests and valleys far away")


def test_stream_never_fetches_reserved_or_extra_row(tiny, stories):
    def rows():
        yield from stories
        raise AssertionError("must not request any row beyond development high")

    selected, audit = select_rows(rows(), tiny)
    assert audit["raw_rows_scanned"] == 10
    assert len(selected["train"]) == 6 and len(selected["development"]) == 4
    bad = copy.deepcopy(tiny)
    bad["data"]["development_range"] = [60000, 60001]
    with pytest.raises(ValueError, match="reserved"):
        select_rows(iter(stories), bad)


def test_shifted_story_contained_windows_and_shared_selection(tiny, assets):
    data = load_assets(tiny, assets)
    meta = data["windows"]
    assert len(meta["evaluation"]["starts"]) == 3
    for role in ("train", "development"):
        for start, group in zip(meta[role]["starts"], meta[role]["groups"]):
            assert start + 4 < meta[role]["story_offsets"][group + 1]
            assert start >= meta[role]["story_offsets"][group]
    assert contract(tiny) == contract({**tiny, "arms": copy.deepcopy(tiny["arms"])})
    s = grouped_windows(list(range(12)), [0] * 4 + [1] * 4 + [2] * 4, 3, 17)
    assert len(set(s["groups"])) == 3


def test_corrupt_and_confirmation_assets_fail(tiny, assets):
    manifest = json.loads((assets / "manifest.json").read_text())
    manifest["confirmation_materialized"] = True
    write_json(assets / "manifest.json", manifest)
    with pytest.raises(ValueError, match="contract"):
        load_assets(tiny, assets)
    manifest["confirmation_materialized"] = False
    write_json(assets / "manifest.json", manifest)
    (assets / "train.npy").write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        load_assets(tiny, assets)


def test_actual_bpe_training_only(tiny, stories, tmp_path):
    pytest.importorskip("tokenizers")
    c = copy.deepcopy(tiny)
    c["model"]["vocab_size"] = 512
    selected, audit = select_rows(iter(stories), c)
    manifest = build(c, selected, audit, tmp_path / "bpe", "e" * 64)
    assert manifest["tokenizer_training"] == "train only"
    assert manifest["files"]["tokenizer.json"] == file_sha256(tmp_path / "bpe/tokenizer.json")
    load_assets(c, tmp_path / "bpe")


def test_historical_seed_reservation_is_metadata_only(proposed):
    from fast001.config import governance, ROOT

    bad = copy.deepcopy(proposed)
    ledger = json.loads((ROOT / "research/governance/formal_seed_registry.json").read_text())
    bad["seed"] = ledger["seeds"][0]["seed"]
    with pytest.raises(ValueError, match="reserved"):
        governance(bad)
    assert governance(proposed)["raw_confirmation_boundary"] == 60000
