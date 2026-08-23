import pytest
from minicells.vocab import CharVocab

def test_round_trip_and_pad():
    vocab=CharVocab(); assert vocab.pad_id == 0; assert vocab.decode(vocab.encode("hello jam!")) == "hello jam!"

def test_unsupported():
    with pytest.raises(ValueError, match="unsupported character"):
        CharVocab().encode("UPPER")
