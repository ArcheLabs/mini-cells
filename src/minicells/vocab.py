from __future__ import annotations

from collections.abc import Sequence


class CharVocab:
    PAD = "<PAD>"
    SYMBOLS = tuple("abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-")

    def __init__(self) -> None:
        self.tokens = (self.PAD, *self.SYMBOLS)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))

    @property
    def pad_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.token_to_id[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"unsupported character: {exc.args[0]!r}") from None

    def decode(self, ids: Sequence[int], strip_pad: bool = True) -> str:
        chars: list[str] = []
        for raw_id in ids:
            idx = int(raw_id)
            if idx not in self.id_to_token:
                raise ValueError(f"unknown token id: {idx}")
            token = self.id_to_token[idx]
            if token == self.PAD and strip_pad:
                continue
            chars.append(token)
        return "".join(chars)
