from __future__ import annotations

import torch

from .metrics import edit_similarity

FIXED_SAMPLES = ("hello", "hello jam", "mini cells", "can you echo", "123456789",
                 "abc abc abc", "local neural cells")


@torch.no_grad()
def predict_text(model, vocab, text: str, device="cpu") -> dict[str, object]:
    encoded = vocab.encode(text)
    if len(encoded) > model.num_cells:
        raise ValueError(f"text has {len(encoded)} characters; maximum is {model.num_cells}")
    ids = torch.full((1, model.num_cells), vocab.pad_id, dtype=torch.long, device=device)
    ids[0, :len(encoded)] = torch.tensor(encoded, device=device)
    model.eval()
    predicted = model(ids).argmax(-1)[0, :len(encoded)].tolist()
    prediction = vocab.decode(predicted)
    return {"input": text, "prediction": prediction,
            "similarity": edit_similarity(prediction, text)}


def sample_panel(model, vocab, texts=FIXED_SAMPLES, device="cpu") -> list[dict[str, object]]:
    return [predict_text(model, vocab, text, device) for text in texts]
