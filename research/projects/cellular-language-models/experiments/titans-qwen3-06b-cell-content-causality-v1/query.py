"""Isolated scoring and generation against a historical Cell snapshot."""
from copy import deepcopy

import torch
from torch.nn import functional as F


@torch.no_grad()
def snapshot(wrapper):
    return deepcopy((wrapper.cell, wrapper.position))


@torch.no_grad()
def query(model, wrapper, saved, prompt, answer, *, read=True, write=False,
          max_new_tokens=16, stop_ids=()):
    """Score an answer and generate independently from prompt tokens alone."""
    device = model.get_input_embeddings().weight.device

    def predict(tokens):
        if wrapper is not None:
            wrapper.reset_episode()
            wrapper.cell, wrapper.position = deepcopy(saved)
            wrapper.read_enabled = read
            wrapper.write_enabled = write
        inputs = torch.tensor([tokens], device=device)
        return model(input_ids=inputs, use_cache=False).logits[0].float()

    try:
        teacher = predict(prompt + answer[:-1])[len(prompt) - 1:]
        nll = F.cross_entropy(teacher, torch.tensor(answer, device=device)).item()
        generated = []
        for _ in range(max_new_tokens):
            token = int(predict(prompt + generated)[-1].argmax())
            if token in stop_ids:
                break
            generated.append(token)
        return {"answer_nll": nll, "generated_ids": generated, "expected_ids": answer,
                "greedy_exact_match": generated == answer}
    finally:
        if wrapper is not None:
            wrapper.reset_episode()
