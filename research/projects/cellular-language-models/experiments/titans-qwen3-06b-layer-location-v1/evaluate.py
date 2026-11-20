"""Run the paired answer-token control matrix."""
from collections import Counter

import torch

from episodes import encode
from execution import answer_loss, forward_episode


@torch.no_grad()
def evaluate(model, wrapper, layer, episode, tokenizer):
    correct = encode(episode, tokenizer)
    donor = encode(episode, tokenizer, donor=True)
    block = model.model.layers[layer]
    matrix = (
        ("E0", correct, None, {"full_history": True}),
        ("E1", correct, None, {}),
        ("E2", correct, wrapper, {}),
        ("E3", correct, wrapper, {"empty_query": True}),
        ("E4", donor, wrapper, {}),
        ("E5", correct, wrapper, {"read_enabled": False}),
        ("E6", correct, wrapper, {"query_write": False}),
    )
    results = {}
    try:
        for name, segments, memory, options in matrix:
            block.self_attn = wrapper.attention if memory is None else wrapper
            logits, targets = forward_episode(model, segments, wrapper=memory, **options)
            predictions = logits.argmax(dim=-1).tolist()
            offset = 0
            rows = []
            for _, answer, entity in episode.queries:
                ids = tokenizer.encode(f" {answer}", add_special_tokens=False)
                predicted = predictions[offset:offset + len(ids)]
                overlap = sum((Counter(ids) & Counter(predicted)).values())
                rows.append({
                    "entity": entity, "exact_match": predicted == ids,
                    "token_f1": overlap / len(ids),
                    "false_recall": predicted != ids if answer == "unknown" else None,
                    "answer_tokens": len(ids),
                    "nll": answer_loss(logits[offset:offset + len(ids)],
                                       targets[offset:offset + len(ids)]).item(),
                })
                offset += len(ids)
            if offset != len(predictions):
                raise RuntimeError("Answer spans do not match scored tokens")
            results[name] = rows
    finally:
        block.self_attn = wrapper
        wrapper.reset_episode()
    return results
