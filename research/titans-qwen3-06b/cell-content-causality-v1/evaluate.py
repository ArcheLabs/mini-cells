"""Build independent historical states and apply paired content interventions."""
import torch

from query import query, snapshot


def encode_history(tokenizer, facts, distractors):
    return [tokenizer.encode(" ".join(f"{entity} works at {value}." for entity, value in rows),
                             add_special_tokens=False)
            for rows in (facts, distractors)]


@torch.no_grad()
def build_history(model, wrapper, segments):
    device = model.get_input_embeddings().weight.device
    wrapper.reset_episode()
    offset = 0
    try:
        for ids in segments:
            if not 0 < len(ids) <= 128:
                raise ValueError(f"History segment has {len(ids)} tokens")
            inputs = torch.tensor([ids], device=device)
            positions = torch.arange(offset, offset + len(ids), device=device)[None]
            model(input_ids=inputs, position_ids=positions, use_cache=False)
            offset += len(ids)
        return snapshot(wrapper)
    finally:
        wrapper.reset_episode()


def evaluate_episode(model, wrapper, tokenizer, episode, unrelated, *, layer=22):
    def build(facts, distractors):
        return build_history(model, wrapper, encode_history(tokenizer, facts, distractors))

    correct = build(episode["facts"], episode["distractors"])
    donor = build(unrelated["facts"], unrelated["distractors"])
    rows = []
    for item in episode["queries"]:
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        prompt = encode(f"Question: {item['prompt']}\nAnswer:")
        answer = encode(" " + item["answer"])
        options = {"stop_ids": (tokenizer.eos_token_id,)}
        history = sum(encode_history(tokenizer, episode["facts"], episode["distractors"]), [])
        block = model.model.layers[layer]
        block.self_attn = wrapper.attention
        try:
            for name, context in (("E0", history + prompt), ("E1", prompt)):
                result = query(model, None, None, context, answer, **options)
                rows.append({"condition": name, "entity": item["entity"],
                             "template": item["template"], **result})
        finally:
            block.self_attn = wrapper
        conditions = {"E2": correct, "E3": (None, 0), "E4": donor, "E5": correct,
                      "E6": correct}
        if "counterfactual" in item:
            conditions["E4b"] = build(item["counterfactual_facts"], episode["distractors"])
        for name, saved in conditions.items():
            expected = encode(" " + item["counterfactual"]) if name == "E4b" else answer
            result = query(model, wrapper, saved, prompt, expected,
                           read=name != "E5", write=name == "E6", **options)
            rows.append({"condition": name, "entity": item["entity"],
                         "template": item["template"], **result})
    return rows
