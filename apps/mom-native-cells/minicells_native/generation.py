"""Generation over the native Backbone or FusionReader."""
import torch


@torch.inference_mode()
def generate(backbone, prompt, step, max_new_tokens, temperature):
    answer = []
    for _ in range(max_new_tokens):
        hidden = step(prompt)
        logits = backbone.model.lm_head(hidden[:, -1]).float()
        token = int(logits.argmax()) if temperature == 0 else int(
            torch.multinomial((logits / temperature).softmax(-1), 1))
        if token == backbone.tokenizer.eos_token_id:
            break
        answer.append(token)
        prompt = [token]
    return backbone.tokenizer.decode(answer, skip_special_tokens=True), answer
