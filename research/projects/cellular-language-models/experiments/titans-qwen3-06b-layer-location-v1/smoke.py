"""Exercise both real attention placements with a two-segment backward pass."""
import argparse

import torch
from transformers import AutoModelForCausalLM

from execution import answer_loss, forward_episode
from integration import LAYERS, MODEL, REVISION, build_memory, install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.offline,
        torch_dtype=torch.float16, attn_implementation="eager",
    ).eval().cuda()
    reference = None
    for layer in LAYERS:
        torch.manual_seed(42)
        wrapper = install(model, layer, build_memory(args.upstream))
        adapter = {name: parameter for name, parameter in wrapper.named_parameters()
                   if parameter.requires_grad}
        current = {name: parameter.detach().cpu().clone()
                   for name, parameter in adapter.items()}
        if reference is None:
            reference = current
        else:
            assert current.keys() == reference.keys()
            assert all(torch.equal(value, reference[name]) for name, value in current.items())
        logits, targets = forward_episode(
            model, [([100, 200], [-100, -100]), ([300, 400], [-100, 400])],
            wrapper=wrapper,
        )
        loss = answer_loss(logits, targets)
        loss.backward()
        assert wrapper.position == 4 and wrapper.cell.seq_index == 4
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        invalid = [name for name, p in adapter.items()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
        if invalid:
            raise RuntimeError(f"Missing or nonfinite gradients at layer {layer}: {invalid}")
        print({"layer": layer, "answer_loss": loss.item(), "tokens": wrapper.position}, flush=True)
        model.model.layers[layer].self_attn = wrapper.attention


if __name__ == "__main__":
    main()
