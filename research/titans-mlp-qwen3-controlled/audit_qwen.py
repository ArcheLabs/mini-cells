"""Pinned Qwen architecture audit. Runs a single CPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch


def inspect(model):
    config = model.config
    assert config.model_type == 'qwen3' and config.architectures == ['Qwen3ForCausalLM']
    assert config.hidden_size == 1024 and config.num_hidden_layers == len(model.model.layers) == 28
    assert all(type(layer.self_attn).__name__ == 'Qwen3Attention' and
               type(layer.mlp).__name__ == 'Qwen3MLP' for layer in model.model.layers)
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            config.intermediate_size) == (16, 8, 128, 3072)
    assert not config.use_sliding_window and config._attn_implementation == 'sdpa'
    assert not any(p.requires_grad for p in model.parameters())
    calls = []

    def hook(index):
        def capture(_layer, args, kwargs, output):
            hidden = args[0] if args else kwargs['hidden_states']
            assert isinstance(output, torch.Tensor)
            calls.append(dict(index=index, input=list(hidden.shape), output=list(output.shape)))
        return capture

    for index, layer in enumerate(model.model.layers):
        assert type(layer).__name__ == 'Qwen3DecoderLayer'
        layer.register_forward_hook(hook(index), with_kwargs=True)
    with torch.inference_mode():
        result = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
        assert result.past_key_values is None
        logits = result.logits
    assert [row['index'] for row in calls] == list(range(28))
    assert all(row['input'] == row['output'] == [1, 4, 1024] for row in calls)
    assert torch.isfinite(logits).all()
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', nargs='?', default='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        parser.error('Supply an immutable Hugging Face commit SHA')
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, __version__

    torch.set_num_threads(2)
    model_id = 'Qwen/Qwen3-0.6B-Base'
    path = Path(snapshot_download(model_id, revision=args.revision,
                                  allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt']))
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                attn_implementation='sdpa').eval().requires_grad_(False)
    calls = inspect(model)
    config = model.config
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in path.iterdir() if p.name.startswith(('config.json', 'tokenizer', 'merges', 'vocab'))}
    print(json.dumps(dict(model=model_id, revision=args.revision, hashes=hashes,
                          model_type=config.model_type, architectures=config.architectures,
                          attention='Qwen3Attention/sdpa', frozen=True, device='cpu', hooks=calls,
                          torch=torch.__version__, transformers=__version__), indent=2))


if __name__ == '__main__':
    main()
