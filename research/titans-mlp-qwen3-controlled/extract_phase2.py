"""Extract frozen post-block-23 representations for a disjoint CPU pilot."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_qwen import REVISION, sha, verify_snapshot


TEMPLATES = {
    'train': ('{entity} works on project {value}.', 'Which project does {entity} work on?'),
    'validation': ('Project {value} is assigned to {entity}.', 'What assignment is listed for {entity}?'),
    'test': ('{entity} was placed on the {value} initiative.', 'Identify the initiative associated with {entity}.'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hashes = verify_snapshot(args.snapshot)
    audit = json.loads(args.audit.read_text())
    if (audit['revision'] != REVISION or not audit['frozen']
            or any(audit['hashes'].get(k) != v for k, v in hashes.items())
            or audit['hooks'][22]['output'] != [1, 4, 1024]):
        raise ValueError('Pretrained Phase 1 audit does not match the snapshot')
    tokenizer_hash = sha(args.snapshot / 'tokenizer.json')
    if audit['hashes'].get('tokenizer.json') != tokenizer_hash:
        raise ValueError('Tokenizer fingerprint mismatch')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(151643)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, dtype=torch.float32, attn_implementation='sdpa', local_files_only=True
    ).eval().requires_grad_(False)
    captured = []
    hook = model.model.layers[22].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(42)
    projections = {field: torch.randn(1024, 512, generator=generator) / 32
                   for field in ('write', 'value', 'query')}
    tensors, splits, statistics = {}, {}, {}
    for split_index, (split, (write, query)) in enumerate(TEMPLATES.items()):
        rows = []
        for index in range(8):
            entity = f'Agent {split_index * 100 + index + 10}'
            value = f'X{split_index * 100 + index + 700}'
            rows.append(dict(entity=entity, value=value, write=write.format(entity=entity, value=value),
                             query=query.format(entity=entity)))
        splits[split] = rows
        statistics[split] = {}
        for field, projection in projections.items():
            inputs = tokenizer([row[field] for row in rows], padding=True, return_tensors='pt')
            lengths = inputs['attention_mask'].sum(1)
            captured.clear()
            with torch.inference_mode():
                model.model(**inputs, use_cache=False)
                assert len(captured) == 1
                hidden = captured[0][torch.arange(len(rows)), lengths - 1].contiguous()
            assert hidden.shape == (len(rows), 1024) and torch.isfinite(hidden).all()
            projected = hidden @ projection
            tensors[f'{split}_{field}'] = hidden
            statistics[split][field] = dict(
                shape=list(hidden.shape), tokens=int(lengths.sum()),
                padding=int(inputs['attention_mask'].numel() - lengths.sum()),
                mean=float(hidden.mean()), std=float(hidden.std()),
                norm_mean=float(hidden.norm(dim=-1).mean()),
                random_projection_shape=list(projected.shape),
                random_projection_std=float(projected.std()),
            )
    hook.remove()
    data = args.output / 'dataset.json'
    data.write_text(json.dumps(splits, indent=2))
    hidden_path = args.output / 'representations.safetensors'
    save_file(tensors, str(hidden_path))
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        revision=REVISION, hashes=hashes, tokenizer_sha256=tokenizer_hash,
        dataset_sha256=sha(data), representations_sha256=sha(hidden_path),
        pooling='last non-padding token', block_index=22, statistics=statistics,
        projection='independent fixed Gaussian matrices, seed 42', training_updates=0), indent=2))


if __name__ == '__main__':
    main()
