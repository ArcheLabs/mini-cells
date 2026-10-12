"""Experiment 002 checkpoint repair through its frozen evaluator, in a new folder."""
import argparse
import json
import time
from pathlib import Path
import torch
from experiments.common.train import prepare_data, write_json
from experiments.cell_internal_routing_002.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--original-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['model_kind'] not in {'cell-internal', 'fixed-cell', 'global-router'}:
        raise ValueError('Only registered Experiment 002 model kinds are accepted')
    config = saved['config']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 600
    data = prepare_data(config, output, deadline)
    original = json.loads((Path(args.original_output) / 'data.json').read_text())
    current = json.loads((output / 'data.json').read_text())
    for key in ('dataset', 'revision', 'text_sha256', 'token_sha256', 'tokenizer_sha256'):
        if original[key] != current[key]:
            raise RuntimeError('Original data identity differs: ' + key)
    metrics = {'steps': saved['step'], 'train_tokens': saved['step'] * config['batch_size'] * config['sequence_length'],
               'tokens_per_sec': None}
    del saved
    evaluate_checkpoint(Path(args.checkpoint), data, output, config, deadline, metrics)
    write_json(output / 'repair.json', {'scientific_decision': False,
               'scope': 'post-hoc diagnostics; historical protocol and decisions remain unchanged'})


if __name__ == '__main__':
    main()
