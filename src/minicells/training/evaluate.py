"""Post-hoc evaluation in a NEW directory; never overwrite observed evidence."""
import argparse
import json
import time
from pathlib import Path
import torch
from experiments.common.train import prepare_data, write_json
from minicells.models.routing_research import LanguageModel
from minicells.training.evidence import assess
from minicells.training.runner import configure_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--original-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    configure_attention()
    config, kind = saved['config'], saved['model_kind']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 600
    data = prepare_data(config, output, deadline)
    original = json.loads((Path(args.original_output) / 'data.json').read_text())
    current = json.loads((output / 'data.json').read_text())
    for key in ('dataset', 'revision', 'text_sha256', 'token_sha256', 'tokenizer_sha256'):
        if original[key] != current[key]:
            raise RuntimeError('Original data identity differs: ' + key)
    model = LanguageModel(config, kind).to('cuda')
    model.load_state_dict(saved['model'], strict=True)
    from minicells.training.stages import stage
    total = config['train_tokens'] // (config['batch_size'] * config['sequence_length'])
    phase, temperature = stage(saved['step'], total, kind)
    model.set_stage('soft' if phase == 'anneal' else phase, temperature)
    write_json(output / 'repair.json', {'checkpoint_step': saved['step'],
               'scientific_decision': False, 'scope': 'post-hoc repair, original decision unchanged'})
    assess(model, data, config, output, deadline)


if __name__ == '__main__':
    main()
