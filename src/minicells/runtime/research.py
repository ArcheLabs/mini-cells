"""Offline GPU orchestration: no Git, Hub downloads, Secrets, or publication."""
import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from minicells.research.config import ROOT, digest, dump, load_definition


def run(definition, assets, manifest_sha256, seed, output, source_commit, session=None):
    descriptor, config, protocol = load_definition(definition)
    if descriptor['runtime'] not in {'routing-003', 'routing-004'}:
        raise ValueError('Historical experiments are archived')
    if seed not in config['seeds']:
        raise ValueError('Seed outside registered development matrix')
    os.environ['MINICELLS_RESEARCH_CONFIG'] = str(ROOT / descriptor['config'])
    os.environ['MINICELLS_RESEARCH_PROTOCOL'] = str(protocol)
    # Workers inherit the exact declarative config and use the established runner.
    from minicells.training.runner import schedule, validate
    from minicells.utils.research_report import report
    from minicells.research.assets import load_assets
    from minicells.research.report import generate
    import torch
    config['seed'] = seed
    if descriptor['runtime'] == 'routing-004':
        from minicells.training.graph_trainer import configure
        configure(config, session)
    elif session is not None:
        raise ValueError('Session selection only applies to 004')
    capacities = validate(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = float(os.environ.get('EXPERIMENT_STARTED_AT', time.time()))
    deadline = started + min(config['session_budget_seconds'], 3300)
    provenance = {'schema_version': 2, 'experiment': descriptor['id'], 'source_commit': source_commit,
                  'seed': seed, 'session': session, 'scientific_decision': False, 'capacities': capacities,
                  'config_sha256': digest(ROOT / descriptor['config']), 'protocol_sha256': digest(protocol),
                  'definition_sha256': digest(definition), 'assets_manifest_sha256': manifest_sha256,
                  'torch': torch.__version__, 'python': sys.version,
                  'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'matplotlib')},
                  'execution': 'two independent T4 workers; no DDP; offline mounted Dataset',
                  'attention_backend': 'SDPA math in every arm and measurement', 'status': 'running'}
    dump(output / 'config.json', config)
    dump(output / 'experiment.json', provenance)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('Registered protocol requires exactly two T4 GPUs')
        names = [torch.cuda.get_device_name(i) for i in range(2)]
        if any('T4' not in name for name in names):
            raise RuntimeError('Registered platform is dual T4')
        provenance['gpus'] = names
        load_assets(assets, config, manifest_sha256)
        if time.time() >= deadline - config['publication_reserve_seconds']:
            raise TimeoutError('No GPU budget remains after verifying inputs')
        for name in ('data.json', 'tokenizer.json', 'assets-manifest.json'):
            shutil.copyfile(Path(assets) / name, output / name)
        schedule(config, Path(assets), output, deadline - config['publication_reserve_seconds'])
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc())
    # The frozen 003 gates are unchanged. Incomplete evidence is still archived.
    if descriptor['runtime'] == 'routing-004':
        from minicells.research.graph_report import summarize_session, visualize
        decision = summarize_session(output, config)
        generate = visualize
    else:
        decision = report(output)
    dump(output / 'experiment.json', {**provenance, 'status': decision['status'],
                                      'elapsed_seconds': time.time() - started})
    try:
        generate(output)
    except Exception:
        (output / 'report-error.txt').write_text(traceback.format_exc())
    checksums = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                 if p.is_file() and p.suffix != '.pt'}
    dump(output / 'evidence-manifest.json', {'source_commit': source_commit, 'seed': seed, 'session': session,
         'definition_sha256': provenance['definition_sha256'], 'config_sha256': provenance['config_sha256'],
         'protocol_sha256': provenance['protocol_sha256'], 'assets_manifest_sha256': manifest_sha256,
         'scientific_decision': False, 'files': checksums})
    print(json.dumps(decision), flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--definition', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--assets-sha256', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--session', choices=['reference','graph'])
    args = parser.parse_args()
    run(args.definition, args.assets, args.assets_sha256, args.seed, args.output, args.source_commit, session=args.session)


if __name__ == '__main__':
    main()
