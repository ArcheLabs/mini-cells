"""Lossless historical evidence index and explicit derived v2 views."""
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minicells.research.config import digest, dump
from minicells.research.report import generate


def migrate():
    entries = []
    families = [('002-cell-internal-routing', '002-cell-internal-router'),
                ('003-hard-cell-transition', '003-hard-cell-transition')]
    entries.append({'experiment': '001-global-router', 'status': 'historical_output_not_in_repository',
                    'configuration': 'experiments/configs/routing/experiment-001.json',
                    'note': 'Do not invent raw metrics, logs, commit identity or routing statistics from narrative.'})
    for original, canonical in families:
        for source in sorted((ROOT / 'artifacts/experiments' / original).glob('*/seed-*')):
            target = ROOT / 'research/artifacts' / canonical / 'historical' / source.parent.name / source.name
            if target.exists():
                raise FileExistsError('Historical view already exists; no overwrites')
            target.mkdir(parents=True)
            hashes = {}
            for path in source.rglob('*'):
                if not path.is_file() or path.suffix == '.pt':
                    continue
                relative = str(path.relative_to(source))
                hashes[relative] = digest(path)
            # Immutable source remains in place; the derived view records pointers
            # and hashes instead of duplicating every archived training log.
            for name in ('metrics.json', 'decision.json', 'config.json', 'experiment.json', 'data.json'):
                if (source / name).exists():
                    shutil.copyfile(source / name, target / name)
            metrics = json.loads((source / 'metrics.json').read_text())
            rows = metrics.get('models', [])
            routing = {}
            with (target / 'logs.jsonl').open('w') as combined:
                for row in rows:
                    model = row['model']
                    directory = target / model;directory.mkdir()
                    for name in ('learning_curve.json', 'routing_stats.json', 'path_length.json'):
                        if (source / model / name).exists():
                            shutil.copyfile(source / model / name, directory / name)
                    route = source / model / 'routing_stats.json'
                    routing[model] = json.loads(route.read_text()) if route.exists() else {'status':'missing'}
                    if isinstance(routing[model], list):
                        routing[model] = {'status': 'collected' if routing[model] else 'missing', 'layers': routing[model]}
                    if not routing[model].get('layers'):
                        for mode in ('native', 'soft', 'hard'):
                            historical = source / model / ('routing-' + mode + '.json')
                            if historical.exists():
                                routing[model + '/' + mode] = json.loads(historical.read_text())
                    log = source / model / 'logs.jsonl'
                    if log.exists():
                        for line in log.read_text().splitlines():
                            combined.write(json.dumps({'model':model, **json.loads(line)}) + '\n')
            dump(target / 'routing_stats.json', routing)
            dump(target / 'migration-manifest.json', {'derived':True,'source_path':str(source.relative_to(ROOT)),
                 'source_file_sha256':hashes,'policy':'Original protocol, metrics and decision remain authoritative; no missing evidence invented'})
            generate(target)
            entries.append({'experiment':canonical,'source':str(source.relative_to(ROOT)),
                            'view':str(target.relative_to(ROOT)), 'source_commit':source.parent.name,
                            'seed':int(source.name.split('-')[-1]), 'decision':json.loads((source / 'decision.json').read_text())})
    dump(ROOT / 'research/artifacts/historical-index.json', {'derived':True,'runs':entries})


if __name__ == '__main__':
    migrate()
