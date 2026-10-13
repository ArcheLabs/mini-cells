"""Verify returned run evidence and create an immutable archive; no checkpoints in Git."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'shared/python')]
from minicells.research.config import digest, dump


def collect(download, preparation, archive, kernel_ref, workflow_url='', collection='Actions', notebook_url=''):
    download = Path(download).resolve()
    preparation = json.loads(Path(preparation).read_text())
    matches = list(download.rglob('evidence-manifest.json'))
    if len(matches) != 1:
        raise ValueError('Expected exactly one run evidence manifest')
    source = matches[0].parent
    manifest = json.loads(matches[0].read_text())
    if preparation.get('session') is not None and manifest.get('session') != preparation['session']:
        raise ValueError('Returned session identity mismatch')
    for field in ('source_commit', 'seed', 'definition_sha256', 'config_sha256', 'protocol_sha256', 'assets_manifest_sha256'):
        if manifest.get(field) != preparation.get(field):
            raise ValueError('Returned run identity mismatch: ' + field)
    if manifest.get('scientific_decision') is not False:
        raise ValueError('Development run must not become a scientific confirmation')
    for name, checksum in manifest['files'].items():
        file = (source / name).resolve()
        if not file.is_relative_to(source) or not file.is_file() or digest(file) != checksum:
            raise ValueError('Evidence checksum mismatch: ' + name)
    for required in ('metrics.json', 'logs.jsonl', 'routing_stats.json', 'decision.json', 'config.json', 'experiment.json'):
        if required not in manifest['files']:
            raise ValueError('Missing required evidence: ' + required)
    experiment = preparation['experiment']
    registry_path = ROOT / 'research/governance/registry.yaml'
    registry = json.loads(registry_path.read_text())
    entry = next(item for item in registry['experiments'] if item['id'] + '-' + item['name'] == experiment or item['directory'].endswith('/' + experiment))
    if archive:
        archive = Path(archive)
        archive = (archive if archive.is_absolute() else ROOT / archive).resolve()
    else:
        archive = ROOT / entry['directory'] / 'artifacts'
    target = archive / preparation['source_commit'] / ('seed-' + str(preparation['seed']))
    if preparation.get('session'):
        target = target / preparation['session']
    if target.exists():
        raise FileExistsError('Observed run archive already exists; never overwrite')
    target.mkdir(parents=True)
    # Copy only files covered by the verified manifest and explicitly sized evidence.
    for name in manifest['files']:
        file = source / name
        if file.suffix == '.pt' or file.stat().st_size > 25 * 1024 * 1024:
            raise ValueError('Oversized evidence belongs in external artifacts: ' + name)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    shutil.copyfile(matches[0], target / 'evidence-manifest.json')
    context = {'kernel_ref': kernel_ref, 'source_commit': preparation['source_commit'],
               'collection': collection}
    if workflow_url:
        context['workflow_url'] = workflow_url
    if notebook_url:
        context['notebook_url'] = notebook_url
    dump(target / 'github-context.json', context)
    # Preserve original hashed plots; any host regeneration has its own derived directory.
    decision = json.loads((target / 'decision.json').read_text())
    entry.setdefault('runs', []).append({'source_commit': preparation['source_commit'],
         'seed': preparation['seed'], 'session': preparation.get('session'), 'status': decision['status'],
         'scientific_decision': False, 'archive': str(target.relative_to(ROOT))})
    entry['status'] = 'development_incomplete' if decision['status'] == 'DEVELOPMENT_INCOMPLETE' else 'development_observed'
    if experiment == '004-cell-graph':
        from minicells.research.graph_report import analyze
        matrix = analyze(target.parent.parent, target.parent.parent / 'matrix-analysis')
        entry['matrix_status'] = matrix['status']
    dump(registry_path, registry)
    print(json.dumps({'archive': str(target), 'status': decision['status'], 'scientific_decision': False}))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', required=True)
    parser.add_argument('--preparation', required=True)
    parser.add_argument('--archive', default=None)
    parser.add_argument('--kernel-ref', required=True)
    parser.add_argument('--workflow-url', default='')
    parser.add_argument('--collection', choices=('Actions', 'manual-notebook'), default='Actions')
    parser.add_argument('--notebook-url', default='')
    args = parser.parse_args()
    collect(args.download, args.preparation, args.archive, args.kernel_ref, args.workflow_url,
            args.collection, args.notebook_url)


if __name__ == '__main__':
    main()
