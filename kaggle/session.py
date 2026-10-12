"""Small Actions adapter: seed groups and cached input identity, no manual hashes."""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'001':'001-global-router', '002':'002-cell-internal-router',
               '003':'003-hard-cell-transition', '004':'004-cell-graph'}


def emit(values):
    with Path(os.environ['GITHUB_ENV']).open('a', encoding='utf-8') as handle:
        for name, value in values.items():
            handle.write(str(name)+'='+str(value)+'\n')


def select(experiment, group):
    definition = (Path('research/08-hybrid-clm/experiments') /
                  DIRECTORIES[experiment] / 'experiment.yaml')
    descriptor = json.loads((ROOT / definition).read_text())
    config = json.loads((ROOT / descriptor['config']).read_text())
    seeds = config.get('seeds', [config.get('seed',42)])
    # Current UI chooses a position in the protocol, so it cannot offer an invalid seed.
    if group in ('1','2','3'):
        seed = seeds[int(group)-1]
    elif int(group) in seeds:
        seed = int(group)  # Compatibility with Experiment Notebook Preparation.
    else:
        # Older UI values used 42/123/456 for every experiment. Preserve their group.
        seed = seeds[[42,123,456].index(int(group))]
    state = ROOT / '.research-input-state'
    identity = state / 'input-reference.json'
    commit_file = state / 'source-commit.txt'
    cached = identity.exists() and commit_file.exists()
    source = commit_file.read_text().strip() if cached else os.environ['GITHUB_SHA']
    emit({'DEFINITION':str(definition), 'SEED':seed, 'SOURCE_SHA':source,
          'REUSE_INPUT':'true' if cached else 'false'})
    print(json.dumps({'experiment':experiment,'seed':seed,'session':os.environ.get('SESSION'),
                      'input':'reusing prepared Dataset' if cached else 'prepare Dataset automatically'}))


def reuse():
    target=ROOT / 'prepared-assets'
    target.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / '.research-input-state/input-reference.json', target / 'input-reference.json')


def remember():
    state=ROOT / '.research-input-state'
    state.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'prepared-assets/input-reference.json', state / 'input-reference.json')
    (state / 'source-commit.txt').write_text(os.environ['SOURCE_SHA']+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['select','reuse','remember'])
    parser.add_argument('--experiment')
    parser.add_argument('--group')
    args=parser.parse_args()
    if args.action=='select':select(args.experiment,args.group)
    elif args.action=='reuse':reuse()
    else:remember()
