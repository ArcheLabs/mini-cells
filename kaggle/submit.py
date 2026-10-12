"""Prepare the original MiniCells kernel with embedded source and fixed inputs."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minicells.research.config import digest, dump, load_definition


def prepare(commit, seed, destination, definition, input_reference, session=None):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A full source SHA is required')
    descriptor, config, protocol = load_definition(definition)
    if descriptor['runtime'] not in {'routing-003','routing-004'} or seed not in config['seeds']:
        raise ValueError('Only registered 003/004 seeds are enabled')
    if descriptor['runtime'] == 'routing-004' and session not in config['sessions']:
        raise ValueError('004 requires reference or graph session')
    if descriptor['runtime'] == 'routing-003' and session is not None:
        raise ValueError('003 does not take a session selector')
    reference = json.loads(Path(input_reference).read_text())
    checksum = reference['assets_manifest_sha256']
    if not re.fullmatch('[0-9a-f]{64}', checksum) or reference.get('immutable_snapshot') is not True:
        raise ValueError('Expected an immutable input snapshot manifest')
    dataset_id = reference['dataset_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/minicells-assets-[0-9a-f]{20}', dataset_id):
        raise ValueError('Unexpected Dataset identity')
    destination = Path(destination);destination.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    files = set(ROOT.glob('src/**/*.py')) | set(ROOT.glob('experiments/**/*.py'))
    files.update([Path(definition).resolve(), ROOT / descriptor['config'], protocol])
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if not path.is_relative_to(ROOT):
                raise ValueError('Bundle paths must belong to the repository')
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2020,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    source_sha = hashlib.sha256(payload).hexdigest()
    code = (ROOT / 'kaggle/templates/offline_bootstrap.py').read_text()
    replacements = {'__SOURCE_COMMIT__':commit, '__SEED__':str(seed), '__ASSET_SHA256__':checksum,
                    '__PAYLOAD_SHA256__':source_sha, '__SOURCE_PAYLOAD__':base64.b64encode(payload).decode(),
                    '__EXPERIMENT_ID__':descriptor['id']}
    for key,value in replacements.items():
        code = code.replace(key,value)
    definition_relative = str(Path(definition).resolve().relative_to(ROOT))
    notebook = json.loads((Path(definition).resolve().parent / 'notebook.ipynb').read_text())
    replacements.update({'__OFFLINE_BOOTSTRAP__': code, '__CONFIG_PATH__': descriptor['config'],
                         '__DEFINITION_PATH__': definition_relative, '__SESSION__':session or ''})
    script = []
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        # Bootstrap already contains the source payload; replace the marker first.
        text = text.replace('__OFFLINE_BOOTSTRAP__', code)
        for marker in ('__CONFIG_PATH__', '__DEFINITION_PATH__', '__SESSION__'):
            text = text.replace(marker, replacements[marker])
        cell['source'] = text.splitlines(keepends=True)
        script.append(text if cell['cell_type'] == 'code' else '\n'.join('# ' + line for line in text.splitlines()))
    notebook['metadata']['pipeline_template'] = False
    dump(destination / 'experiment.ipynb', notebook)
    (destination / 'experiment.py').write_text('\n\n'.join(script)+'\n')
    metadata=json.loads((ROOT / 'kaggle/metadata/kernel-metadata.json').read_text())
    metadata['dataset_sources']=[dataset_id]
    dump(destination / 'kernel-metadata.json', metadata)
    dump(destination / 'preparation.json', {'experiment':descriptor['id'], 'source_commit':commit,'seed':seed,'session':session,
         'definition_sha256':digest(definition), 'config_sha256':digest(ROOT / descriptor['config']),
         'protocol_sha256':digest(protocol), 'assets_manifest_sha256':checksum,'dataset_id':dataset_id,
         'payload_sha256':source_sha,'notebook_sha256':digest(destination / 'experiment.ipynb'),
         'kernel_id':'bingjiangli/minicells','application_timeout_seconds':3300,'server_timeout_seconds':3600,
         'runtime_secret_labels':[], 'scientific_decision':False})
    return destination


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--definition',default=str(ROOT / 'research/experiments/003-hard-cell-transition/experiment.yaml'))
    parser.add_argument('--input-reference',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--push',action='store_true')
    parser.add_argument('--session',choices=['reference','graph'])
    args=parser.parse_args()
    destination=prepare(args.commit,args.seed,args.output,args.definition,args.input_reference,args.session)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        result=api.kernels_push(str(destination),timeout=3600)
        if result is None or result.error:
            raise RuntimeError('Kaggle push failed: '+str(result.error if result else 'empty response'))
        dump(destination / 'push.json', {'url':result.url,'version':result.versionNumber})
        print(json.dumps({'url':result.url,'version':result.versionNumber}))
    else:
        print('Prepared offline notebook and script for the original bingjiangli/minicells kernel.')


if __name__=='__main__':
    main()
