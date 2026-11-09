"""Gate 0 preserves existing M4 bytes; never imports or invokes M4 training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

BASE=Path(__file__).resolve().parents[1]


def archive_module():
    path=BASE.parent/'m3/archive.py';spec=importlib.util.spec_from_file_location('m5_m4_archiver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.BASE=BASE  # Existing pack/upload helpers use M5's explicit M4 preservation contract.
    return module


def verify_bundle(path,frozen):
    with tarfile.open(path,'r:gz') as archive:
        manifest=json.load(archive.extractfile('archive-manifest.json'))
        if manifest['run_id']!=frozen['run_id'] or manifest['source_commit']!=frozen['training_source_commit']:
            raise ValueError('remote M4 archive identity differs')
        for name,expected in frozen['expected_gpu_files'].items():
            digest=hashlib.sha256()
            with archive.extractfile('gpu/'+name) as stream:
                for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
            if digest.hexdigest()!=expected:raise ValueError('M4 archived payload differs: '+name)


def run(out):
    import requests
    out=Path(out);out.mkdir(parents=True,exist_ok=True);old=archive_module()
    frozen=json.loads((BASE/'freeze/BASELINE.json').read_text())
    root=next(p for p in BASE.parents if (p/'pyproject.toml').exists());sys.path.insert(0,str(root))
    from scripts.research.publish_experiment_results import load_github_token
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+load_github_token('GITHUB_TOKEN'),
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    api='https://api.github.com/repos/'+frozen['archive']['repository'];tag=frozen['archive']['tag']
    response=session.get(api+'/releases/tags/'+tag,timeout=(15,60))
    if response.status_code==404:
        response=session.post(api+'/releases',json={'tag_name':tag,'target_commitish':frozen['training_source_commit'],
            'name':'EPCLM-004 M4 frozen engineering baseline','prerelease':True,
            'body':'M4 model, optimizer and Cell preservation. No retraining. H4 NOT_EVALUATED.'},timeout=(15,60))
        response.raise_for_status()
    if response.status_code in (200,201):
        release=response.json()
        if release['target_commitish']!=frozen['training_source_commit']:raise ValueError('M4 archive release source differs')
        assets={a['name']:a for a in release['assets']}
        if frozen['archive']['asset'] in assets and 'archive-receipt.json' in assets:
            proof_asset=assets['archive-receipt.json']
            response=session.get(api+'/releases/assets/'+str(proof_asset['id']),headers={'Accept':'application/octet-stream'},timeout=(15,60))
            response.raise_for_status();proof=response.json();asset=assets[frozen['archive']['asset']]
            if proof['run_id']!=frozen['run_id'] or proof['status']!='REMOTE_ARCHIVE_READBACK_VERIFIED':raise ValueError('M4 archive receipt differs')
            path=out/frozen['archive']['asset']
            if not path.exists() or old.sha256(path)!=proof['sha256']:
                pending=path.with_suffix('.partial')
                with session.get(api+'/releases/assets/'+str(asset['id']),headers={'Accept':'application/octet-stream'},stream=True,timeout=(15,120)) as stream:
                    stream.raise_for_status()
                    with pending.open('wb') as file:
                        for part in stream.iter_content(1024*1024):file.write(part)
                if old.sha256(pending)!=proof['sha256'] or pending.stat().st_size!=proof['bytes']:raise ValueError('M4 archive download hash differs')
                pending.replace(path)
            verify_bundle(path,frozen)
            result={'status':'PASS','archive':proof,'payload_verified':True,'source_version_number':187,'new_m4_training_updates':0,'reused_remote_archive':True}
            (out/'gate0.json').write_text(json.dumps(result,indent=2)+'\n');return result
    elif response.status_code!=404:response.raise_for_status()
    config=json.loads((BASE/'archive-source.json').read_text());cache=out/'source-cache'
    # Reuse the already tested authenticated transport and exact-version inventory.
    backend,inventory,fetch=old.recovery_backend();inv=inventory(backend,config['version_number'])
    names={r['name'] for r in inv['files']};suffix='gpu/smoke.json';expected=frozen['expected_gpu_files']
    for report in sorted(n for n in names if n==suffix or n.endswith('/'+suffix)):
        prefix=report[:-len(suffix)];required={prefix+'gpu/'+n:sha for n,sha in expected.items()}
        if not set(required)<=names:continue
        if not (cache/report).exists() or old.sha256(cache/report)!=expected['smoke.json']:fetch(backend,inv,[report],cache)
        if old.sha256(cache/report)!=expected['smoke.json']:continue
        missing=[n for n,sha in required.items() if not (cache/n).exists() or old.sha256(cache/n)!=sha]
        if missing:fetch(backend,inv,missing,cache)
        path,proof=old.pack(cache/prefix,out,frozen);proof=old.upload(path,proof,frozen);verify_bundle(path,frozen)
        result={'status':'PASS','archive':proof,'payload_verified':True,'source_version_number':187,
                'source_script_version_id':inv.get('metadata',{}).get('script_version_id'),
                'new_m4_training_updates':0,'reused_remote_archive':False}
        (out/'gate0.json').write_text(json.dumps(result,indent=2)+'\n');return result
    raise RuntimeError('M4_GATE0_BLOCKED: frozen M4 payload not found in exact Version 187; no other versions or training attempted')
