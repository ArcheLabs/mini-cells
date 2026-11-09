"""Archive the exact frozen M3 bytes. No torch import, model load or training."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

BASE=Path(__file__).resolve().parent


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def checked_files(source,frozen):
    files={}
    for name,expected in frozen['expected_gpu_files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe frozen path')
        path=Path(source)/'gpu'/name
        if not path.is_file() or sha256(path)!=expected:raise ValueError('missing/mismatched frozen asset: '+name)
        files['gpu/'+name]=path
    tokenizer=BASE.parent/'assets/tokenizer.json'
    if sha256(tokenizer)!=frozen['tokenizer_sha256']:raise ValueError('frozen tokenizer differs')
    files['tokenizer.json']=tokenizer
    for path in (BASE/'freeze').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='ARCHIVE-STATUS.json':files[str(path.relative_to(BASE))]=path
    return files


def pack(source,destination,frozen):
    files=checked_files(source,frozen);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest={'format':'minicells.epclm004.m3-weight-archive.v1','run_id':frozen['run_id'],
        'script_version_id':frozen['execution']['script_version_id'],'source_commit':frozen['training_source_commit'],
        'new_training_updates':0,'files':{n:{'sha256':sha256(p),'bytes':p.stat().st_size} for n,p in files.items()}}
    payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    target=destination/frozen['archive']['asset'];pending=target.with_suffix('.partial')
    with pending.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in sorted(files):
                info=tarfile.TarInfo(name);info.size=files[name].stat().st_size;info.mode=0o644
                with files[name].open('rb') as stream:archive.addfile(info,stream)
            info=tarfile.TarInfo('archive-manifest.json');info.size=len(payload);info.mode=0o644
            archive.addfile(info,io.BytesIO(payload))
    # Re-read the package, not just the source files; never unpickle model weights.
    with tarfile.open(pending,'r:gz') as archive:
        for name,expected in manifest['files'].items():
            h=hashlib.sha256()
            with archive.extractfile(name) as stream:
                for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
            if h.hexdigest()!=expected['sha256']:raise ValueError('archive read-back mismatch: '+name)
    digest=sha256(pending)
    if target.exists():
        if sha256(target)!=digest:raise FileExistsError('existing archive differs; refusing overwrite')
        pending.unlink()
    else:pending.replace(target)
    return target,{'status':'LOCAL_ARCHIVE_READBACK_VERIFIED','sha256':digest,'bytes':target.stat().st_size,
                   'run_id':frozen['run_id'],'script_version_id':frozen['execution']['script_version_id'],
                   'new_training_updates':0,'remote_archive_verified':False}


def upload(path,receipt,frozen):
    import requests
    import sys
    root=next(p for p in BASE.parents if (p/'pyproject.toml').exists());sys.path.insert(0,str(root))
    from scripts.research.publish_experiment_results import load_github_token
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+load_github_token('GITHUB_TOKEN'),
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    repository=frozen['archive']['repository'];api='https://api.github.com/repos/'+repository
    tag=frozen['archive']['tag'];response=session.get(api+'/releases/tags/'+tag,timeout=(15,60))
    if response.status_code==404:
        response=session.post(api+'/releases',json={'tag_name':tag,'target_commitish':frozen['training_source_commit'],
            'name':'EPCLM-004 M3 frozen engineering baseline','prerelease':True,
            'body':'Exact frozen model/optimizer checkpoint and learned Cell. No additional training. H3 NOT_EVALUATED.'},timeout=(15,60))
    response.raise_for_status();release=response.json()
    # Resolve the immutable tag to the actual training source, including annotated tags.
    response=session.get(api+'/git/ref/tags/'+tag,timeout=(15,60));response.raise_for_status();obj=response.json()['object']
    if obj['type']=='tag':
        response=session.get(api+'/git/tags/'+obj['sha'],timeout=(15,60));response.raise_for_status();obj=response.json()['object']
    if obj['sha']!=frozen['training_source_commit']:raise ValueError('existing release tag points to another source')
    existing=[a for a in release['assets'] if a['name']==path.name]
    if existing:asset=existing[0]
    else:
        url=release['upload_url'].split('{',1)[0]
        with path.open('rb') as stream:
            response=session.post(url,params={'name':path.name},data=stream,
                headers={'Content-Type':'application/gzip'},timeout=(15,120))
        response.raise_for_status();asset=response.json()
    # A remote URL/size alone is insufficient. Download the uploaded bytes and hash them.
    response=session.get(api+'/releases/assets/'+str(asset['id']),headers={'Accept':'application/octet-stream'},
                         stream=True,timeout=(15,120));response.raise_for_status()
    digest=hashlib.sha256();count=0
    for part in response.iter_content(1024*1024):digest.update(part);count+=len(part)
    if digest.hexdigest()!=receipt['sha256'] or count!=receipt['bytes']:
        raise ValueError('remote archive read-back differs; no asset was deleted or overwritten')
    proof={**receipt,'status':'REMOTE_ARCHIVE_READBACK_VERIFIED','remote_archive_verified':True,
           'release_url':release['html_url'],'asset_url':asset['browser_download_url'],'asset_id':asset['id']}
    # Preserve the read-back proof beside the weights, not only in expiring Output.
    payload=(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode()
    receipts=[a for a in release['assets'] if a['name']=='archive-receipt.json']
    if receipts:proof_asset=receipts[0]
    else:
        response=session.post(release['upload_url'].split('{',1)[0],params={'name':'archive-receipt.json'},
            data=payload,headers={'Content-Type':'application/json'},timeout=(15,60))
        response.raise_for_status();proof_asset=response.json()
    response=session.get(api+'/releases/assets/'+str(proof_asset['id']),
        headers={'Accept':'application/octet-stream'},stream=True,timeout=(15,60));response.raise_for_status()
    if b''.join(response.iter_content(1024*1024))!=payload:raise ValueError('remote receipt differs; refusing overwrite')
    return proof


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--upload',action='store_true')
    args=parser.parse_args();frozen=json.loads((BASE/'freeze/BASELINE.json').read_text());started=time.monotonic()
    source=args.source
    if source is None:
        # Mounted saved output or still-live original Output only; no Kaggle recovery/history scan.
        candidates=[]
        for root in (Path('/kaggle/working'),Path('/kaggle/input')):
            if root.exists():candidates.extend(p.parents[2] for p in root.rglob('gpu/training/checkpoint.pt'))
        for candidate in candidates:
            try:checked_files(candidate,frozen)
            except ValueError:continue
            source=candidate;break
    if source is None:raise RuntimeError('FROZEN_WEIGHTS_NOT_MOUNTED: attach saved Output scriptVersionId=367010882; no training will run')
    path,receipt=pack(source,args.destination,frozen)
    if args.upload:receipt=upload(path,receipt,frozen)
    receipt['wall_seconds']=time.monotonic()-started
    (args.destination/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
