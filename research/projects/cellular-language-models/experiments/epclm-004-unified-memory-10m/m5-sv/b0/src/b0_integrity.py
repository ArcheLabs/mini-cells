"""Gate A identities, immutable bundles and independent remote payload verification."""
import gzip,hashlib,importlib.util,json,os,platform,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
import torch
from b0_models import EXPERIMENT,state_hash,parameter_report
from data import digest,file_hash
from e0 import ROOT,write,load_checkpoint


def publisher():
    name='_b0_existing_publisher'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/research/publish_experiment_results.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def environment():
    def command(args):
        try:return subprocess.check_output(args,text=True,stderr=subprocess.DEVNULL,timeout=15).strip()
        except Exception:return None
    gpu=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'os':platform.platform(),'cpu':platform.processor(),'cpu_count':os.cpu_count(),
        'gpu':gpu.name if gpu else None,'gpu_count':torch.cuda.device_count(),'gpu_memory_bytes':gpu.total_memory if gpu else None,
        'driver':command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),
        'system_memory':command(['free','-b']),'disk_free_bytes':shutil.disk_usage('.').free,
        'tf32':torch.backends.cuda.matmul.allow_tf32,'amp':False,'precision':'fp32',
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cuda_mapping':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_utilization':'NOT_MEASURED','idle_failure_seconds':'NOT_MEASURED'}


def verify_checkpoint(folder):
    folder=Path(folder);metrics=json.loads((folder/'metrics.json').read_text());saved=load_checkpoint(folder/'checkpoint.pt')
    if file_hash(folder/'checkpoint.pt')!=metrics['checkpoint']['sha256'] or saved['ledger']!=metrics['ledger']:
        raise ValueError('LEDGER_CHECKPOINT_MISMATCH')
    if len(saved['history'])!=saved['ledger']['committed_updates']:raise ValueError('LEDGER_CHECKPOINT_MISMATCH')
    return saved


def freeze_bundle(folder):
    folder=Path(folder);files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file()
        and p.name not in ('sha256-manifest.json','publication.json','bundle.tar.gz') and not p.name.endswith('.pending')}
    path=folder/'sha256-manifest.json'
    if path.exists() and json.loads(path.read_text())!=files:raise ValueError('INTEGRITY_FAILED: immutable run changed')
    write(path,files);target=folder/'bundle.tar.gz'
    if not target.exists():
        pending=folder/'bundle.pending'
        with pending.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
            for name in sorted([*files,'sha256-manifest.json']):
                info=tarfile.TarInfo(name);info.size=(folder/name).stat().st_size;info.mode=0o644
                with (folder/name).open('rb') as stream:archive.addfile(info,stream)
        pending.replace(target)
    verify_bundle(target,files);return target,files


def verify_bundle(package,expected):
    with tarfile.open(package,'r:gz') as archive:
        if set(archive.getnames())!=set(expected)|{'sha256-manifest.json'}:raise ValueError('INTEGRITY_FAILED: bundle inventory differs')
        if json.load(archive.extractfile('sha256-manifest.json'))!=expected:raise ValueError('INTEGRITY_FAILED: embedded manifest differs')
        for name,sha in expected.items():
            h=hashlib.sha256()
            with archive.extractfile(name) as stream:
                for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
            if h.hexdigest()!=sha:raise ValueError('INTEGRITY_FAILED: member differs: '+name)


def publish_bundle(folder):
    """Retries never invoke models; Release assets are retained without expiry promises."""
    import requests
    folder=Path(folder);manifest=json.loads((folder/'run-manifest.json').read_text());package,files=freeze_bundle(folder)
    expected=file_hash(package);session=requests.Session();session.headers.update({'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN'),
        'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    api='https://api.github.com/repos/ArcheLabs/mini-cells';tag=manifest['run_id']
    response=session.get(api+'/releases/tags/'+tag,timeout=(15,60))
    if response.status_code==404:
        response=session.post(api+'/releases',json={'tag_name':tag,'target_commitish':manifest['source_commit'],
            'name':'EPCLM-004 B0 '+manifest['phase']+' '+manifest['arm'],'prerelease':True,
            'body':'Immutable B0 execution bundle. Smoke is engineering evidence only; no H5 confirmation.'},timeout=(15,60))
    response.raise_for_status();release=response.json()
    response=session.get(api+'/git/ref/tags/'+tag,timeout=(15,60));response.raise_for_status();obj=response.json()['object']
    if obj['type']=='tag':
        response=session.get(api+'/git/tags/'+obj['sha'],timeout=(15,60));response.raise_for_status();obj=response.json()['object']
    if obj['sha']!=manifest['source_commit']:raise ValueError('release tag source differs')
    assets=[a for a in release['assets'] if a['name']==package.name]
    if assets:asset=assets[0]
    else:
        with package.open('rb') as stream:response=session.post(release['upload_url'].split('{')[0],params={'name':package.name},data=stream,headers={'Content-Type':'application/gzip'},timeout=(15,300))
        response.raise_for_status();asset=response.json()
    with tempfile.TemporaryDirectory(prefix='b0-readback-') as temp:
        downloaded=Path(temp)/'bundle.tar.gz'
        with session.get(api+'/releases/assets/'+str(asset['id']),headers={'Accept':'application/octet-stream'},stream=True,timeout=(15,300)) as response:
            response.raise_for_status()
            with downloaded.open('wb') as stream:
                for block in response.iter_content(1024*1024):stream.write(block)
        if file_hash(downloaded)!=expected:raise ValueError('INTEGRITY_FAILED: remote bundle hash differs')
        verify_bundle(downloaded,files)
    proof={'publication_status':'VERIFIED','run_id':tag,'artifact_id':asset['id'],'download_url':asset['browser_download_url'],
        'release_url':release['html_url'],'sha256':expected,'bytes':package.stat().st_size,'files_verified':len(files),
        'retention':'GitHub Release; retained until explicitly removed, not an expiring Actions artifact'}
    write(folder/'publication.json',proof);return proof


def publish_index(manifest,proof):
    """Main-derived metadata branch; exact source fetched inside the isolated clone."""
    pub=publisher()
    schema='minicells.b0.overfit-index.v1' if manifest['phase']=='overfit' else 'minicells.b0.gate-a-index.v1'
    with tempfile.TemporaryDirectory(prefix='b0-index-') as temp:
        root=Path(temp)/'repo';subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
        subprocess.run(['git','remote','set-url','origin',origin],cwd=root,check=True)
        subprocess.run(['git','fetch','--quiet','--depth=1','origin',manifest['source_commit']],cwd=root,check=True,timeout=180)
        subprocess.run(['git','checkout','--quiet','--detach',manifest['source_commit']],cwd=root,check=True)
        stage=root/'results/b0-gate-a-index';stage.mkdir(parents=True)
        write(stage/'decision.json',{'format':schema,'status':'PUBLICATION_VERIFIED',
            'training_status':'COMPLETED','publication_status':'VERIFIED','scientific_status':proof.get('scientific_status','NOT_EVALUATED'),'manifest':manifest,'artifact':proof,'hypotheses':'NOT_EVALUATED'})
        branch='kaggle/'+manifest['run_id'];destination='artifacts/experiments/epclm004-m5-sv-b0/'+manifest['run_id']
        spec=pub.ExperimentSpec(str(stage.relative_to(root)),destination,('decision.json',),branch,schema)
        folder=pub.prepare_artifacts(root,'epclm004-m5-sv-b0',spec,None)
        pub.push_results(root,folder,'epclm004-m5-sv-b0',branch,'GITHUB_TOKEN','origin/main')
        return {'branch':branch,'remote_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'status':'REMOTE_SHA_VERIFIED'}
