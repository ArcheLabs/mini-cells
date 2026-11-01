"""Reuse Kaggle's existing authenticated transport and immutable output manifests."""
import json
import signal
import time
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
from frozen_model import AssetUnavailable,BaselineIntegrityError,check_deadline,safe_path
from native_clm_runtime import sha256_file


_BACKENDS={}

def backend(root,trace=None):
    """Install/initialize the pinned transport at most once in this process."""
    key=str(root.resolve())
    if key not in _BACKENDS:
        sys.path.insert(0,str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
        from kaggle_output_recovery import _load_backend
        started=time.monotonic()
        if trace is not None:trace['active_operation']='BACKEND_INITIALIZATION'
        try:
            with timed_budget(30):_BACKENDS[key]=_load_backend(root)
        finally:
            if trace is not None:trace['backend_initialization_seconds']=time.monotonic()-started
    return _BACKENDS[key]


def version_files(api,version,deadline,trace=None):
    names=set();token=None;seen=set()
    if trace is not None:
        trace.setdefault('searched_versions',[]).append(version);trace['active_operation']='OUTPUT_INVENTORY';trace['current_version']=version
    while True:
        check_deadline(deadline);page=api.page(version,token)
        names.update(row['file_name'] for row in page['files']);token=page.get('next_page_token')
        if not token:return names
        if token in seen:raise BaselineIntegrityError('pagination loop')
        seen.add(token)


def recover_inputs(root,output,gc,versions,deadline,include_weights=False,trace=None):
    """Known-version input lookup. A/B never request historical model weights."""
    data_names=('tokenizer.json','train.json','development.json')
    manifest_path=output/'assets/manifest.json'
    def validate_local():
        if not manifest_path.exists() or sha256_file(manifest_path)!=gc['training_identity']['data']:return None
        metadata=json.loads(manifest_path.read_text())
        files={'assets/manifest.json':gc['training_identity']['data']}
        files.update({'assets/'+n:metadata['files'][n] for n in data_names})
        if include_weights:files.update({'C1.pt':gc['C1_checkpoint']['sha256'],'F0.pt':gc['F0_checkpoint']['sha256']})
        if not all((output/n).exists() and sha256_file(output/n)==sha for n,sha in files.items()):return None
        return files
    local=validate_local()
    if local is not None:
        proof={'status':'VERIFIED_LOCAL_INPUTS','files':local,'historical_weights_requested':include_weights}
        if trace is not None:trace['verified_assets']=local
        return proof
    api,fetch=backend(root,trace);namespace=gc['C1_checkpoint']['namespace']
    cache=output/'download-cache';output.mkdir(parents=True,exist_ok=True)
    for version in versions:
        names=version_files(api,version,deadline,trace)
        for name in sorted(n for n in names if n.endswith(namespace+'/recovery.json')):
            safe_path(name);destination=cache/str(version)
            fetch(api,{'version_number':version,'files':[{'name':name,'bytes':None}]},[name],destination)
            check_deadline(deadline);manifest=json.loads((destination/name).read_text())
            if manifest['identity']!=gc['training_identity']:continue
            if manifest['files'].get('assets/manifest.json')!=gc['training_identity']['data']:
                raise BaselineIntegrityError('input manifest identity differs')
            logical=['assets/manifest.json']+['assets/'+n for n in data_names]
            if include_weights:logical+=['C1.pt','F0.pt']
            if not set(logical)<=manifest['files'].keys():raise AssetUnavailable('known input bundle is incomplete')
            if include_weights and any(manifest['files'][n]!=gc[n[:2]+'_checkpoint']['sha256'] for n in ('C1.pt','F0.pt')):
                raise BaselineIntegrityError('historical weight hash differs')
            prefix=name.rsplit('/',1)[0]+'/'
            locations={n:prefix+str(safe_path(manifest.get('objects',{}).get(n,n))) for n in logical}
            if not set(locations.values())<=names:raise AssetUnavailable('known input objects missing')
            needed=list(dict.fromkeys(locations.values()))
            if trace is not None:trace.update(active_operation='INPUT_FILES_DOWNLOAD',requested_assets=logical)
            fetch(api,{'version_number':version,'files':[{'name':n,'bytes':None} for n in needed]},needed,destination)
            for logical_name,physical in locations.items():
                check_deadline(deadline);saved=destination/physical
                if sha256_file(saved)!=manifest['files'][logical_name]:raise BaselineIntegrityError('input file hash differs')
                target=output/safe_path(logical_name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(saved,target)
                if trace is not None:trace.setdefault('verified_assets',{})[logical_name]=manifest['files'][logical_name]
            verified=validate_local()
            if verified is None:raise BaselineIntegrityError('input manifest/file hashes disagree')
            proof={'status':'VERIFIED_KNOWN_VERSION_INPUTS','version':version,'files':verified,'historical_weights_requested':include_weights}
            (output/'input-recovery.json').write_text(json.dumps(proof,indent=2)+'\n')
            if trace is not None:trace['verified_assets']=verified
            return proof
    raise AssetUnavailable('known input version unavailable; no wide history scan or new dataset/tokenizer training')


def public_result(root,branch,artifact,source=None,protocol=None,commit=None,optional=False):
    ref=subprocess.run(['git','ls-remote','--heads','origin','refs/heads/'+branch],cwd=root,capture_output=True,text=True,timeout=60)
    if ref.returncode:raise AssetUnavailable('GitHub result lookup failed; no completed work will be retrained')
    if not ref.stdout.strip():
        if optional:return None
        raise AssetUnavailable('required published batch missing: '+branch)
    revision=ref.stdout.split()[0]
    if commit is not None and revision!=commit:raise BaselineIntegrityError('historical result branch moved from pinned commit')
    result=subprocess.run(['git','fetch','--depth=1','origin','refs/heads/'+branch],cwd=root,capture_output=True,text=True,timeout=60)
    if result.returncode:raise AssetUnavailable('GitHub result fetch failed')
    fetched=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=root,text=True).strip()
    if fetched!=revision:raise BaselineIntegrityError('result branch changed during fetch')
    raw=subprocess.check_output(['git','show',fetched+':'+artifact+'/decision.json'],cwd=root)
    value=json.loads(raw)
    if source is not None and value['identity']['source']!=source:raise BaselineIntegrityError('batch source differs')
    if protocol is not None and value['identity']['protocol']!=protocol:raise BaselineIntegrityError('batch protocol differs')
    return value,{'branch':branch,'commit':revision,'decision_sha256':__import__('hashlib').sha256(raw).hexdigest()}


def recover_namespace(root,output,namespace,identity,deadline,max_versions=25,selected=None,trace=None,resume_only=False,cache=None):
    """Exact-source progress or explicitly pinned historical weights, never an implicit optimizer reuse."""
    api,fetch=backend(root,trace)
    from kaggle_output_recovery import _version_number
    if trace is not None:trace['active_operation']='CURRENT_VERSION_METADATA'
    latest=_version_number(api.metadata())
    cache=Path(cache) if cache is not None else Path('/kaggle/temp/gc001-repair-recovery')/namespace
    for version in range(latest,max(0,latest-max_versions),-1):
        names=version_files(api,version,deadline,trace)
        for name in sorted(n for n in names if n.endswith(namespace+'/recovery.json')):
            safe_path(name);destination=cache/str(version)
            fetch(api,{'version_number':version,'files':[{'name':name,'bytes':None}]},[name],destination)
            manifest=json.loads((destination/name).read_text())
            if manifest['identity']!=identity:continue
            if trace is not None:trace['matching_progress_found']={'version':version,'namespace':namespace}
            logical=list(manifest['files']) if selected is None else list(selected)
            if resume_only and selected is None:
                decision='decision.json';physical=name.rsplit('/',1)[0]+'/'+str(safe_path(manifest.get('objects',{}).get(decision,decision)))
                if decision not in manifest['files'] or physical not in names:raise AssetUnavailable('matching progress lacks decision')
                fetch(api,{'version_number':version,'files':[{'name':physical,'bytes':None}]},[physical],destination)
                decision_file=destination/physical
                if sha256_file(decision_file)!=manifest['files'][decision]:raise BaselineIntegrityError('progress decision checksum differs')
                report=json.loads(decision_file.read_text())
                if any(report.get('identity',{}).get(k)!=v for k,v in identity.items()):raise BaselineIntegrityError('progress report identity differs from manifest')
                needed={decision,'blind-review.csv','private/human-key.json','human-review-scored.csv'}
                if report['status']!='COMPLETE':
                    needed.update(model+'.pt' for model,entry in report.get('models',{}).items() if entry.get('origin','').startswith('NEW_REPAIR') and entry.get('status')!='COMPLETE')
                logical=[n for n in logical if n in needed or n.startswith('failures/')]
                if trace is not None:
                    trace['deferred_private_assets']={'version':version,'namespace':namespace,'files':{n:sha for n,sha in manifest['files'].items() if n not in logical}}

            if not set(logical)<=manifest['files'].keys():raise AssetUnavailable('matching saved bundle lacks required files')
            prefix=name.rsplit('/',1)[0]+'/'
            locations={n:prefix+str(safe_path(manifest.get('objects',{}).get(n,n))) for n in logical}
            for n in logical:
                safe_path(n)
                if Path(n).suffix not in ('.pt','.json','.md','.csv'):raise BaselineIntegrityError('unexpected repair artifact')
                if selected is not None and selected[n]!=manifest['files'][n]:raise BaselineIntegrityError('pinned historical artifact hash differs')
            if not set(locations.values())<=names:raise AssetUnavailable('saved bundle incomplete')
            required=list(dict.fromkeys(locations.values()));check_deadline(deadline)
            fetch(api,{'version_number':version,'files':[{'name':n,'bytes':None} for n in required]},required,destination)
            for n in logical:
                check_deadline(deadline);saved=destination/locations[n]
                if sha256_file(saved)!=manifest['files'][n]:raise BaselineIntegrityError('saved repair artifact checksum differs')
                target=output/safe_path(n);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(saved,target)
            return {'version':version,'namespace':namespace,'identity':identity,'files':{n:manifest['files'][n] for n in logical}}
    return None


@contextmanager
def timed_budget(seconds):
    """Nested deadlines restore the enclosing timer instead of cancelling it."""
    previous=signal.getitimer(signal.ITIMER_REAL);began=time.monotonic()
    def expired(signum,frame):raise TimeoutError('repair execution/recovery budget exhausted')
    old=signal.signal(signal.SIGALRM,expired)
    limit=min(seconds,previous[0]) if previous[0]>0 else seconds
    signal.setitimer(signal.ITIMER_REAL,max(limit,1e-6))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if previous[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous[0]-(time.monotonic()-began)),previous[1])
