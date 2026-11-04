"""Existing bounded Kaggle transport; immutable objects prevent checkpoint/hash races."""
import json,os,sys,time,shutil,subprocess
from dependencies import BASE,ROOT,shared


def commit_bundle(out,run_identity,updated):
    index=out/'recovery.json';bundle=json.loads(index.read_text()) if index.exists() else {'identity':{'run_identity':run_identity},'files':{},'objects':{}}
    if bundle['identity']!={'run_identity':run_identity}:raise ValueError('recovery namespace identity differs')
    store=out/'_recovery_objects';store.mkdir(exist_ok=True)
    for source in [out/'manifest.json',updated]:
        if not source.exists():continue
        name=source.relative_to(out).as_posix()
        temporary=store/('snapshot-'+__import__('uuid').uuid4().hex+source.suffix)
        try:os.link(source,temporary)
        except OSError:shutil.copyfile(source,temporary)
        digest=shared.digest(temporary);object_name='_recovery_objects/'+digest+source.suffix;physical=out/object_name
        if physical.exists():temporary.unlink()
        else:temporary.replace(physical)
        bundle['files'][name]=digest;bundle['objects'][name]=object_name
    pending=out/'recovery.pending';pending.write_text(json.dumps(bundle,indent=2)+'\n');pending.replace(index)
    # Prune only after publishing the new immutable-object manifest locally.
    retained=set(bundle['objects'].values())
    for source in store.iterdir():
        if source.relative_to(out).as_posix() not in retained:source.unlink()


def restore(out,run_identity,deadline,artifact_dir=None):
    core=BASE.parent/'epclm-core-001'
    sys.path[:0]=[str(core/'gc001-baseline-repair'),str(core/'implementation')]
    from repair_io import recover_namespace,timed_budget
    trace={};proof=None
    prior=subprocess.run(['git','ls-remote','--heads','origin','refs/heads/kaggle/'+out.name+'-*'],cwd=ROOT,capture_output=True,text=True,timeout=10)
    if prior.returncode:raise RuntimeError('cannot verify previous runs; refusing an implicit fresh restart')
    published_progress=bool(prior.stdout.strip())
    if artifact_dir and published_progress:
        branches=[line.split()[1][len('refs/heads/'):]  for line in prior.stdout.splitlines()]
        if len(branches)<=3:
            try:
                from repair_io import public_result
                with timed_budget(max(.1,min(20,deadline-time.monotonic()))):
                    reports=[public_result(ROOT,branch,artifact_dir)[0] for branch in branches]
                # Failed preflight publications without any training may restart safely.
                published_progress=any(report.get('pilot_training_started',True) for report in reports)
            except Exception:published_progress=True
    try:
        with timed_budget(max(.1,min(45,deadline-time.monotonic()))):
            proof=recover_namespace(ROOT,out,out.name,{'run_identity':run_identity},deadline,max_versions=3,trace=trace,cache=out.parent/'recovery-downloads')
    except Exception as error:
        trace['error']=type(error).__name__+': '+str(error)
        if trace.get('matching_progress_found'):raise RuntimeError('matching saved progress found but recovery incomplete; refusing retraining: '+str(error)) from error
    if proof is None and published_progress:raise RuntimeError('same-source result branches already exist but saved progress could not be recovered; refusing implicit retraining; use explicit --no-recovery only for a deliberate fresh execution')
    if proof is not None:
        for checkpoint in sorted(out.rglob('checkpoint.pt')):commit_bundle(out,run_identity,checkpoint)
    shared.write(out/'recovery-attempt.json',{'proof':proof,'trace':trace,'search_limit_seconds':45,'versions':3})
    return proof
