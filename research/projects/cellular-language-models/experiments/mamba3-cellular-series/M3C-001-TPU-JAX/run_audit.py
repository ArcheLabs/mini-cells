"""TPU candidate validation, SISO before MIMO. Results use existing publisher."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'scripts/research/publish_experiment_results.py').exists())


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(out,candidate,require_tpu=True,publish=True):
    out.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    d={'format':'minicells.m3c-001-tpu-jax.v1','run_id':'m3c-001-tpu-jax-'+source[:12]+'-'+str(time.time_ns()),
       'source_commit':source,'status':'INCOMPLETE','siso':'NOT_RUN','mimo':'NOT_RUN',
       'training_tokens':0,'optimizer_updates':0,'scientific_hypotheses':'NOT_EVALUATED',
       'official_kernel_equivalence':'NOT_VERIFIED','m3c002':'BLOCKED_PENDING_TPU_GATES_AND_REVIEW'}
    try:
        lock=json.loads((BASE/'source-lock.json').read_text())
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=candidate,text=True).strip()==lock['commit'],'candidate commit mismatch'
        for name,digest in lock['files'].items():assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==digest,name+' mismatch'
        write(out/'source-lock.json',lock)
        sys.path.insert(0,str(candidate.resolve()))
        import jax
        devices=jax.devices()
        write(out/'environment.json',{'python':sys.version,'jax':jax.__version__,'devices':[str(x) for x in devices],
              'platforms':[x.platform for x in devices], 'process_count':jax.process_count(),
              'dependency_freeze':sorted(dist.metadata['Name']+'=='+dist.version for dist in importlib.metadata.distributions())})
        if require_tpu and not any(x.platform=='tpu' for x in devices):
            d.update(status='BLOCKED_TPU_UNAVAILABLE',reason='No TPU device; CPU fallback is not a TPU result.')
        else:
            import mamba3_jax
            assert Path(mamba3_jax.__file__).resolve().is_relative_to(candidate.resolve()),'candidate import not pinned checkout'
            from gates import check_mode
            # Reuse one upstream NumPy golden and sequential reference per mode, no full test matrix.
            references={'siso':['TestGoldenParity.test_logits_parity','TestSSDForward.test_chunked_matches_sequential'],
                        'mimo':['TestMamba3MIMO.test_chunked_matches_sequential']}
            for mode in ('siso','mimo'):
                d[mode]='INCOMPLETE';write(out/'decision.json',d)
                command=[sys.executable,'-m','pytest','-q']+[str(candidate/'tests/test_mamba3.py')+'::'+x.replace('.', '::') for x in references[mode]]
                result=subprocess.run(command,cwd=candidate,text=True,capture_output=True,timeout=600)
                write(out/(mode+'-reference.json'),{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                      'provenance':'candidate-authored NumPy reference; not official PyTorch kernel evidence'})
                if result.returncode:raise RuntimeError(mode+' reference gate failed')
                report=check_mode(mode=='mimo');write(out/(mode+'.json'),report);d[mode]=report['status'];write(out/'decision.json',d)
            d.update(status='TPU_CANDIDATE_GATES_PASS' if require_tpu else 'CPU_ENGINEERING_PASS',
                     m3c002='ELIGIBLE_FOR_CANDIDATE_BACKEND_DESIGN_REVIEW' if require_tpu else 'BLOCKED_NO_TPU_EVIDENCE')
    except Exception:d['error']=traceback.format_exc()
    write(out/'decision.json',d)
    (out/'RESEARCH-REPORT.zh-CN.md').write_text('# M3C-001-TPU-JAX\n\n状态：'+d['status']+'\n\nSISO：'+d['siso']+'；MIMO：'+d['mimo']+'。\n\n社区候选实现检查；官方内核等价性 NOT_VERIFIED，科学假设 NOT_EVALUATED。0 训练 Token、0 参数更新。M3C-002 仅在 TPU 门禁通过后具备设计审查资格，不自动训练。历史 PyTorch BLOCKED_HARDWARE 保留。\n')
    if publish:return_results(out,d)
    return d


def return_results(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    key='m3c-001-tpu-jax';spec=EXPERIMENTS[key];staging=ROOT/spec.source_dir
    if staging.exists():shutil.rmtree(staging)
    shutil.copytree(out,staging)
    dest=prepare_artifacts(ROOT,key,replace(spec,files=tuple(p.name for p in staging.iterdir() if p.is_file())),None)
    branch='kaggle/'+decision['run_id'];push_results(ROOT,dest,key,branch,'GITHUB_TOKEN','origin/main')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert remote_branch_sha(ROOT,branch)==sha,'remote publication SHA mismatch'
    write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':sha})
    print({'status':decision['status'],'publication':'REMOTE_SHA_VERIFIED','branch':branch},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--publish-only',action='store_true')
    a=p.parse_args()
    if a.publish_only:return_results(a.output,json.loads((a.output/'decision.json').read_text()))
    else:run(a.output,a.candidate)
