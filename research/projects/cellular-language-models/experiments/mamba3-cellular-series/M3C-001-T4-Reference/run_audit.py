"""FP32 PyTorch Reference validation, SISO before MIMO. Results use existing publisher."""
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


def run(out,candidate,require_gpu=True,publish=True):
    out.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    d={'format':'minicells.m3c-001-t4-reference.v1','run_id':'m3c-001-t4-reference-'+source[:12]+'-'+str(time.time_ns()),
       'source_commit':source,'status':'INCOMPLETE','siso':'NOT_RUN','mimo':'NOT_RUN',
       'training_tokens':0,'optimizer_updates':0,'precision':'fp32','tf32':False,'initialization':'upstream create_toy_model initializer','scientific_hypotheses':'NOT_EVALUATED',
       'official_kernel_equivalence':'NOT_VERIFIED','m3c002':'BLOCKED_PENDING_GPU_GATES_AND_REVIEW'}
    try:
        lock=json.loads((BASE/'source-lock.json').read_text())
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=candidate,text=True).strip()==lock['commit'],'candidate commit mismatch'
        for name,digest in lock['files'].items():assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==digest,name+' mismatch'
        write(out/'source-lock.json',lock)
        sys.path.insert(0,str(candidate.resolve()))
        import torch
        write(out/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'dependency_freeze':sorted(dist.metadata['Name']+'=='+dist.version for dist in importlib.metadata.distributions())})
        if require_gpu and not torch.cuda.is_available():
            d.update(status='BLOCKED_GPU_UNAVAILABLE',reason='No CUDA device; CPU evidence is not T4 evidence.')
        else:
            import mamba3
            assert Path(mamba3.__file__).resolve().is_relative_to(candidate.resolve()),'candidate import not pinned checkout'
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            from gates import check_mode
            for mode in ('siso','mimo'):
                d[mode]='INCOMPLETE';write(out/'decision.json',d)
                report=check_mode(mode=='mimo','cuda' if require_gpu else 'cpu')
                write(out/(mode+'.json'),report);d[mode]=report['status'];write(out/'decision.json',d)
            d.update(status='GPU_REFERENCE_GATES_PASS' if require_gpu else 'CPU_ENGINEERING_PASS',
                     m3c002='REFERENCE_BACKEND_READY_FOR_REVIEW' if require_gpu else 'BLOCKED_NO_GPU_EVIDENCE')
    except Exception:d['error']=traceback.format_exc()
    write(out/'decision.json',d)
    (out/'RESEARCH-REPORT.zh-CN.md').write_text('# M3C-001-T4-Reference\n\n状态：'+d['status']+'\n\nSISO：'+d['siso']+'；MIMO：'+d['mimo']+'。\n\n第三方 PyTorch Reference 检查；官方内核等价性 NOT_VERIFIED，科学假设 NOT_EVALUATED。0 训练 Token、0 参数更新。M3C-002 仅在 GPU 门禁通过后具备设计审查资格，不自动训练。历史 PyTorch BLOCKED_HARDWARE 保留。\n')
    if publish:return_results(out,d)
    return d


def return_results(out,decision):
    sys.path.insert(0,str(ROOT))
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
    key='m3c-001-t4-reference';spec=EXPERIMENTS[key];staging=ROOT/spec.source_dir
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
