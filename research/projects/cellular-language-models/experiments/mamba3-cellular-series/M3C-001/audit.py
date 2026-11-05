"""Official module audit only: no optimizer, training data or checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / 'scripts/research/publish_experiment_results.py').exists())
PIN = json.loads((BASE / 'source-lock.json').read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def hardware_gate(capability):
    # Official dependency floor Triton 3.5 targets NVIDIA CC >= 8.0.
    return capability[0] >= 8


def close(a, b):
    import torch
    torch.testing.assert_close(a.float(), b.float(), atol=0.02, rtol=0.02)
    return float((a.float() - b.float()).abs().max())


def inspect_mode(mimo):
    import torch
    from mamba_ssm.modules.mamba3 import Mamba3
    torch.manual_seed(18001)
    model = Mamba3(d_model=256, d_state=128, headdim=64, expand=2,
                   is_mimo=mimo, mimo_rank=4, chunk_size=16 if mimo else 64,
                   layer_idx=0, device='cuda', dtype=torch.bfloat16).eval()
    x = torch.randn(2, 129, 256, device='cuda', dtype=torch.bfloat16)
    state_names = ('angle_dt', 'ssm', 'k', 'v')
    def cache():
        states = model.allocate_inference_cache(2, 129)
        return SimpleNamespace(seqlen_offset=0, key_value_memory_dict={0: states})
    report = {'status': 'INCOMPLETE', 'configuration': {'d_model': 256, 'd_state': 128,
              'expand': 2, 'headdim': 64, 'chunk_size': model.chunk_size, 'is_mimo': mimo},
              'precision': 'bfloat16 parameters/input; official FP32 angle/SSM state',
              'backward': 'NOT_RUN', 'optimizer_updates': 0,
              'multi_token_chunk_continuation': 'NOT_EXPOSED_BY_MODULE_Input_States_None'}
    try:
        with torch.inference_mode():
            full = model(x)
            assert torch.isfinite(full).all()
            c = cache(); prefix = model(x[:, :64], inference_params=c)
            snapshot = tuple(s.clone() for s in c.key_value_memory_dict[0])
            outputs = [prefix]
            for t in range(64, 129):
                c.seqlen_offset = t
                outputs.append(model(x[:, t], inference_params=c).unsqueeze(1))
            report['cached_full_max_error'] = close(full, torch.cat(outputs, dim=1))
            restored = cache(); restored.key_value_memory_dict[0] = tuple(s.clone() for s in snapshot)
            restored.seqlen_offset = 64
            report['snapshot_restore_max_error'] = close(outputs[1], model(x[:,64], inference_params=restored).unsqueeze(1))
            c2 = cache(); report['reset_max_error'] = close(prefix, model(x[:,:64], inference_params=c2))
            separate = torch.cat([model(x[i:i+1]) for i in range(2)])
            report['batch_isolation_max_error'] = close(full, separate)
            future = x.clone(); future[:,65:] *= -1
            report['causality_max_error'] = close(full[:,:65], model(future)[:,:65])
            report['states'] = {n: {'shape': list(s.shape), 'dtype': str(s.dtype), 'bytes': s.numel()*s.element_size()}
                                for n,s in zip(state_names, snapshot)}
            # Timed fixed inputs; warm-up and synchronization exclude JIT compilation.
            times=[]; torch.cuda.reset_peak_memory_stats()
            for i in range(8):
                torch.cuda.synchronize(); start=time.perf_counter(); model(x); torch.cuda.synchronize()
                if i>=3: times.append((time.perf_counter()-start)*1000)
            times.sort(); report['prefill_ms']={'p50':times[2], 'p95':times[-1], 'samples':5, 'warmup':3, 'batch':2, 'length':129}
            c3=cache(); model(x[:,:128],inference_params=c3); c3.seqlen_offset=128
            times=[]
            for i in range(8):
                c3.key_value_memory_dict[0]=tuple(s.clone() for s in snapshot)
                torch.cuda.synchronize(); start=time.perf_counter(); model(x[:,128],inference_params=c3); torch.cuda.synchronize()
                if i>=3:times.append((time.perf_counter()-start)*1000)
            times.sort();report['decode_ms']={'p50':times[2],'p95':times[-1],'samples':5,'warmup':3,'batch':2,'scope':'single module, fixed input and cloned state; no LM generation'}
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['status']='MODULE_INFERENCE_PASS'
    except Exception:
        report['error']=traceback.format_exc()
    return report


def launch(out, official, publish=True):
    import torch
    out.mkdir(parents=True, exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    decision={'format':'minicells.m3c-001.audit.v1','run_id':'m3c-001-audit-'+source[:12]+'-'+str(time.time_ns()),
              'source_commit':source,'status':'INCOMPLETE','training_tokens':0,'optimizer_updates':0,
              'scientific_hypotheses':'NOT_EVALUATED','full_lm_baseline':'NOT_IMPLEMENTED_NOT_AUTHORIZED',
              'checkpoint_created':False}
    try:
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()
        assert actual==PIN['commit'], 'official source commit mismatch'
        for name,digest in PIN['files'].items():
            assert hashlib.sha256((official/name).read_bytes()).hexdigest()==digest, name+' source mismatch'
        write(out/'source-lock.json',PIN)
        env={'python':sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
             'capability':list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}
        for package in ('triton','tilelang','quack-kernels','nvidia-cutlass-dsl','mamba_ssm'):
            try:env[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:env[package]=None
        write(out/'environment.json',env)
        if env['capability'] is None or not hardware_gate(env['capability']):
            decision.update(status='BLOCKED_HARDWARE',reason='Pinned Triton >=3.5 requires NVIDIA CC >=8.0; T4 is SM75. No dependency build or substitute kernel attempted.',siso='NOT_RUN',mimo='NOT_RUN')
        else:
            # Official install, bounded separately; preserve the target CUDA PyTorch installation.
            subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation',str(official)],check=True,timeout=600)
            import mamba_ssm
            write(out/'resolved-dependencies.json',subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
            write(out/'import-identity.json',{'module_file':mamba_ssm.__file__,'official_commit':actual})
            results={}
            for mode in ('siso','mimo'):
                try:results[mode]=inspect_mode(mode=='mimo')
                except Exception:results[mode]={'status':'INCOMPLETE','error':traceback.format_exc()}
                write(out/(mode+'.json'),results[mode])
            decision.update(status='AUDIT_COMPLETE' if all(r['status']=='MODULE_INFERENCE_PASS' for r in results.values()) else 'AUDIT_PARTIAL',modes={k:v['status'] for k,v in results.items()})
    except Exception:
        decision['error']=traceback.format_exc()
    write(out/'decision.json',decision)
    (out/'RESEARCH-REPORT.zh-CN.md').write_text('# M3C-001 官方模块审计\n\n状态：'+decision['status']+'\n\n详见 decision.json、environment.json 和各模式报告。仅审计随机初始化的官方单模块；不构成语言模型基线复现或科学确认。训练、反向、参数更新及完整 LM 包装均未执行。T4 被依赖兼容门禁阻塞时，不安装 CUDA 内核。\n')
    if publish:
        from dataclasses import replace
        import shutil
        sys.path.insert(0,str(ROOT))
        from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results,remote_branch_sha
        spec=EXPERIMENTS['m3c-001-audit']; staging=ROOT/spec.source_dir
        if staging.exists():shutil.rmtree(staging)
        shutil.copytree(out,staging)
        names=tuple(p.name for p in staging.iterdir() if p.is_file())
        destination=prepare_artifacts(ROOT,'m3c-001-audit',replace(spec,files=names),None)
        branch='kaggle/'+decision['run_id']
        push_results(ROOT,destination,'m3c-001-audit',branch,'GITHUB_TOKEN','origin/main')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert remote_branch_sha(ROOT,branch)==commit, 'remote SHA mismatch'
        write(out/'publication-verification.json',{'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':commit})
    print(decision,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--official',type=Path,required=True)
    a=p.parse_args();launch(a.output,a.official)
