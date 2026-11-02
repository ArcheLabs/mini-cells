"""Repository-first source/asset audit; no download and no real training."""
import json
import subprocess
import torch
from common import BASE,ROOT,NATIVE,INCREMENTAL,native,phase_c,digest,write,protocol
from architecture import build
from flops import report


def run(destination=None):
    destination=destination or BASE/'reports';destination.mkdir(parents=True,exist_ok=True)
    paths=[NATIVE/name for name in ('native_clm_runtime.py','native_clm_phase_b.py','native_clm_phase_c.py','README.md')]
    paths += [BASE.parent/'epclm-core-001/implementation/model.py',BASE.parent/'epclm-core-001/gc001-baseline-repair/repair_models.py',
              BASE.parent/'epclm-core-001/topology-001/topology.py',INCREMENTAL/'epclm001/incremental.py',INCREMENTAL/'epclm001/retrain_data.py']
    frozen=json.loads((INCREMENTAL.parent/'retrain_config.json').read_text());pin=json.loads((INCREMENTAL.parent/'formal_config.json').read_text())
    counts={name:report(build(name)) for name in ('A0','A1','A2','A3')}
    tokenizer=ROOT/frozen['tokenizer_path']
    value={'status':'SOURCE_AND_PINNED_MANIFEST_AUDITED','repository_head_at_audit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'implementation_files':{p.name:digest(p) for p in sorted(BASE.glob('*.py'))},
        'files':{str(p.relative_to(ROOT)):{'exists':p.exists(),'sha256':digest(p) if p.exists() else None} for p in paths},
        'baseline':{'architecture':'Native Phase-C X4','parameters':counts['A0']['parameter_count'],'alpha':protocol()['baseline']['alpha'],
                    'actual_alpha':__import__('common').N1_ALPHA,'phase_c_initialization_seed_rule':'run_native_clm_phase_c temporarily sets base.C0_NAME=name, so model_seed=data_seed=91001','parameter_count_matches':counts['A0']['parameter_count']==1893912},
        'history':{'phase_c_seed_91001_results':'README points to implementation/results/dev/phase-c-baseline-seed-91001; not present locally',
                   'controlled_retrain_report':str((INCREMENTAL.parent/'reports/retrain-published/fbf1959b2a364022959ded11c1e4fea6/MODEL_X4_REPORT.json').relative_to(ROOT)),
                   'controlled_retrain_seed':91002,'weights_status':'HISTORICAL_WEIGHTS_UNAVAILABLE','weights_scope':'local weights absent; pinned remote located but not downloaded or verified in this job',
                   'pinned_external_asset':{'repo':pin['repo'],'run_id':pin['asset_run_id'],**pin['assets']['X4']},
                   'policy':'external asset is located, not downloaded or relabeled as fresh initialization'},
        'data':{'manifest_source':str((INCREMENTAL.parent/'retrain_config.json').relative_to(ROOT)),
                'manifest':frozen['historical_corpus_manifest'],'runtime_default_revision_discrepancy':True,
                'effective_phase_c_revision':native.DATASET_REVISION,'tokenizer_local_sha256':digest(tokenizer),
                'tokenizer_matches':digest(tokenizer)==frozen['tokenizer_sha256'],
                'raw_train_validation_status':'NOT_PRESENT_LOCALLY_NOT_RECONSTRUCTED_FOR_CPU_TESTS',
                'preparation':'reuse epclm001.retrain_data.prepare; pinned tokenizer and original writer, byte hashes must match'},
        'incremental_reuse':{'module':'epclm001.incremental.IncrementalLM','reused':['per-stage/per-iteration layout','_attend','_append','rope_at','cache validation','cache_bytes'],
                            'extension':'fixed 32-token state commits and explicit absolute-position local-window semantics'},
        'flops':counts,'training_status':'NOT_RUN_YET' if torch.cuda.is_available() and torch.cuda.get_device_name()=='Tesla T4' else 'NOT_RUN_RESOURCE_UNAVAILABLE','local_gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'independent_test_accessed':False,'scientific_confirmation':False}
    write(destination/'REPOSITORY-AUDIT.json',value)
    return value


if __name__=='__main__':run()
