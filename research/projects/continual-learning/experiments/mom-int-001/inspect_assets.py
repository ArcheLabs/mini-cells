"""Audit immutable assets before attempting a MoM runtime integration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = '033057c2f80941cb0b70ef62a99dd7ddfc94bf52'
MODEL = 'linear-moe-hub/MoM-Gated-Deltanet-340M'
REVISION = 'f4081422ea2cf01b1b5d646cac05cad62a5d5438'
EXPECTED = dict(model_type='mom_gated_deltanet', num_hidden_layers=24,
                hidden_size=1024, num_heads=4, head_dim=256, num_memories=4,
                topk=2, shared_mem=True, use_short_conv=True, conv_size=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('snapshot', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    report = dict(model=MODEL, revision=REVISION, expected_source=SOURCE,
                  status='BLOCKED', integration='NOT_EVALUATED')
    try:
        source = subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
        if source != SOURCE:
            raise ValueError('Unexpected MoM source revision')
        config = a.snapshot/'config.json'
        data = json.loads(config.read_text())
        mismatches = {k:[v,data.get(k)] for k,v in EXPECTED.items() if data.get(k)!=v}
        if mismatches:
            raise ValueError(f'Unexpected MoM configuration: {mismatches}')
        weights = a.snapshot/'model.safetensors'
        if not weights.is_file():
            raise FileNotFoundError(weights)
        with weights.open('rb') as f:
            report['local_weights_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
        report.update(status='ASSETS_PRESENT',config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        report['reason']=str(exc)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ASSETS_PRESENT' else 2


if __name__=='__main__':
    raise SystemExit(main())
