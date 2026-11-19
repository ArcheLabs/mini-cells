"""Run the EPCLM-008 engineering preflight using the frozen baseline modules."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'epclm-007-cell-retrieval-learning'))
from preflight007 import data_gate
from portability_preflight import checks
from portable_model import EP4, E3, M2, PortableModel
from baselines import build, cached_forward
from data import digest, file_hash
from e0 import write


def preflight(output, device):
    config = json.loads((BASE / 'protocol.json').read_text())
    output.mkdir(parents=True)
    e0 = build('E0', M2['arms']['E0']['model'], 42).to(device).eval()
    target = PortableModel().to(device).eval()
    counts = {
        'E0': sum(p.numel() for p in e0.parameters()),
        'E3': sum(p.numel() for p in target.backbone.parameters()),
        'E3_Titans_total': sum(p.numel() for p in target.parameters()),
    }
    counts['Titans_additional'] = counts['E3_Titans_total'] - counts['E3']
    counts['relative_backbone_delta'] = abs(counts['E3'] - counts['E0']) / counts['E0']
    if counts['relative_backbone_delta'] > .01:
        raise ValueError('Backbone parameter mismatch exceeds 1%')
    architectures = {'E0': M2['arms']['E0'], 'E3': E3,
                     'Titans': {'dim': target.memory.dim, 'hidden': target.memory.hidden}}
    write(output / 'architecture.json', architectures)
    write(output / 'parameter_manifest.json', counts)
    write(output / 'config.lock.json', {
        'protocol': config, 'protocol_sha256': digest(config),
        'architecture_sha256': digest(architectures),
        'tokenizer_sha256': file_hash(EP4 / 'assets/tokenizer.json'),
        'm2_protocol_sha256': file_hash(EP4 / 'm2/configs/protocol.json'),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                                 text=True).strip(),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'device': torch.cuda.get_device_name() if device == 'cuda' else platform.machine(),
    })
    ids = torch.arange(16, device=device)[None]
    with torch.no_grad():
        full = e0(ids)
        left, state = cached_forward(e0, ids[:, :7])
        right, _ = cached_forward(e0, ids[:, 7:], state)
        streamed = torch.cat((left, right), dim=1)
        torch.testing.assert_close(streamed, full, atol=1e-4, rtol=1e-4)
    del e0, target
    write(output / 'dataset_audit.json', data_gate())
    write(output / 'portability_preflight.json', checks(output / 'fixture', device))
    return {'status': 'INCONCLUSIVE', 'engineering_preflight': 'PASS',
            'official_training_updates': 0, 'test': 'SEALED',
            'e0_chunk_max_abs_error': float((streamed - full).abs().max()),
            'formal_benchmark': 'PENDING'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight',), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = preflight(args.output, args.device)
    except Exception as error:
        if args.output.is_dir() and not (args.output / 'decision.json').exists():
            write(args.output / 'decision.json', {'status': 'INCONCLUSIVE',
                  'engineering_preflight': 'FAIL', 'error': str(error), 'test': 'SEALED'})
        raise
    write(args.output / 'decision.json', result)
    print(result)
