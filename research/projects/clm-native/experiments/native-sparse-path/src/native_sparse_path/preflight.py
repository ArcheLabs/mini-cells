"""M0 evidence audit and hardware/parameter feasibility; no downloads."""
import csv
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
import torch
from .models import ARMS, NativeModel
from .reuse import ROOT, canonical_sha256, file_sha256, write_json


def hardware(output=None):
    try:
        gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                                       '--format=csv,noheader'], text=True, timeout=10).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpus = []
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    record = {'platform': platform.platform(), 'cpu_logical_cores': os.cpu_count(),
        'ram_bytes': ram, 'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'gpus': gpus,
        'paid_resources_requested': False,
        'gpu_runtime_verified': False, 'selected_screening_device': 'cpu'}
    if output:
        write_json(output, record)
    return record


def prior_evidence(output=None):
    base = ROOT / 'research/projects/cellular-computing/experiments'
    record = {}
    for name in ('007-minicells-30m', '015-emergent-sparse-topology',
                 '015c-local-substrate-emergent-topology', '016-growing-cellular-lm'):
        directory = base / name / 'artifacts'
        metadata = json.loads((directory / 'metadata.json').read_text())
        checks = []
        for row in metadata['files']:
            path = directory / row['path']
            checks.append({'path': row['path'], 'exists': path.is_file(),
                'sha256_matches': path.is_file() and file_sha256(path) == row['sha256']})
        record[name] = {'source_commit': metadata.get('source_commit'),
            'decision_sha256': file_sha256(directory / 'decision.json'),
            'archived_decision': json.loads((directory / 'decision.json').read_text()),
            'manifest_checks': checks, 'fully_reproduced': False,
            'raw_training_logs_available': any(directory.glob('*.jsonl')),
            'status': 'ARCHIVE_HASHES_VERIFIED' if all(r['sha256_matches'] for r in checks) else 'ARCHIVE_HASH_MISMATCH',
            'scope': 'archived outcomes verified against recorded hashes; no historical retraining'}
    directory = base / '007-minicells-30m/artifacts'
    rows = list(csv.DictReader((directory / 'model-summary.csv').open()))
    record['007-minicells-30m']['throughput_ratio_transformer_over_cells'] = (
        float(rows[1]['tokens_per_second']) / float(rows[0]['tokens_per_second']))
    lang = ROOT / 'research/projects/cellular-language-models/experiments'
    for name, commit in [('003-hard-cell-transition', '819df6113ca381a79e901330b7e0a1c394e0ee3b'),
                         ('004-cell-graph', 'a59402238b686ff4981786f307d4a3f110698d61')]:
        directory = lang / name / 'artifacts' / commit / 'seed-42'
        if name.startswith('004'):
            directory /= 'reference'
        path = directory / 'metrics.json'
        record[name] = {'source_commit': commit, 'metrics_sha256': file_sha256(path),
            'models': json.loads(path.read_text())['models'], 'status': 'DEVELOPMENT_ARCHIVE_ONLY',
            'scope': 'one seed/session, some incomplete controls; not a full matrix claim'}
    if output:
        write_json(output, record)
    return record


def preflight(config, output=None):
    counts = {arm: sum(p.numel() for p in NativeModel(config, arm).parameters()) for arm in ARMS}
    spread = (max(counts.values()) - min(counts.values())) / min(counts.values())
    result = {'status': 'M0_ENGINEERING_FEASIBLE', 'scientific_decision': False,
              'config_sha256': canonical_sha256(config), 'parameters': counts,
              'parameter_spread': spread, 'within_five_percent': spread <= 0.05,
              'under_one_million': max(counts.values()) < 1_000_000,
              'resource': hardware(), 'screening_jobs': 12,
              'formal_confirmation': 'NOT_REGISTERED'}
    if not result['within_five_percent'] or not result['under_one_million']:
        raise RuntimeError('baseline capacity gate failed')
    if output:
        write_json(output, result)
    return result
