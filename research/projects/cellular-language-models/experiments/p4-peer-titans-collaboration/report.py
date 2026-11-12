"""Render the M1 measurements without re-executing acquisition."""
import json
from pathlib import Path
from statistics import mean
import sys


def render(manifest):
    lines = [
        '# P4-M1 single Cell competence',
        '',
        f"Status: {manifest['status']}. CPU FP32 synthetic development, PyTorch {manifest['torch']}.",
        '',
        'The frozen M3 Gate 5 control reached 100% Recall@1 after 128 online updates. '
        'This control uses the deterministic initializer rather than the smoke-trained weights.',
        '',
        '| Bindings | Recall@1 across seeds | Random | Mean cosine | Mean MSE | Unknown false positive |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for n in manifest['config']['sizes']:
        episodes = [e for e in manifest['episodes'] if e['n'] == n]
        scores = [e['recall_at_1'] for e in episodes]
        cosine = mean(e['cosine_similarity'] for e in episodes)
        mse = mean(e['mse'] for e in episodes)
        unknown = mean(e['unknown_key_false_positive'] for e in episodes)
        lines.append(f'| {n} | {mean(scores):.2%} ({min(scores):.2%}–{max(scores):.2%}) | '
                     f'{1/n:.2%} | {cosine:.4f} | {mse:.4f} | {unknown:.2%} |')
    episodes = manifest['episodes']
    updates = sum(e['updates'] for e in episodes) + manifest['positive_control']['inner_updates']
    restore_error = max(e['restore_max_abs_error'] for e in episodes)
    lines += [
        '',
        f'{updates:,} inner updates including the control. '
        f'Wall time {manifest["wall_seconds"]:.2f} seconds. Maximum restore error {restore_error:g}. '
        'Each read checked the fast weights, momentum, update counter and metadata for mutations. '
        'Scoring uses the candidate bank only after the Memory returns its vector.',
        '',
        'Each size has three independent mappings. Keys and values use separate QR draws up to 32 '
        'bindings and independent normalized Gaussian draws at 64. The change in vector geometry '
        'and equal update count across sizes limit interpretation as an intrinsic capacity bound. '
        'Unknown-key false positives use a fixed maximum candidate cosine threshold of 0.8.',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    print(render(json.loads(Path(sys.argv[1]).read_text())), end='')
