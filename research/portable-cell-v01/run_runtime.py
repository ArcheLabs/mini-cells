"""Run fixed-init and trained two-chunk comparisons in isolated workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from portable import Cell, export, restore

UPSTREAM = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
CHUNK = 16
SEED = 42
ATOL, RTOL = 1e-6, 1e-5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def worker(args):
    sys.path.insert(0, str(args.upstream))
    from titan_llama import TitanDecoderLayer, TitanLLaMAConfig, TitanLLaMAForCausalLM

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    config = TitanLLaMAConfig(segment_len=CHUNK, neural_memory_segment_len=CHUNK,
                              neural_memory_layers=(8,), neural_memory_batch_size=CHUNK,
                              use_flex_attn=False, history_blend=True)
    model = TitanLLaMAForCausalLM.from_pretrained_llama(
        MODEL, config, dtype=torch.float32, revision=args.revision,
        token=os.getenv('HF_TOKEN')).to('cuda')
    model.freeze_backbone()
    trainable = parameters(model)
    layer = model.backbone.model.layers[8]
    assert isinstance(layer, TitanDecoderLayer) and layer.has_neural_memory
    assert layer.neural_memory.store_chunk_size == CHUNK
    assert trainable
    weights = args.output / 'outer.safetensors'
    if args.mode == 'train':
        model.train()
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4)
        losses = []
        tracked = {k: p.detach().clone() for k, p in trainable.items()
                   if "history_mix" in k or "memory_model_parameters" in k}
        for index in range(args.steps):
            model.reset_memory_states()
            ids = inputs(index + 100, 2 * CHUNK)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, use_cache=False)["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits[:, CHUNK:-1].flatten(0, 1), ids[:, CHUNK + 1:].flatten())
            loss.backward()
            if not torch.isfinite(loss) or not any(
                p.grad is not None and p.grad.abs().max().item() > 0
                for p in trainable.values()
            ):
                raise RuntimeError('Invalid outer training signal')
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f'Outer step {index+1}/{args.steps}: {losses[-1]:.6f}', flush=True)
        changes = {k: float((trainable[k].detach() - v).abs().max()) for k, v in tracked.items()}
        for group in ('history_mix', 'memory_model_parameters'):
            if not any(group in k and delta > 0 for k, delta in changes.items()):
                raise RuntimeError(f'No update to {group}')
        save_file({k: p.detach().cpu().contiguous() for k, p in trainable.items()}, str(weights))
        (args.output / 'training.json').write_text(json.dumps(
            {'steps': args.steps, 'seed': SEED, 'losses': losses, 'sha256': sha(weights),
             'parameters': sorted(trainable), 'parameter_max_abs_delta': changes, 'data': 'seeded synthetic token IDs'}))
        return

    if args.variant == 'trained':
        saved = load_file(str(weights))
        if saved.keys() != trainable.keys():
            raise ValueError('Incompatible outer checkpoint parameter names')
        with torch.no_grad():
            for name, p in trainable.items():
                value = saved[name]
                if value.shape != p.shape or value.dtype != p.dtype or not value.isfinite().all():
                    raise ValueError(f'Incompatible outer parameter {name}')
                p.copy_(value)
    from artifact import canonical, digest
    fingerprint = digest(canonical({
        'base_revision': args.revision, 'upstream': UPSTREAM, 'config': vars(config),
        'seed': SEED, 'outer_sha256': sha(weights) if args.variant == 'trained' else None,
    }))
    model.eval()
    outputs = []
    hook = layer.neural_memory.register_forward_hook(
        lambda _module, _args, result: outputs.append(result[0].detach().cpu()))

    def advance(cell, ids):
        # Both paths intentionally start from empty Session caches.
        model.reset_memory_states()
        if cell.memories:
            layer.state = type(layer.state)(memory=cell.memories[8])
        layer.portable_memory_position = cell.position
        outputs.clear()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        if len(outputs) != 1 or not torch.isfinite(outputs[0]).all():
            raise RuntimeError('Invalid memory retrieval output')
        return cell.staged({8: layer.state.memory}, tokens=CHUNK, chunk_size=CHUNK)

    initial = Cell.create(fingerprint)
    root = args.output / args.variant
    root.mkdir(exist_ok=True)
    if args.mode == 'restore':
        # Derive trusted tensor shapes from an unrelated input. No replay of Chunk 1.
        template = advance(initial, inputs(99))
        cell = restore(root / 'chunk1', template)
    else:
        cell = advance(initial, inputs(1))
        if args.mode == 'export':
            export(cell, root / 'chunk1')
            return
    cell = advance(cell, inputs(2))
    export(cell, root / args.mode)
    save_file({'retrieval': outputs[0].contiguous()}, str(root / f'{args.mode}-read.safetensors'))
    hook.remove()


def inputs(index, length=CHUNK):
    generator = torch.Generator().manual_seed(SEED + index)
    return torch.randint(100, 12000, (1, length), generator=generator).cuda()


def compare(root):
    from artifact import read
    from portable import SCHEMA

    left, a = read(root / 'continuous', SCHEMA)
    right, b = read(root / 'restore', SCHEMA)
    for key in ('position', 'revision', 'fingerprint', 'layout'):
        if left[key] != right[key]:
            raise AssertionError(f'Cell metadata mismatch: {key}')
    if left['position'] != 2 * CHUNK or left['revision'] != 2:
        raise AssertionError('Incorrect final cursor')
    a.update(load_file(str(root / 'continuous-read.safetensors')))
    b.update(load_file(str(root / 'restore-read.safetensors')))
    if a.keys() != b.keys():
        raise AssertionError('State tensor keys differ')
    errors = {}
    for key, value in a.items():
        torch.testing.assert_close(value, b[key], rtol=RTOL, atol=ATOL)
        errors[key] = float((value - b[key]).abs().max()) if value.numel() else 0.0
    return {'status': 'PASS', 'position': left['position'], 'seq_index': left['position'],
            'revision': left['revision'], 'layout': left['layout'], 'max_abs_error': errors,
            'atol': ATOL, 'rtol': RTOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--steps', type=int, default=4, choices=range(1, 17))
    parser.add_argument('--outer-checkpoint', type=Path)
    parser.add_argument('--outer-sha256')
    parser.add_argument('--mode', choices=('continuous', 'export', 'restore', 'train'))
    parser.add_argument('--variant', choices=('initial', 'trained'), default='initial')
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('--revision requires the immutable Llama model revision')
    if args.mode:
        worker(args)
        return

    args.output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'],
                               text=True).strip() != UPSTREAM:
        raise ValueError('Unexpected upstream revision')
    patch = Path(__file__).with_name('upstream-position.patch')
    subprocess.run(['git', '-C', str(args.upstream), 'apply', '--reverse', '--check', str(patch)],
                   check=True)
    commands = [sys.executable, str(Path(__file__).resolve()), '--upstream', str(args.upstream.resolve()),
                '--output', str(args.output.resolve()), '--revision', args.revision, '--steps', str(args.steps)]
    environment = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    report = {'status': 'RUNNING', 'upstream': UPSTREAM, 'patch_sha256': sha(patch),
              'base_revision': args.revision, 'seed': SEED, 'chunk': CHUNK,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'session_policy': 'reset both arms before chunk 2',
              'semantic_validation': 'NOT_EVALUATED', 'timings_seconds': {}}
    summary = args.output / 'runtime-report.json'

    def run(mode, variant):
        start = time.monotonic()
        print(f'{variant}: {mode}', flush=True)
        try:
            subprocess.run(commands + ['--mode', mode, '--variant', variant], check=True, env=environment)
        finally:
            report['timings_seconds'][f'{variant}/{mode}'] = time.monotonic() - start

    try:
        for variant in ('initial', 'trained'):
            if variant == 'trained':
                initial_elapsed = sum(report['timings_seconds'].values())
                print(f'Fixed-init workflow: {initial_elapsed:.1f}s. '
                      f'Trained comparison repeats the same three workers after {args.steps} outer steps.',
                      flush=True)
                if args.outer_checkpoint:
                    if sha(args.outer_checkpoint) != args.outer_sha256:
                        raise ValueError('External outer checkpoint SHA256 mismatch')
                    (args.output / 'outer.safetensors').write_bytes(args.outer_checkpoint.read_bytes())
                    report['outer_source'] = 'user-supplied'
                else:
                    print(f'Planning estimate: {initial_elapsed * 2:.1f}s for both parity runs '
                          f'plus model load and {args.steps} short outer updates.', flush=True)
                    run('train', variant)
                    report['outer_source'] = 'synthetic short training'
                report['outer_sha256'] = sha(args.output / 'outer.safetensors')
            for mode in ('continuous', 'export', 'restore'):
                run(mode, variant)
            report[variant] = compare(args.output / variant)
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='ERROR', error=str(error))
        raise
    finally:
        summary.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
