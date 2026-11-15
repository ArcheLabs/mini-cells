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

from portable import Cell, export, fork, restore

UPSTREAM = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
MODEL = 'unsloth/Llama-3.2-1B-Instruct'
CHUNK = 512
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
    import dataclasses

    settings = json.loads((args.upstream / 'configs/seg512_pretrain/training_config.json').read_text())
    settings.update(use_flash_attn=False, use_flex_attn=False)
    settings['neural_memory_layers'] = tuple(settings['neural_memory_layers'])
    config = TitanLLaMAConfig(**{f.name: settings[f.name] for f in dataclasses.fields(TitanLLaMAConfig)
                                if f.name in settings})
    if args.mode in ('prepare-data', 'train'):
        import train_titan_llama as upstream_training
        from bootstrap_data import LocalBootstrap, prepare

        settings.update(batch_size=1, micro_batch_size=1, gradient_accumulation_steps=1,
                        sequence_length=2 * CHUNK, total_tokens=args.steps * 2 * CHUNK,
                        base_model_name=str(args.model_path), tokenizer_name=str(args.model_path),
                        model_dtype='float32', use_ddp=False, save_interval=1,
                        fast_resume_data=False, output_dir=str(args.output / 'checkpoints'))
        fields = {field.name for field in dataclasses.fields(upstream_training.TrainingConfig)}
        training = upstream_training.TrainingConfig(**{k: v for k, v in settings.items() if k in fields})
        (args.output / 'training_config.json').write_text(json.dumps(vars(training), indent=2))
        if args.mode == 'prepare-data':
            from importlib.metadata import version
            from bootstrap_data import cache_key

            cache_root = args.data_cache or args.output.parent / 'portable-bootstrap-cache'
            inputs_root = Path('/kaggle/input')
            mirrors = tuple(inputs_root.rglob('portable-bootstrap-cache')) if inputs_root.is_dir() else ()
            tokenizer = cache_key({
                'files': {path.name: sha(path) for path in args.model_path.iterdir()
                          if path.is_file() and (path.name.startswith(('tokenizer', 'special_tokens'))
                                                 or path.suffix in ('.jinja', '.model'))},
                'transformers': version('transformers'), 'tokenizers': version('tokenizers'),
            })
            prepare(args.output, upstream_training.make_dataset, training, cache_root, tokenizer,
                    {'commit': UPSTREAM, 'patch': sha(Path(__file__).with_name('upstream-position.patch')),
                     'datasets_version': version('datasets')}, mirrors)
            return
        dataset = LocalBootstrap(args.output, training)
        upstream_training.make_dataset = lambda _config: dataset
        print(f'Training data verified locally ({training.total_steps} batches)', flush=True)
        original_create = upstream_training.create_model_and_optimizer
        before = {}
        names = []

        def create(config, device):
            model, optimizer, scheduler = original_create(config, device)
            trainable = parameters(model)
            permitted = all(
                name.startswith('backbone.model.layers.') and
                ('.neural_memory.' in name or '.to_learned_v_mix.' in name)
                for name in trainable
            )
            optimized = {id(param) for group in optimizer.param_groups for param in group['params']}
            if not trainable or not permitted or optimized != {id(param) for param in trainable.values()}:
                raise RuntimeError('Frozen Backbone contract failed before training')
            print(f'Verified {sum(p.numel() for p in trainable.values()):,} '
                  'trainable NMM/Gate parameters in the upstream optimizer', flush=True)
            names.extend(trainable)
            before.update({k: v.detach().cpu().clone() for k, v in trainable.items()
                           if 'history_mix' in k or 'memory_model_parameters' in k})
            return model, optimizer, scheduler

        upstream_training.create_model_and_optimizer = create
        original_save = upstream_training.save_checkpoint
        measurements = []
        last = time.monotonic()

        def save_checkpoint(model, optimizer, scheduler, config, step, loss):
            nonlocal last
            path = original_save(model, optimizer, scheduler, config, step, loss)
            now = time.monotonic()
            measurements.append(dict(upstream_step=step, loss=loss, elapsed_seconds=now - last))
            last = now
            (args.output / 'training_metrics.json').write_text(json.dumps(measurements, indent=2))
            if len(measurements) == 2:
                print(f'Estimated remaining training and saves: '
                      f'{measurements[-1]["elapsed_seconds"] * max(args.steps - 2, 0):.0f}s', flush=True)
            return path

        upstream_training.save_checkpoint = save_checkpoint
        upstream_training.main(training)
        state = torch.load(args.output / 'checkpoints/latest_checkpoint.pt',
                           map_location='cpu', weights_only=True)
        outer = {name: state['model_state_dict'][name].contiguous() for name in names}
        changes = {name: float((outer[name] - value).abs().max()) for name, value in before.items()}
        for group in ('history_mix', 'memory_model_parameters'):
            if not any(group in name and delta > 0 for name, delta in changes.items()):
                raise RuntimeError(f'No update to {group}')
        if not all(torch.isfinite(t).all() for t in outer.values()):
            raise RuntimeError('Nonfinite outer checkpoint')
        if not torch.isfinite(torch.as_tensor(state['loss'])) or state['step'] <= 0:
            raise RuntimeError('No finite completed training step')
        weights = args.output / 'outer.safetensors'
        save_file(outer, str(weights))
        (args.output / 'training.json').write_text(json.dumps(
            {'step': state['step'], 'loss': state['loss'], 'sha256': sha(weights),
             'parameter_max_abs_delta': changes}, indent=2))
        return

    model = TitanLLaMAForCausalLM.from_pretrained_llama(
        str(args.model_path), config, dtype=torch.float32).to('cuda')
    model.freeze_backbone()
    trainable = parameters(model)
    layers = {i: model.backbone.model.layers[i] for i in config.neural_memory_layers}
    assert all(isinstance(layer, TitanDecoderLayer) and layer.has_neural_memory and
               layer.neural_memory.store_chunk_size == CHUNK for layer in layers.values())
    weights = args.output / 'outer.safetensors'
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
        'base_revision': args.revision, 'backbone_fingerprint': args.backbone_fingerprint,
        'upstream': UPSTREAM, 'config': vars(model.config),
        'seed': SEED, 'outer_sha256': sha(weights) if args.variant == 'trained' else None,
    }))
    model.eval()
    outputs = {}
    for index, layer in layers.items():
        layer.neural_memory.register_forward_hook(
            lambda _module, _args, result, i=index: outputs.update({f'retrieval_{i}': result[0].detach().cpu()}))

    def forward(cell, ids, use_cache=False, **kwargs):
        model.reset_memory_states()
        for index, layer in layers.items():
            if cell.memories:
                layer.state = type(layer.state)(memory=cell.memories[index])
            layer.portable_memory_position = cell.position
        outputs.clear()
        with torch.no_grad():
            result = model(input_ids=ids, use_cache=use_cache, **kwargs)
        if len(outputs) != len(layers) or not all(t.isfinite().all() for t in outputs.values()):
            raise RuntimeError('Invalid memory retrieval output')
        logits = result['logits']
        if not logits.isfinite().all():
            raise RuntimeError('Invalid model logits')
        outputs['logits'] = logits.detach().cpu()
        return result

    def advance(cell, ids):
        forward(cell, ids)
        state = {index: layer.state.memory for index, layer in layers.items()}
        return cell.staged(state, tokens=CHUNK, chunk_size=CHUNK)

    initial = Cell.create(fingerprint)
    root = args.output / args.variant
    root.mkdir(exist_ok=True)
    if args.mode == 'semantic':
        from transformers import AutoTokenizer
        import random

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        rng = random.Random(SEED)
        codes = rng.sample(range(100000, 999999), 16)
        names = rng.sample(range(100000, 999999), 16)

        def history(values):
            text = '\n'.join(f'Project {name} has deployment code {value}.'
                             for name, value in zip(names, values))
            ids = tokenizer.encode(text)
            if len(ids) > CHUNK:
                raise ValueError('Frozen semantic history exceeds one chunk')
            ids = [tokenizer.bos_token_id] * (CHUNK - len(ids)) + ids
            return torch.tensor([ids], device='cuda')

        cells = {'correct': advance(initial, history(codes)),
                 'donor': advance(initial, history(codes[1:] + codes[:1])),
                 'empty': initial}
        export(cells['correct'], root / 'semantic_cell')
        cells['correct'] = restore(root / 'semantic_cell', cells['correct'])
        items = []
        for name, code in zip(names, codes):
            query = tokenizer.encode(f'Question: What is the deployment code for Project {name}?\nAnswer:')
            answer = tokenizer.encode(f' {code}', add_special_tokens=False)
            ids = torch.tensor([query + answer[:-1]], device='cuda')
            target = torch.tensor(answer)
            scores, matches = {}, {}
            for label, cell in cells.items():
                forward(cell, ids, logits_to_keep=len(answer))
                logits = outputs['logits'][0]
                scores[label] = logits.log_softmax(-1).gather(1, target[:, None]).mean().item()
                matches[label] = torch.equal(logits.argmax(-1), target)
            items.append(dict(entity=name, code=code, answer_tokens=len(answer),
                              nats_per_token=scores, greedy_exact_match=matches))
        means = {arm: sum(item['nats_per_token'][arm] for item in items) / len(items)
                 for arm in cells}
        gains = {arm: means['correct'] - means[arm] for arm in ('empty', 'donor')}
        donor_wins = sum(item['nats_per_token']['correct'] > item['nats_per_token']['donor']
                         for item in items)
        passed = min(gains.values()) >= 0.05 and donor_wins >= 12
        (args.output / 'semantic_recall.json').write_text(json.dumps(
            dict(status='PCC_SEMANTIC_RECALL_PASS' if passed else 'FAIL',
                 gains=gains, means=means, correct_beats_donor=donor_wins, items=items,
                 exact_match={arm: sum(row['greedy_exact_match'][arm] for row in items) / 16 for arm in cells},
                 false_accept_rate={arm: sum(row['greedy_exact_match'][arm] for row in items) / 16
                                    for arm in ('empty', 'donor')}), indent=2))
        return
    if args.mode == 'smoke':
        result = forward(initial, inputs(1), use_cache=True)
        token = result['logits'][:, -1:].argmax(-1)
        if result['past_key_values'] is None:
            raise RuntimeError('Missing runtime KV cache')
        cell = initial.staged({i: layer.state.memory for i, layer in layers.items()},
                              tokens=CHUNK, chunk_size=CHUNK)
        for layer in layers.values():
            layer.portable_memory_position = cell.position
        with torch.no_grad():
            generated = model(input_ids=token, past_key_values=result['past_key_values'],
                              use_cache=True, logits_to_keep=1)['logits']
        if not generated.isfinite().all():
            raise RuntimeError('Nonfinite decode logits')
        export(cell, root / 'smoke')
        child = fork(cell)
        if child.id == cell.id or child.parent != (cell.id, cell.revision):
            raise RuntimeError('Invalid runtime Cell fork')
        from portable import tree
        source_tensors = tree(cell)[1]
        child_tensors = tree(child)[1]
        if any(t.numel() and t.data_ptr() == child_tensors[k].data_ptr()
               for k, t in source_tensors.items()):
            raise RuntimeError('Fork aliases source storage')
        (root / 'generation.json').write_text(json.dumps(
            {'status': 'PCC_REAL_RUNTIME_SMOKE_PASS', 'generated_token': token.item(),
             'next_token': generated.argmax(-1).item(), 'revision': cell.revision,
             'fork': 'PCC_FORK_PASS'}))
        return
    if args.mode == 'restore':
        # Derive trusted tensor shapes from an unrelated input. No replay of Chunk 1.
        template = advance(initial, inputs(99))
        cell = restore(root / 'chunk1', template)
    else:
        cell = advance(initial, inputs(1))
        if args.mode == 'export':
            export(cell, root / 'chunk1')
            return
    previous = cell
    cell = advance(cell, inputs(2))
    for index in layers:
        before, after = previous.memories[index], cell.memories[index]
        if not any(not torch.equal(a, b) for a, b in zip(before.weights, after.weights)):
            raise RuntimeError(f'Fast Weights did not change at layer {index}')
        if not any(not torch.equal(a, b) for a, b in zip(before.updates, after.updates)):
            raise RuntimeError(f'Updates did not change at layer {index}')
    export(cell, root / args.mode)
    restore(root / args.mode, cell)
    save_file({k: v.contiguous() for k, v in outputs.items()},
              str(root / f'{args.mode}-read.safetensors'))


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
    from safetensors.torch import save
    from artifact import canonical, digest

    def normalized(metadata, tensors):
        identity = {key: value for key, value in metadata.items() if key not in ('id', 'parent')}
        hashes = {key: digest(save({'tensor': value.contiguous()})) for key, value in tensors.items()}
        return digest(canonical({'metadata': identity, 'tensors': hashes}))

    hashes = [normalized(left, a), normalized(right, b)]
    a.update(load_file(str(root / 'continuous-read.safetensors')))
    b.update(load_file(str(root / 'restore-read.safetensors')))
    if a.keys() != b.keys():
        raise AssertionError('State tensor keys differ')
    errors = {}
    for key, value in a.items():
        torch.testing.assert_close(value, b[key], rtol=RTOL, atol=ATOL)
        delta = (value - b[key]).abs()
        errors[key] = {'max_abs': float(delta.max()) if delta.numel() else 0.0,
                       'mean_abs': float(delta.mean()) if delta.numel() else 0.0,
                       'relative_l2': float(delta.norm() / value.norm().clamp_min(1e-12))}
    return {'status': 'PASS', 'position': left['position'], 'seq_index': left['position'],
            'revision': left['revision'], 'layout': left['layout'], 'tensor_errors': errors, 'normalized_cell_sha256': hashes,
            'normalized_hash_match': hashes[0] == hashes[1],
            'atol': ATOL, 'rtol': RTOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--backbone-fingerprint', required=True)
    parser.add_argument('--steps', type=int, default=12, choices=range(11, 129))
    parser.add_argument('--data-cache', type=Path)
    parser.add_argument('--outer-checkpoint', type=Path)
    parser.add_argument('--outer-sha256')
    parser.add_argument('--mode', choices=('continuous', 'export', 'restore', 'train', 'smoke', 'semantic', 'prepare-data'))
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
                '--output', str(args.output.resolve()), '--revision', args.revision, '--steps', str(args.steps),
                '--model-path', str(args.model_path.resolve()),
                '--backbone-fingerprint', args.backbone_fingerprint]
    if args.data_cache:
        commands += ['--data-cache', str(args.data_cache.resolve())]
    environment = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'WANDB_MODE': 'disabled'}
    (args.output / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'python': sys.version, 'gpu': torch.cuda.get_device_name(),
        'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
    }, indent=2))
    report = {'status': 'RUNNING', 'upstream': UPSTREAM, 'patch_sha256': sha(patch),
              'base_revision': args.revision, 'backbone_fingerprint': args.backbone_fingerprint, 'seed': SEED, 'chunk': CHUNK,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'session_policy': 'reset both arms before chunk 2',
              'semantic_validation': 'NOT_EVALUATED', 'timings_seconds': {}}
    summary = args.output / 'runtime-report.json'

    def run(mode, variant):
        start = time.monotonic()
        print(f'{variant}: {mode}', flush=True)
        try:
            worker_env = dict(environment)
            if mode == 'prepare-data':
                worker_env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_ETAG_TIMEOUT='30',
                                  HF_HUB_DOWNLOAD_TIMEOUT='45')
            elif mode == 'train':
                worker_env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
            subprocess.run(commands + ['--mode', mode, '--variant', variant], check=True, env=worker_env)
        finally:
            report['timings_seconds'][f'{variant}/{mode}'] = time.monotonic() - start

    try:
        run('smoke', 'initial')
        report['acceptance'] = dict(runtime='PCC_REAL_RUNTIME_SMOKE_PASS', fork='PCC_FORK_PASS')
        (args.output / 'runtime_smoke.json').write_bytes(
            (args.output / 'initial/generation.json').read_bytes())
        for variant in ('initial', 'trained'):
            if variant == 'trained':
                initial_elapsed = sum(value for phase, value in report['timings_seconds'].items()
                                      if phase.startswith('initial/'))
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
                    run('prepare-data', variant)
                    run('train', variant)
                    report['outer_source'] = 'BOUNDED_BOOTSTRAP_CHECKPOINT'
                report['outer_sha256'] = sha(args.output / 'outer.safetensors')
            for mode in ('continuous', 'export', 'restore'):
                run(mode, variant)
            report[variant] = compare(args.output / variant)
        manifest = dict(model_id=MODEL, revision=args.revision,
                        backbone_fingerprint=args.backbone_fingerprint, titan_source_commit=UPSTREAM,
                        upstream_config_sha256=sha(args.upstream / 'configs/seg512_pretrain/training_config.json'),
                        runtime_precision='float32', use_flex_attn=False, use_flash_attn=False,
                        outer_sha256=report['outer_sha256'], source=report['outer_source'])
        (args.output / 'checkpoint_manifest.json').write_text(json.dumps(manifest, indent=2))
        report['acceptance'] = dict(runtime='PCC_REAL_RUNTIME_SMOKE_PASS',
                                    checkpoint='PCC_TITAN_CHECKPOINT_READY',
                                    storage='PCC_CELL_STORAGE_PASS', fork='PCC_FORK_PASS',
                                    restore='PCC_CROSS_PROCESS_RESTORE_PASS',
                                    equivalence='PCC_CONTINUE_EQUIVALENCE_PASS')
        (args.output / 'restore_comparison.json').write_text(json.dumps(report['trained'], indent=2))
        run('semantic', 'trained')
        report['semantic_validation'] = json.loads(
            (args.output / 'semantic_recall.json').read_text())['status']
        report['status'] = ('COMPLETE' if report['semantic_validation'] == 'PCC_SEMANTIC_RECALL_PASS'
                            else 'ENGINEERING_COMPLETE_SEMANTIC_NOT_ESTABLISHED')
    except Exception as error:
        report.update(status='ERROR', error=str(error))
        raise
    finally:
        data_report = args.output / 'bootstrap-data.json'
        if data_report.exists():
            report['bootstrap_data'] = json.loads(data_report.read_text())
        summary.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
