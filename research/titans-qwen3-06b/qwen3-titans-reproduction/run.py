"""Single seed reproduction. Preflight and full training are separate actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
TITANS = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
SEED = 42
DATASETS = ('zai-org/LongAlign-10k', 'HuggingFaceFW/fineweb-edu', 'OldKingMeister/LoongRL-Train-Data')
STAGES = ('pretrain', 'gold_sft', 'quality_sft')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')


def profiles(upstream):
    return {stage: json.loads((upstream / f'configs/seg512_{stage}/training_config.json').read_text())
            for stage in STAGES}


def lr_scale(step, cfg, steps):
    base = cfg['neural_mem_learning_rate']
    lo = cfg['min_neural_mem_learning_rate'] / base
    start = cfg['neural_mem_warmup_start_learning_rate'] / base
    warm = cfg['warmup_steps']
    if step < warm:
        return start + (1 - start) * step / warm
    return lo + (1 - lo) * (1 + math.cos(math.pi * min(1., (step - warm) / max(1, steps - warm)))) / 2


def objective(model, batch, mode='titans', use_checkpoint=False):
    from model import hidden_states
    device = next(model.parameters()).device
    ids, mask, labels = (batch[k].to(device) for k in ('input_ids', 'attention_mask', 'labels'))
    length = int(mask.sum(dim=1).max())
    ids, mask, labels = (x[:, :length] for x in (ids, mask, labels))
    labels = labels[:, 1:].masked_fill(~mask[:, 1:].bool(), -100)
    count = int((labels != -100).sum())
    if not count:
        raise ValueError('No supervised next tokens in batch')

    def forward(i, m):
        return hidden_states(model, i, m, mode)

    def tile(h, target):
        logits = model.lm_head(h).float()
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100, reduction='sum')
        correct = ((logits.argmax(-1) == target) & (target != -100)).sum()
        return loss, correct

    h = forward(ids, mask)
    evaluate = lambda a, b: checkpoint(tile, a, b, use_reentrant=False) if use_checkpoint else tile(a, b)
    pairs = [evaluate(h[:, i:i+512], labels[:, i:i+512]) for i in range(0, length - 1, 512)]
    return sum(x[0] for x in pairs), count, sum(int(x[1]) for x in pairs)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])


def train_stage(model, loader, cfg, stage, root, identity, resume=False, smoke=False, warm_start=None):
    from fp32_adamw import FP32AdamW
    from model import DIAGNOSTICS
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = FP32AdamW([{'params': list(params.values()), 'lr': cfg['neural_mem_learning_rate'],
                          'betas': (.9, .95), 'weight_decay': cfg['neural_mem_weight_decay']}])
    batch_size = cfg['batch_size']
    steps = cfg['total_tokens'] // (batch_size * cfg['sequence_length'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_scale(step, cfg, steps))
    directory = root / 'training' / 'checkpoints' / stage
    directory.mkdir(parents=True, exist_ok=True)
    cursor = 0
    start = 0
    pointer = directory / 'latest.json'
    if pointer.exists() and not resume:
        raise FileExistsError(f'{stage} already has a checkpoint; use --resume')
    if pointer.exists():
        receipt = json.loads(pointer.read_text())
        path = directory / receipt['file']
        if sha(path) != receipt['sha256']:
            raise ValueError('Checkpoint integrity failure')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['stage'] != stage:
            raise ValueError('Checkpoint identity mismatch')
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved['params'][n])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        cursor, start = saved['sampler_position'], saved['step']
        saved_rng = saved['rng']
    elif warm_start is not None:
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(warm_start[n])
    def batches():
        while True:
            seen = False
            for batch in loader:
                seen = True
                yield batch
            if not seen:
                raise ValueError(f'{stage} yielded no samples')

    iterator = batches()
    for _ in range(cursor):
        next(iterator)
    if pointer.exists():
        restore_rng(saved_rng)

    def save(step):
        path = directory / f'{step:06d}.pt'
        temp = path.with_suffix('.tmp')
        torch.save(dict(identity=identity, stage=stage, step=step, sampler_position=cursor,
                        rng=rng_state(), params={n: p.detach().cpu() for n, p in params.items()},
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temp)
        temp.replace(path)
        write(pointer, {'file': path.name, 'sha256': sha(path), 'step': step})

    if not pointer.exists():
        save(0)
    for step in range(start + 1, (2 if smoke else steps) + 1):
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        loss, tokens, correct, diagnostics = 0., 0, 0, []
        micro = 1 if smoke else batch_size
        for i in range(micro):
            batch = next(iterator)
            cursor += 1
            if smoke and (batch['input_ids'].shape != (1, 16384) or not batch['attention_mask'].all()):
                raise ValueError('Smoke requires one packed 16K sequence per update')
            torch.cuda.synchronize()
            forward_start = time.monotonic()
            # Collect diagnostics on the first actual sequence, without
            # retaining graph references or including checkpoint recompute.
            handle = DIAGNOSTICS.set(diagnostics if i == 0 else None)
            try:
                nll, n, hits = objective(model, batch, use_checkpoint=True)
            finally:
                DIAGNOSTICS.reset(handle)
            if not torch.isfinite(nll):
                raise ValueError('Nonfinite training loss')
            torch.cuda.synchronize()
            forward_seconds = time.monotonic() - forward_start
            forward_peak = torch.cuda.max_memory_allocated()
            if smoke:
                torch.cuda.reset_peak_memory_stats()
            backward_start = time.monotonic()
            (nll / (n * micro)).backward()
            torch.cuda.synchronize()
            backward_seconds = time.monotonic() - backward_start
            backward_peak = torch.cuda.max_memory_allocated()
            loss += float(nll.detach())
            tokens += n
            correct += hits
        bad = sum(int((~p.grad.isfinite()).sum()) for p in params.values() if p.grad is not None)
        norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True))
        if bad or norm == 0 or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Gradient audit failed')
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        metrics = dict(stage=stage, step=step, tokens=tokens, nll=loss/tokens, accuracy=correct/tokens, lr=lr)
        elapsed = time.monotonic() - begin
        append(root / 'training' / 'metrics.jsonl', metrics)
        append(root / 'training' / 'timing.jsonl', dict(stage=stage, step=step, seconds=elapsed,
                                                      peak_bytes=max(forward_peak, backward_peak),
                forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                forward_peak_bytes=forward_peak, backward_peak_bytes=backward_peak))
        append(root / 'training' / 'diagnostics.jsonl', dict(stage=stage, step=step, layers=diagnostics,
               backbone_gradient_norm=0., titans_gradient_norm=norm, nonfinite_gradient_count=bad))
        print(json.dumps(dict(metrics, seconds=elapsed)), flush=True)
        if smoke or step % cfg['save_interval'] == 0 or step == steps:
            save(step)
    return {n: p.detach().cpu().clone() for n, p in params.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('upstream', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--dataset-lock', type=Path)
    p.add_argument('--validation', type=Path, help='Frozen held-out Qwen-tokenized JSONL')
    p.add_argument('--quality-train', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-receipt', type=Path)
    args = p.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != TITANS or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no']):
        raise ValueError('Use a clean checkout of the pinned Titans revision')
    sys.path.insert(0, str(upstream))
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM
    import train_datasets as data
    from train_titan_llama import TrainingConfig, make_dataset
    from model import install
    previous = args.output / 'dataset-manifest.json'
    if args.dataset_lock:
        lock = json.loads(args.dataset_lock.read_text())
    elif args.resume and previous.exists():
        lock = json.loads(previous.read_text())['revisions']
    else:
        api = HfApi()
        lock = {name: api.dataset_info(name).sha for name in DATASETS}
    if not set(DATASETS).issubset(lock) or any(len(lock[name]) != 40 for name in DATASETS):
        p.error('Dataset lock must contain the three upstream dataset revisions')
    if not args.smoke and not args.smoke_receipt:
        p.error('Formal training requires a completed real-data smoke receipt')
    if not args.smoke and (not args.quality_train or not args.quality_train.is_file()):
        p.error('The upstream QuALITY train file is required for the formal sequence of stages')
    if not torch.cuda.is_available():
        p.error('A CUDA GPU is required for the 16K training workload')
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.smoke:
        torch.use_deterministic_algorithms(True)
    root = args.output
    root.mkdir(parents=True, exist_ok=args.resume)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    snapshot = snapshot_download(MODEL, revision=REVISION,
                                 allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    locked = lambda name, *a, **kw: load_dataset(name, *a, revision=lock[name], **kw)
    data.load_dataset = locked
    all_profiles = profiles(upstream)
    identity = dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    backbone=MODEL, backbone_revision=REVISION, titans_revision=TITANS, seed=SEED,
                    backbone_dtype=str(dtype), memory_dtype='torch.float32', torch=str(torch.__version__),
                    gpu=torch.cuda.get_device_name(), model_config_sha256=sha(Path(snapshot)/'config.json'),
                    tokenizer_sha256=sha(Path(snapshot)/'tokenizer.json'), configs=all_profiles,
                    datasets=lock, validation_sha256=sha(args.validation) if args.validation else None,
                    quality_train_sha256=sha(args.quality_train) if args.quality_train else None)
    smoke_identity = {k: v for k, v in identity.items()
                      if k not in ('quality_train_sha256', 'validation_sha256')}
    if not args.smoke:
        receipt = json.loads(args.smoke_receipt.read_text())
        if receipt.get('engineering') != 'SMOKE_ONLY' or receipt.get('smoke_identity') != smoke_identity:
            raise ValueError('Smoke receipt must match source, data, precision and GPU')
    for file in ('manifest.json', 'source-lock.json'):
        target = root / file
        if target.exists() and json.loads(target.read_text()) != identity:
            raise ValueError('Existing run has a different identity')
        write(target, identity)
    write(root / 'config.json', all_profiles)
    write(root / 'dataset-manifest.json', dict(revisions=lock, quality_train_sha256=identity['quality_train_sha256']))
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype,
                    attn_implementation='sdpa').to('cuda').eval()
    from model import hidden_states, LAYERS
    ids = torch.tensor([[1, 2, 3, 4]], device='cuda')
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        original = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    install(model)
    with torch.no_grad():
        torch.testing.assert_close(hidden_states(model, ids, mask, 'native'), original, rtol=0, atol=0)
        for i in LAYERS:
            memory = model.model.layers[i].self_attn.memory
            for j, param in enumerate(memory.memory_model_parameters):
                expected = torch.eye(128, device='cuda').expand(8, 128, 128) * (4 if j == 2 else 1)
                torch.testing.assert_close(param, expected, rtol=0, atol=0)
    write(root / 'compatibility-report.json', dict(native_equivalence='PASS', identity_initialization='PASS',
                                                   native_positions=40960, experiment_position_limit=32768))
    from model import checkpoint_native_layers, install_full_attention_kernel
    install_full_attention_kernel(model)
    checkpoint_native_layers(model)
    named = {n: p.numel() for n,p in model.named_parameters() if p.requires_grad}
    write(root / 'trainable-parameters.json', dict(total=sum(named.values()), parameters=named))
    weights = None
    for stage in (STAGES[:1] if args.smoke else STAGES):
        cfg = all_profiles[stage]
        args_cfg = dict(cfg, use_ddp=False, micro_batch_size=1, tokenizer_name=snapshot)
        if stage == 'quality_sft':
            from sft_kvwrite_quality_mc import QualityMCSFTDataset
            ds = QualityMCSFTDataset(snapshot, cfg['sequence_length'], cfg['quality_sft_max_examples'], args.quality_train)
        else:
            ds = make_dataset(TrainingConfig(**args_cfg))
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
        weights = train_stage(model, loader, cfg, stage, root, identity, args.resume, args.smoke, weights)
    if args.smoke:
        uninterrupted = root / 'training/checkpoints/pretrain'
        restored = root / 'restore'
        dest = restored / 'training/checkpoints/pretrain'
        dest.mkdir(parents=True)
        mid = uninterrupted / '000001.pt'
        shutil.copyfile(mid, dest / mid.name)
        write(dest / 'latest.json', dict(file=mid.name, sha256=sha(mid), step=1))
        ds = make_dataset(TrainingConfig(**args_cfg))
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
        train_stage(model, loader, cfg, 'pretrain', restored, identity, resume=True, smoke=True)
        a, b = [torch.load(folder / '000002.pt', map_location='cpu', weights_only=True)
                for folder in (uninterrupted, dest)]
        def compare(x, y):
            if torch.is_tensor(x):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            elif isinstance(x, dict):
                if x.keys() != y.keys():
                    raise AssertionError('Checkpoint keys differ')
                for key in x:
                    compare(x[key], y[key])
            elif isinstance(x, (tuple, list)):
                for lhs, rhs in zip(x, y, strict=True):
                    compare(lhs, rhs)
            elif x != y:
                raise AssertionError('Checkpoint field differs')
        compare(a, b)
        write(root / 'resume-parity.json', dict(status='PASS', tolerance=0, step=2,
              scope='disk restore with new optimizer, scheduler and data iterator in the same process'))
    if args.validation:
        from evaluate import evaluate
        evaluate(model, args.validation, root / 'evaluation')
    write(root / 'decision.json', dict(engineering='SMOKE_ONLY' if args.smoke else 'TRAINING_COMPLETE',
                                       scientific='NOT_EVALUATED',
                                       smoke_identity=smoke_identity if args.smoke else None))


if __name__ == '__main__':
    main()
