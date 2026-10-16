"""Serial, post-worker inference profiling; no diagnostic dispatcher in live training."""

from pathlib import Path
import statistics
import time
import numpy as np
import torch
from .model import Model
from .training import Trainer
from .checkpoint import restore, identity
from .data import load_assets, batches
from .common import source_sha, hash_accepted_state
from .accounting import profile
from .report import seal
from .automatic import device_queues


@torch.no_grad()
def completed(c, output, assets):
    data = load_assets(c, assets)
    for index, arm in enumerate(c["arms"]):
        folder = Path(output) / "arms" / arm
        if not (folder / "metrics.json").exists():
            continue
        device = "cuda:" + str(next(i for i, queue in enumerate(device_queues(c)) if arm in queue))
        t = Trainer(Model(c, arm).to(device))
        rng = np.random.default_rng(1)
        restore(folder, t, rng, identity(c, arm, source_sha(), data))
        x = batches(data, c, rng, device)[0][0]
        before = hash_accepted_state(t.model)
        inventory = profile(t.model, x, repeats=3)
        length = min(x.shape[1] - 1, 16)
        past = t.model(x[:, :length], policy="greedy").memory
        times = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            t.model.decode_step(x[:, length], past, policy="greedy")
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - begin) * 1000)
        if hash_accepted_state(t.model) != before:
            raise RuntimeError("profile mutated model")
        seal(
            folder / "cost-profile.json",
            {
                **inventory,
                "decode_p50_ms": statistics.median(times),
                "decode_scope": "one cached greedy step, batch="
                + str(x.shape[0])
                + ", prefix="
                + str(length),
                "phase": "SERIAL_AFTER_ALL_TRAIN_WORKERS_EXIT",
                "source_sha": source_sha(),
            },
        )
