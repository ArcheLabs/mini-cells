"""Observation and safe-boundary persistence around the unchanged historical loop."""

import inspect
import time
from types import SimpleNamespace

from .common import write


class ResourceStop(RuntimeError):
    pass


class TrainingGuard:
    def __init__(self, seconds, root, source_sha, config_sha, run_id):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.root = root
        self.source_sha, self.config_sha, self.run_id = source_sha, config_sha, run_id
        self.actual_updates = 0
        self.save_seconds = 0.0
        self.validation_seconds = 0.0

    def checkpoint(self, payload, path, save):
        payload.update(
            asset_generation="CONTROLLED_RETRAIN_001",
            training_source_sha=self.source_sha,
            controlled_config_sha256=self.config_sha,
            retrain_run_id=self.run_id,
            actual_optimizer_updates=self.actual_updates,
        )
        begin = time.monotonic()
        save(payload, path)
        self.save_seconds += time.monotonic() - begin
        if time.monotonic() >= self.deadline:
            raise ResourceStop("GPU_BUDGET_EXHAUSTED_CHECKPOINT_RETAINED")

    def after_update(self, frame):
        step = frame.f_locals["step"]
        milestone = step * 1000 in frame.f_locals["checkpoint_set"]
        if (step % 100 == 0 or time.monotonic() >= self.deadline) and not milestone:
            # After GradScaler.update: model/optimizer/scaler/data RNG all agree.
            frame.f_locals["save_resume"](step)

    def proxy(self, torch):
        guard = self

        def adam(*args, **kwargs):
            kwargs.update(
                eps=1e-8,
                amsgrad=False,
                foreach=None,
                maximize=False,
                capturable=False,
                differentiable=False,
                fused=None,
            )
            optimizer = torch.optim.AdamW(*args, **kwargs)
            original = optimizer.step

            def step(*a, **kw):
                value = original(*a, **kw)
                guard.actual_updates += 1
                return value

            optimizer.step = step
            return optimizer

        def scaler(*args, **kwargs):
            kwargs.update(
                init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000
            )
            result = torch.amp.GradScaler(*args, **kwargs)
            original = result.update

            def update(*a, **kw):
                value = original(*a, **kw)
                frame = inspect.currentframe().f_back
                guard.after_update(frame)
                return value

            result.update = update
            return result

        class Proxy:
            amp = SimpleNamespace(GradScaler=scaler)
            optim = SimpleNamespace(AdamW=adam)

            def __getattr__(self, name):
                return getattr(torch, name)

        return Proxy()

    def record(self):
        elapsed = time.monotonic() - self.started
        write(
            self.root,
            {
                "gpu_worker_wall_seconds": elapsed,
                "checkpoint_io_seconds": self.save_seconds,
                "validation_seconds": self.validation_seconds,
                "training_and_initialization_seconds": max(
                    0, elapsed - self.save_seconds - self.validation_seconds
                ),
                "actual_optimizer_updates": self.actual_updates,
                "budget_scope": "Conservative GPU worker wall time including validation/checkpoint IO; those costs separately recorded",
            },
        )
