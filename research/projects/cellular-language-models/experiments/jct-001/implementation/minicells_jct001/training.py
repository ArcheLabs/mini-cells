"""Four-arm CPU engineering transactions; no scientific-stage admission here."""

import copy
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from minicells_r3e.common import fingerprint
from .model import KnownPathModel
from .data import batches
from .cost import Journal
from .connection import propose


def snapshot_groups(model):
    return {
        group: {name: p.detach().clone() for name, p in getattr(model, group).named_parameters()}
        for group in ("cells", "connection")
    }


def group_delta(before, model, group):
    values = [
        (p.detach() - before[group][name]).square().sum()
        for name, p in getattr(model, group).named_parameters()
    ]
    return float(torch.stack(values).sum().sqrt())


class Trainer:
    def __init__(self, config, arm, identity, journal, base_state=None):
        self.config = copy.deepcopy(config)
        self.f = config["fixture"]
        self.arm = arm
        self.identity = dict(identity, arm=arm)
        self.model = KnownPathModel(self.f)
        self.cell_optimizer = torch.optim.AdamW(
            self.model.cells.parameters(),
            lr=self.f["cell_lr"],
            weight_decay=self.f["cell_weight_decay"],
        )
        self.data_rng = np.random.default_rng(self.f["seed"] + 1)
        self.route_rng = torch.Generator().manual_seed(self.f["seed"] + 2)
        self.cf_rng = torch.Generator().manual_seed(self.f["seed"] + 3)
        self.step = 0
        self.data_cursor = 0
        self.curve = []
        self.connection_optimizer_state = None
        self.last_proposal = None
        self.parent_base_sha = None
        self.c = dict(
            engine=dict(training=dict(effective_batch=self.f["batch_size"])),
            training={
                key: self.f[key]
                for key in (
                    "connection_lr",
                    "connection_weight_decay",
                    "connection_clip",
                    "policy_steps",
                )
            },
            proposal=dict(backtrack=self.f["backtrack"]),
            engineering_trust=self.f["engineering_trust"],
        )
        self.c["training"]["states"] = self.f["counterfactual_states"]
        self.journal = Journal(journal, self.identity)
        self.model.meter = self.journal.meter
        self._mask()
        if base_state is not None:
            if (
                base_state["identity"]["arm"] != "BASE"
                or base_state["step"] != 0
                or any(base_state["identity"].get(k) != v for k, v in identity.items())
            ):
                raise RuntimeError("JCT_NOT_A_SHARED_INITIAL_STATE")
            self.parent_base_sha = fingerprint(base_state)
            self._restore_values(base_state)
            self._mask()

    def _mask(self):
        treatment = self.config["arms"].get(self.arm, dict(cells=False, connection=False))
        for group in ("cells", "connection"):
            for parameter in getattr(self.model, group).parameters():
                parameter.requires_grad_(treatment[group])
                parameter.grad = None

    def epsilon(self):
        return self.f["epsilon"]

    def state(self):
        return copy.deepcopy(
            dict(
                identity=self.identity,
                parent_base_sha=self.parent_base_sha,
                model=self.model.state_dict(),
                cell_optimizer=self.cell_optimizer.state_dict(),
                connection_optimizer=dict(
                    lifecycle=self.f["connection_optimizer_lifecycle"],
                    last_completed=self.connection_optimizer_state,
                    pending=None,
                ),
                step=self.step,
                data_cursor=self.data_cursor,
                curve=self.curve,
                data_rng=self.data_rng.bit_generator.state,
                route_rng=self.route_rng.get_state(),
                cf_rng=self.cf_rng.get_state(),
                global_cpu_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                schedule=dict(
                    cell_lr=self.f["cell_lr"],
                    connection_lr=self.f["connection_lr"],
                    epsilon=self.epsilon(),
                    temperature=1.0,
                ),
                parameter_mask={n: p.requires_grad for n, p in self.model.named_parameters()},
                work=self.model.work,
                last_proposal=self.last_proposal,
            )
        )

    def _restore_values(self, state):
        if (
            state["connection_optimizer"]["lifecycle"] != self.f["connection_optimizer_lifecycle"]
            or state["connection_optimizer"]["pending"] is not None
        ):
            raise RuntimeError("JCT_CONNECTION_OPTIMIZER_LIFECYCLE_CHANGED")
        expected = dict(
            cell_lr=self.f["cell_lr"],
            connection_lr=self.f["connection_lr"],
            epsilon=self.epsilon(),
            temperature=1.0,
        )
        if state["schedule"] != expected:
            raise RuntimeError("JCT_SCHEDULE_CHANGED")
        self.model.load_state_dict(state["model"], strict=True)
        self.cell_optimizer.load_state_dict(state["cell_optimizer"])
        self.connection_optimizer_state = copy.deepcopy(
            state["connection_optimizer"]["last_completed"]
        )
        self.data_rng.bit_generator.state = copy.deepcopy(state["data_rng"])
        self.route_rng.set_state(state["route_rng"].cpu())
        self.cf_rng.set_state(state["cf_rng"].cpu())
        torch.set_rng_state(state["global_cpu_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if state["data_cursor"] != state["step"] * self.f["batch_size"]:
            raise RuntimeError("JCT_DATA_CURSOR_MISMATCH")
        self.data_cursor = state["data_cursor"]
        self.step = state["step"]
        self.curve = copy.deepcopy(state["curve"])
        self.last_proposal = copy.deepcopy(state["last_proposal"])
        self.model.work = copy.deepcopy(state["work"])

    def restore(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved["schema"] != "minicells.jct001.checkpoint.v1"
            or saved["identity"] != self.identity
            or saved["state_sha256"] != fingerprint(saved["state"])
        ):
            raise RuntimeError("JCT_CHECKPOINT_IDENTITY_OR_STATE_HASH_CHANGED")
        self.journal.verify_prefix(saved["ledger_watermark"])
        state = saved["state"]
        if state["parent_base_sha"] != self.parent_base_sha or state["parameter_mask"] != {
            n: p.requires_grad for n, p in self.model.named_parameters()
        }:
            raise RuntimeError("JCT_BASELINE_OR_MASK_CHANGED")
        self._restore_values(state)
        if fingerprint(self.state()) != saved["state_sha256"]:
            raise RuntimeError("JCT_RESTORED_VALUES_DIFFER")
        return saved

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state()
        payload = dict(
            schema="minicells.jct001.checkpoint.v1",
            identity=self.identity,
            state=state,
            state_sha256=fingerprint(state),
            ledger_watermark=self.journal.watermark(),
        )
        fd, tmp = tempfile.mkstemp(prefix="jct-checkpoint-", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, tmp)
            with open(tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return dict(
            file=path.name,
            sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
            state_sha256=payload["state_sha256"],
            step=self.step,
            cross_session_verified=False,
            scope="LOCAL_CPU_RESTORE_ONLY",
        )

    def advance(self, data, checkpoint, failure_hook=None):
        if self.arm not in self.config["arms"] or self.step >= self.f["logical_updates"]:
            raise RuntimeError("JCT_ENGINEERING_FIXTURE_UPDATE_BOUND")
        summary = self.journal.summary()
        if summary["uncertain_optimizer_calls"] or summary["pending_attempts"]:
            raise RuntimeError("JCT_UNCERTAIN_LEDGER_QUARANTINE")
        treatment = self.config["arms"][self.arm]
        attempt = "attempt-" + str(len(self.journal.events))
        self.journal.append(
            "ATTEMPT_BEGIN",
            dict(id=attempt, logical_step=self.step + 1, examples=self.f["batch_size"]),
        )
        before = snapshot_groups(self.model)
        try:
            self.cell_optimizer.zero_grad(set_to_none=True)
            with self.journal.phase("main_training"):
                selected = batches(data, self.c["engine"], self.data_rng, "cpu")
                x, y = selected[0]
                out = self.model(x, epsilon=self.epsilon(), route_rng=self.route_rng)
                loss = F.cross_entropy(out.logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("JCT_NONFINITE_MAIN_LOSS")
                if treatment["cells"]:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.cells.parameters(), 1.0)
                    self.journal.optimizer_call(
                        "cell", self.cell_optimizer.step, list(self.model.cells.parameters())
                    )
            if treatment["connection"]:
                self.last_proposal = propose(self, dict(data, current_training_batch=selected))
            if failure_hook is not None:
                failure_hook(self)
            deltas = {
                group: group_delta(before, self.model, group) for group in ("cells", "connection")
            }
            for group, allowed in treatment.items():
                if not allowed and (
                    deltas[group] != 0
                    or any(p.grad is not None for p in getattr(self.model, group).parameters())
                ):
                    raise RuntimeError("JCT_FROZEN_GROUP_MUTATED_" + group)
            self.step += 1
            self.data_cursor += len(y)
            row = dict(
                step=self.step,
                loss=float(loss.detach()),
                parameter_delta_l2=deltas,
                connection_admitted=bool(
                    self.last_proposal and self.last_proposal["chosen_scale"] is not None
                ),
                source_examples=self.f["batch_size"],
                input_batch_sha256=fingerprint(selected),
                connection_optimizer_lifecycle=self.f["connection_optimizer_lifecycle"],
            )
            self.curve.append(row)
            with self.journal.phase("checkpoint"):
                self.save(checkpoint)
            self.journal.append("LOGICAL_COMMIT", dict(step=self.step))
            self.journal.append("ATTEMPT_END", dict(id=attempt, logical_step=self.step))
            return row
        except Exception as error:
            self.journal.append(
                "ATTEMPT_ERROR",
                dict(id=attempt, logical_step=self.step + 1, error_type=type(error).__name__),
            )
            raise
