"""Private copies and a bounded, hash-chained lifecycle. No accepted-state training here."""

from copy import deepcopy
from dataclasses import dataclass, field
import torch
from .common import hash_accepted_state, canonical_sha256, AcceptedModelSnapshot
from .training import Trainer, NumericalFailure, NoLearningSignal, ContractFailure
from .integrity import Witness
import traceback
import random
import numpy as np
from .common import tensor_hash
from .sampling import streams

TRANSITIONS = {
    "PROPOSED": {"PROBATION", "EXPIRED"},
    "PROBATION": {"EVALUATED", "EXPIRED", "FAILED"},
    "EVALUATED": {"PROMOTED", "REJECTED", "EXPIRED"},
    "FAILED": set(),
    "PROMOTED": set(),
    "REJECTED": set(),
    "EXPIRED": set(),
}


@dataclass
class Candidate:
    kind: str
    model: object
    parent_sha: str
    parent_registry: str
    trainer: object
    state: str = "PROPOSED"
    journal: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def transition(self, state, details=None):
        if state not in TRANSITIONS[self.state]:
            raise ValueError("invalid Shadow lifecycle transition")
        previous = self.journal[-1]["sha"] if self.journal else self.parent_sha
        event = {"old": self.state, "new": state, "details": details or {}, "previous": previous}
        event["sha"] = canonical_sha256(event)
        self.journal.append(event)
        self.state = state


class Nursery:
    def __init__(self):
        self.candidates = []
        self.births = 0
        self.groups = 0
        self.events = {}

    def propose(self, accepted, kind, accepted_optimizer=None):
        if kind not in ("cell", "connection", "path"):
            raise ValueError("unknown Shadow proposal")
        c = accepted.experiment_config
        if kind == "cell":
            if self.births >= c["shadow"]["max_cell_births"]:
                raise ValueError("Cell birth budget exhausted")
            self.births += 1
        else:
            # One grouped event may contain one connection and one path, never repeated proposals of either kind.
            if (
                any(x.kind == kind for x in self.candidates)
                or self.groups > c["shadow"]["max_connection_path_groups"]
            ):
                raise ValueError("connection/path group budget exhausted")
            if self.groups == 0:
                self.groups = 1
            if self.groups > c["shadow"]["max_connection_path_groups"]:
                raise ValueError("group disabled")
        witness = AcceptedModelSnapshot(accepted)
        branch = deepcopy(accepted)
        if kind == "cell":
            slot = branch.registry.cells
            branch.cells[slot].load_state_dict(branch.cells[0].state_dict())
            branch.registry.activate(slot)
        elif kind == "connection":
            with torch.no_grad():
                branch.controller.edge_bias[0, 0] += c["shadow"]["connection_prior_delta"]
            branch.registry.version += 1
            branch.registry.policy_version += 1
        else:
            plan = list(c["shadow"]["path_plan"])
            if len(plan) > branch.config["budget"]:
                raise ValueError("plan exceeds path budget")
            for depth, a in enumerate(plan):
                if not branch.registry.legal(
                    depth, branch.config["budget"], branch.config["minimum_actions"], "cpu"
                )[a]:
                    raise ValueError("illegal path proposal")
                if a == branch.registry.stop and depth != len(plan) - 1:
                    raise ValueError("path contains work after STOP")
            branch.registry.plan = plan
            branch.registry.version += 1
            branch.registry.policy_version += 1
        # Only candidate Cell and routing dependencies may mature; shared accepted computations remain frozen.
        for name, param in branch.named_parameters():
            param.requires_grad_(
                name.startswith("controller.")
                or (kind == "cell" and name.startswith("cells.%d." % branch.registry.cells))
            )
        if branch.arm == "QF-R2":
            branch.controller.critic.requires_grad_(False)
        trainer = Trainer(branch)
        trainer.rng = streams(
            c["seed"] + 7000 + len(self.candidates) * 100, next(branch.parameters()).device
        )
        # Independent candidate optimizer: no Adam history from changed topology/parameters.
        if accepted_optimizer is not None:
            trainer.step = max(
                (int(v.get("step", 0)) for v in accepted_optimizer.state.values()), default=0
            )
        candidate = Candidate(
            kind, branch, hash_accepted_state(accepted), accepted.registry.fingerprint, trainer
        )
        witness.assert_unchanged(accepted)
        candidate.metrics.update(
            proposed_parent_sha=candidate.parent_sha,
            candidate_uuid=canonical_sha256(
                {
                    "experiment": c["experiment_id"],
                    "arm": accepted.arm,
                    "kind": kind,
                    "parent": candidate.parent_sha,
                    "ordinal": len(self.candidates),
                }
            ),
        )
        self._seed(candidate, "proposal")
        self.candidates.append(candidate)
        return candidate

    def _seed(self, candidate, phase):
        key = {
            "experiment": candidate.model.experiment_config["experiment_id"],
            "arm": candidate.model.arm,
            "uuid": candidate.metrics["candidate_uuid"],
            "phase": phase,
            "parent": candidate.parent_sha,
        }
        seed = int(canonical_sha256(key)[:15], 16)
        candidate.trainer.rng = streams(seed, next(candidate.model.parameters()).device)

    def rebase(self, candidate, accepted, accepted_optimizer):
        if candidate.state != "PROPOSED":
            raise ValueError("only unopened proposal can rebase")
        staging = Nursery()
        replacement = staging.propose(accepted, candidate.kind, accepted_optimizer)
        old = candidate.parent_sha
        candidate.model = replacement.model
        candidate.trainer = replacement.trainer
        candidate.parent_sha = replacement.parent_sha
        candidate.parent_registry = replacement.parent_registry
        candidate.metrics["rebased_parent_sha"] = candidate.parent_sha
        self._seed(candidate, "rebased-probation")
        previous = candidate.journal[-1]["sha"] if candidate.journal else old
        event = {
            "old": "PROPOSED",
            "new": "PROPOSED",
            "details": {
                "reason": "blueprint rebased before probation",
                "old_parent": old,
                "new_parent": candidate.parent_sha,
            },
            "previous": previous,
        }
        event["sha"] = canonical_sha256(event)
        candidate.journal.append(event)

    def probation(self, candidate, accepted, batches, steps, *, accepted_trainer=None):
        if candidate.state != "PROPOSED":
            raise ValueError("probation already started")
        if steps != accepted.experiment_config["shadow"]["private_train_steps"]:
            raise ValueError("private training budget mismatch")
        witness = Witness(accepted, accepted_trainer)
        candidate.transition("PROBATION")
        global_numpy = np.random.get_state()
        global_python = random.getstate()
        global_cpu = torch.get_rng_state()
        global_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for private_step in range(steps):
                try:
                    candidate.trainer.update(batches, role="train")
                except (NumericalFailure, NoLearningSignal, ContractFailure) as error:
                    # OracleUntrusted and generic exceptions deliberately propagate.
                    witness.assert_unchanged(accepted)
                    diagnostic = {
                        "failure_code": "CANDIDATE_NUMERICAL_FAILURE"
                        if isinstance(error, NumericalFailure)
                        else (
                            "CANDIDATE_CONTRACT_FAILURE"
                            if isinstance(error, ContractFailure)
                            else "CANDIDATE_NO_LEARNING_SIGNAL"
                        ),
                        "candidate_type": candidate.kind,
                        "private_step": private_step + 1,
                        "stage": getattr(error, "stage", "backward"),
                        "details": getattr(error, "details", {}),
                        "traceback": traceback.format_exc(),
                        "accepted_witness_ok": True,
                        "parent_sha": candidate.parent_sha,
                        "candidate_uuid": candidate.metrics["candidate_uuid"],
                        "rng_sha": canonical_sha256(
                            {
                                k: tensor_hash(g.get_state())
                                for k, g in candidate.trainer.rng.items()
                            }
                        ),
                        "batch_digest": canonical_sha256(
                            [{"x": tensor_hash(x), "y": tensor_hash(y)} for x, y in batches]
                        ),
                        "next_action": "reject_private_branch_continue_accepted",
                    }
                    candidate.metrics["failure"] = diagnostic
                    candidate.transition("FAILED", diagnostic)
                    return False
            witness.assert_unchanged(accepted)
            return True
        finally:
            np.random.set_state(global_numpy)
            random.setstate(global_python)
            torch.set_rng_state(global_cpu)
            if global_cuda is not None:
                torch.cuda.set_rng_state_all(global_cuda)
            witness.assert_unchanged(accepted)

    def expire_stale(self, accepted):
        for candidate in self.candidates:
            if candidate.state in ("PROPOSED", "PROBATION", "EVALUATED") and (
                candidate.parent_sha != hash_accepted_state(accepted)
                or candidate.parent_registry != accepted.registry.fingerprint
            ):
                candidate.transition("EXPIRED", {"reason": "accepted parent changed"})
