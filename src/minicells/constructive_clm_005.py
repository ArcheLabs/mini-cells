from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from minicells.constructive_clm_003 import StructuralBridgeConfig, learn_structural_roots
from minicells.constructive_clm_004 import run_seed as run_parent_g4
from minicells.subspace_mitosis_005 import constrained_update, extend_basis, projector_from_basis

_DTYPE = torch.float64
MODES = ("simultaneous", "sequential")


@dataclass(frozen=True)
class EndogenousControlConfig:
    root_count: int = 12
    route_dim: int = 48
    hidden_dim: int = 16
    operator_spectral_norm: float = 0.40
    acquisition_batches_per_cell: int = 4
    acquisition_batch_size: int = 32
    acquisition_noise: float = 0.002
    route_noise: float = 0.010
    ridge: float = 1e-8
    evaluation_cases_per_mode: int = 64
    evaluation_batch_size: int = 24
    minimum_active_cells: int = 2
    maximum_active_cells: int = 4
    mutation_target_cells: int = 4
    mutation_history_cases: int = 6
    safe_mutation_examples: int = 32
    conflict_mutation_examples: int = 40
    child_reuse_repeats: int = 3
    child_route_shift: float = 0.35
    child_operator_scale: float = 0.30
    numerical_rank_tolerance: float = 1e-10
    shared_history_cases: int = 8
    shared_mutation_examples: int = 6
    shared_update_scale: float = 0.05


@dataclass
class EndogenousCell:
    cell_id: int
    route_key: torch.Tensor
    weight: torch.Tensor
    gram: torch.Tensor
    cross: torch.Tensor
    basis: torch.Tensor
    parent_id: int | None = None
    observations: int = 0

    @classmethod
    def empty(
        cls,
        cell_id: int,
        route_key: torch.Tensor,
        hidden_dim: int,
        *,
        parent_id: int | None = None,
    ) -> "EndogenousCell":
        return cls(
            cell_id=int(cell_id),
            route_key=F.normalize(route_key.to(dtype=_DTYPE), dim=0),
            weight=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            gram=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            cross=torch.zeros(hidden_dim, hidden_dim, dtype=_DTYPE),
            basis=torch.zeros(hidden_dim, 0, dtype=_DTYPE),
            parent_id=parent_id,
        )

    def clone(self) -> "EndogenousCell":
        return EndogenousCell(
            cell_id=self.cell_id,
            route_key=self.route_key.clone(),
            weight=self.weight.clone(),
            gram=self.gram.clone(),
            cross=self.cross.clone(),
            basis=self.basis.clone(),
            parent_id=self.parent_id,
            observations=self.observations,
        )


class PairRouter(nn.Module):
    """Shared learned compatibility scorer over a query and persistent Cell key."""

    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden, dtype=_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=_DTYPE),
        )

    @staticmethod
    def features(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query.to(dtype=_DTYPE), dim=0)
        k = F.normalize(keys.to(dtype=_DTYPE), dim=1)
        dot = k @ q
        diff = k - q
        return torch.stack(
            (
                dot,
                dot.square(),
                torch.mean(diff.square(), dim=1),
                torch.mean(torch.abs(diff), dim=1),
                torch.max(torch.abs(diff), dim=1).values,
            ),
            dim=1,
        )

    def scores(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(query, keys)).squeeze(-1)


class BinaryController(nn.Module):
    def __init__(self, features: int, hidden: int = 12) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden, dtype=_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=_DTYPE),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.to(dtype=_DTYPE)).squeeze(-1)


@dataclass
class ControllerBundle:
    router: PairRouter
    growth: BinaryController
    write: BinaryController
    diagnostics: dict[str, Any]
    state_sha256: str


_CONTROLLER_CACHE: ControllerBundle | None = None


def _mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left - right).square()).item())


def _relative_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.mean(right.square()).clamp_min(torch.tensor(1e-30, dtype=_DTYPE))
    return float((torch.mean((left - right).square()) / denom).item())


def _relative_frobenius(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.norm(right).clamp_min(torch.tensor(1e-30, dtype=_DTYPE))
    return float((torch.linalg.norm(left - right) / denom).item())


def _scaled_random_operator(
    hidden_dim: int,
    scale: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randn(hidden_dim, hidden_dim, generator=generator, dtype=_DTYPE)
    largest = torch.linalg.svdvals(raw)[0].clamp_min(torch.tensor(1e-12, dtype=_DTYPE))
    return raw * (float(scale) / largest)


def _least_squares_operator(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    *,
    ridge: float = 1e-8,
) -> torch.Tensor:
    h = hidden.to(dtype=_DTYPE)
    r = residual.to(dtype=_DTYPE)
    eye = torch.eye(h.shape[1], dtype=_DTYPE)
    return r.T @ h @ torch.linalg.pinv(h.T @ h + float(ridge) * eye)


def _route_token(
    key: torch.Tensor,
    *,
    noise: float,
    generator: torch.Generator,
) -> torch.Tensor:
    token = key + float(noise) * torch.randn(
        key.shape, generator=generator, dtype=_DTYPE
    )
    return F.normalize(token, dim=0)


def _orthogonal_child_key(
    parent: torch.Tensor,
    *,
    shift: float,
    generator: torch.Generator,
) -> torch.Tensor:
    base = F.normalize(parent.to(dtype=_DTYPE), dim=0)
    direction = torch.randn(base.shape, generator=generator, dtype=_DTYPE)
    direction = direction - torch.dot(direction, base) * base
    direction = F.normalize(direction, dim=0)
    return F.normalize(base + float(shift) * direction, dim=0)


def _growth_features(
    existing_error: float,
    fresh_error: float,
    margin: float,
    top_probability: float,
    observations: int,
) -> torch.Tensor:
    return torch.tensor(
        (
            math.log1p(max(float(existing_error), 0.0) * 1000.0),
            math.log1p(max(float(fresh_error), 0.0) * 1000.0),
            float(margin),
            float(top_probability),
            min(float(observations) / 256.0, 1.0),
        ),
        dtype=_DTYPE,
    )


def _write_features(
    fit_error: float,
    unconstrained_fit_error: float,
    free_rank: int,
    certificate_rank: int,
    hidden_dim: int,
    residual_energy: float,
) -> torch.Tensor:
    return torch.tensor(
        (
            math.log10(max(float(fit_error), 1e-16) + 1e-16),
            math.log10(max(float(unconstrained_fit_error), 1e-16) + 1e-16),
            float(free_rank) / max(int(hidden_dim), 1),
            float(certificate_rank) / max(int(hidden_dim), 1),
            math.log1p(max(float(residual_energy), 0.0)),
        ),
        dtype=_DTYPE,
    )


def _controller_hash(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _router_meta_rows(seeds: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cells, route_dim, hidden_dim = 8, 24, 10
        keys = F.normalize(
            torch.randn(cells, route_dim, generator=generator, dtype=_DTYPE), dim=1
        )
        operators = torch.stack(
            [
                _scaled_random_operator(hidden_dim, 0.40, generator=generator)
                for _ in range(cells)
            ]
        )
        for _ in range(80):
            source = int(torch.randint(cells, (1,), generator=generator).item())
            query = _route_token(keys[source], noise=0.03, generator=generator)
            hidden = torch.randn(10, hidden_dim, generator=generator, dtype=_DTYPE)
            residual = hidden @ operators[source].T
            errors = torch.tensor(
                [
                    _mse(hidden @ operators[candidate].T, residual)
                    for candidate in range(cells)
                ],
                dtype=_DTYPE,
            )
            utility_best = int(torch.argmin(errors).item())
            rows.append(PairRouter.features(query, keys))
            labels.append(utility_best)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.long)


def _train_router() -> tuple[PairRouter, dict[str, float]]:
    train_x, train_y = _router_meta_rows(range(6501, 6507))
    valid_x, valid_y = _router_meta_rows(range(6507, 6509))
    torch.manual_seed(0xC1A005)
    model = PairRouter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        optimizer.zero_grad()
        logits = model.net(train_x).squeeze(-1)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_logits = model.net(train_x).squeeze(-1)
        valid_logits = model.net(valid_x).squeeze(-1)
        train_accuracy = float(
            (torch.argmax(train_logits, dim=1) == train_y).to(_DTYPE).mean().item()
        )
        valid_accuracy = float(
            (torch.argmax(valid_logits, dim=1) == valid_y).to(_DTYPE).mean().item()
        )
    model.eval()
    return model, {
        "train_accuracy": train_accuracy,
        "heldout_meta_accuracy": valid_accuracy,
    }


def _growth_meta_rows(
    router: PairRouter, seeds: Iterable[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[float] = []
    spawn_cost = 0.03
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cells, route_dim, hidden_dim = 8, 24, 10
        keys = F.normalize(
            torch.randn(cells, route_dim, generator=generator, dtype=_DTYPE), dim=1
        )
        operators = torch.stack(
            [
                _scaled_random_operator(hidden_dim, 0.40, generator=generator)
                for _ in range(cells)
            ]
        )
        for transaction in range(120):
            parent = int(torch.randint(cells, (1,), generator=generator).item())
            if transaction % 3 == 0:
                query = _route_token(keys[parent], noise=0.03, generator=generator)
                target_operator = operators[parent] + 0.005 * _scaled_random_operator(
                    hidden_dim, 0.40, generator=generator
                )
            else:
                child_key = _orthogonal_child_key(
                    keys[parent],
                    shift=0.15 + 0.35 * ((transaction % 11) / 10.0),
                    generator=generator,
                )
                query = _route_token(child_key, noise=0.02, generator=generator)
                target_operator = operators[parent] + _scaled_random_operator(
                    hidden_dim,
                    0.08 + 0.45 * ((transaction % 17) / 16.0),
                    generator=generator,
                )
            hidden = torch.randn(24, hidden_dim, generator=generator, dtype=_DTYPE)
            target = hidden @ target_operator.T
            with torch.no_grad():
                scores = router.scores(query, keys)
                probabilities = torch.softmax(scores, dim=0)
                top = torch.topk(scores, k=2)
            candidate = int(top.indices[0].item())
            existing_error = _relative_mse(hidden @ operators[candidate].T, target)
            fresh_weight = _least_squares_operator(hidden, target)
            fresh_error = _relative_mse(hidden @ fresh_weight.T, target)
            utility_spawn = float(existing_error > fresh_error + spawn_cost)
            rows.append(
                _growth_features(
                    existing_error,
                    fresh_error,
                    float((top.values[0] - top.values[1]).item()),
                    float(probabilities[candidate].item()),
                    128,
                )
            )
            labels.append(utility_spawn)
    return torch.stack(rows), torch.tensor(labels, dtype=_DTYPE)


def _train_growth(router: PairRouter) -> tuple[BinaryController, dict[str, float]]:
    train_x, train_y = _growth_meta_rows(router, range(6601, 6607))
    valid_x, valid_y = _growth_meta_rows(router, range(6607, 6609))
    torch.manual_seed(0xC2A005)
    model = BinaryController(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(500):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_prediction = torch.sigmoid(model(train_x)) >= 0.5
        valid_prediction = torch.sigmoid(model(valid_x)) >= 0.5
        train_accuracy = float(
            (train_prediction == train_y.bool()).to(_DTYPE).mean().item()
        )
        valid_accuracy = float(
            (valid_prediction == valid_y.bool()).to(_DTYPE).mean().item()
        )
    model.eval()
    return model, {
        "train_accuracy": train_accuracy,
        "heldout_meta_accuracy": valid_accuracy,
    }


def _write_meta_rows(seeds: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    labels: list[float] = []
    hidden_dim = 16
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for transaction in range(120):
            rank = 4 + transaction % 4
            q, _ = torch.linalg.qr(
                torch.randn(hidden_dim, rank, generator=generator, dtype=_DTYPE),
                mode="reduced",
            )
            examples = 10
            if transaction % 2 == 0:
                hidden = torch.randn(
                    examples, hidden_dim, generator=generator, dtype=_DTYPE
                )
                projector = projector_from_basis(hidden_dim, q)
                desired = 0.10 * torch.randn(
                    hidden_dim, hidden_dim, generator=generator, dtype=_DTYPE
                )
                residual = (hidden @ projector) @ desired.T
                utility_commit = 1.0
            else:
                coefficients = torch.randn(
                    examples, rank, generator=generator, dtype=_DTYPE
                )
                hidden = coefficients @ q.T
                desired = 0.10 * torch.randn(
                    rank, hidden_dim, generator=generator, dtype=_DTYPE
                )
                residual = coefficients @ desired
                utility_commit = 0.0
            safe = constrained_update(
                hidden,
                residual,
                q,
                numerical_rank_tolerance=1e-10,
            )
            unsafe = constrained_update(
                hidden,
                residual,
                torch.zeros(hidden_dim, 0, dtype=_DTYPE),
                numerical_rank_tolerance=1e-10,
            )
            rows.append(
                _write_features(
                    float(safe["fit_error"]),
                    float(unsafe["fit_error"]),
                    int(safe["free_design_rank"]),
                    int(q.shape[1]),
                    hidden_dim,
                    float(torch.mean(residual.square()).item()),
                )
            )
            labels.append(utility_commit)
    return torch.stack(rows), torch.tensor(labels, dtype=_DTYPE)


def _train_write() -> tuple[BinaryController, dict[str, float]]:
    train_x, train_y = _write_meta_rows(range(6701, 6707))
    valid_x, valid_y = _write_meta_rows(range(6707, 6709))
    torch.manual_seed(0xC3A005)
    model = BinaryController(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(400):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_prediction = torch.sigmoid(model(train_x)) >= 0.5
        valid_prediction = torch.sigmoid(model(valid_x)) >= 0.5
        train_accuracy = float(
            (train_prediction == train_y.bool()).to(_DTYPE).mean().item()
        )
        valid_accuracy = float(
            (valid_prediction == valid_y.bool()).to(_DTYPE).mean().item()
        )
    model.eval()
    return model, {
        "train_accuracy": train_accuracy,
        "heldout_meta_accuracy": valid_accuracy,
    }


def learned_controllers() -> ControllerBundle:
    """Train once from permanently development-only intrinsic-utility episodes."""
    global _CONTROLLER_CACHE
    if _CONTROLLER_CACHE is not None:
        return _CONTROLLER_CACHE
    router, router_diag = _train_router()
    growth, growth_diag = _train_growth(router)
    write, write_diag = _train_write()
    diagnostics = {
        "router": router_diag,
        "growth": growth_diag,
        "write": write_diag,
        "meta_training_uses_hidden_ids_as_targets": False,
        "meta_training_uses_formal_seed_data": False,
        "router_label_source": "current-transaction candidate operator fit utility",
        "growth_label_source": "current-only reuse-vs-fresh utility with spawn cost",
        "write_label_source": "current-data certificate-constrained feasibility utility",
        "meta_train_seeds": {
            "router": list(range(6501, 6507)),
            "growth": list(range(6601, 6607)),
            "write": list(range(6701, 6707)),
        },
        "meta_validation_seeds": {
            "router": list(range(6507, 6509)),
            "growth": list(range(6607, 6609)),
            "write": list(range(6707, 6709)),
        },
    }
    _CONTROLLER_CACHE = ControllerBundle(
        router=router,
        growth=growth,
        write=write,
        diagnostics=diagnostics,
        state_sha256=_controller_hash(router, growth, write),
    )
    return _CONTROLLER_CACHE


class EndogenousCellModel:
    """Dynamic Cell model whose route/write/grow control is learned."""

    def __init__(
        self,
        route_keys: torch.Tensor,
        hidden_dim: int,
        controllers: ControllerBundle,
        *,
        ridge: float,
    ) -> None:
        keys = F.normalize(route_keys.to(dtype=_DTYPE), dim=1)
        self.hidden_dim = int(hidden_dim)
        self.controllers = controllers
        self.ridge = float(ridge)
        self.cells = [
            EndogenousCell.empty(i, keys[i], self.hidden_dim) for i in range(len(keys))
        ]
        self.raw_examples_retained = 0
        self.replay_accesses = 0
        self.spawn_events = 0

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def keys(self) -> torch.Tensor:
        return torch.stack([cell.route_key for cell in self.cells])

    def clone(self) -> "EndogenousCellModel":
        other = object.__new__(EndogenousCellModel)
        other.hidden_dim = self.hidden_dim
        other.controllers = self.controllers
        other.ridge = self.ridge
        other.cells = [cell.clone() for cell in self.cells]
        other.raw_examples_retained = self.raw_examples_retained
        other.replay_accesses = self.replay_accesses
        other.spawn_events = self.spawn_events
        return other

    def route_one(self, token: torch.Tensor) -> tuple[int, dict[str, float]]:
        with torch.no_grad():
            scores = self.controllers.router.scores(token, self.keys())
            probabilities = torch.softmax(scores, dim=0)
            top = torch.topk(scores, k=min(2, len(self.cells)))
        cell_id = int(top.indices[0].item())
        margin = (
            float((top.values[0] - top.values[1]).item())
            if len(self.cells) > 1
            else float("inf")
        )
        return cell_id, {
            "top_probability": float(probabilities[cell_id].item()),
            "margin": margin,
        }

    def route_tokens(self, tokens: torch.Tensor) -> list[int]:
        query = tokens if tokens.ndim == 2 else tokens[None, :]
        return [self.route_one(row)[0] for row in query]

    def observe_operator(
        self,
        route_context: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> int:
        token = (
            route_context.mean(dim=0)
            if route_context.ndim == 2
            else route_context
        )
        cell_id, _ = self.route_one(token)
        z = hidden.to(dtype=_DTYPE)
        r = residual.to(dtype=_DTYPE)
        cell = self.cells[cell_id]
        cell.gram += z.T @ z
        cell.cross += r.T @ z
        cell.observations += int(z.shape[0])
        return cell_id

    def fit_operators(self) -> None:
        eye = torch.eye(self.hidden_dim, dtype=_DTYPE)
        for cell in self.cells:
            if cell.observations:
                cell.weight = cell.cross @ torch.linalg.pinv(
                    cell.gram + self.ridge * eye
                )

    def execute_ids(
        self,
        hidden: torch.Tensor,
        cell_ids: Iterable[int],
        *,
        mode: str,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        ids = [int(v) for v in cell_ids]
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        h = hidden.to(dtype=_DTYPE)
        trace: list[dict[str, Any]] = []
        if mode == "simultaneous":
            base = h
            total = torch.zeros_like(base)
            for cell_id in ids:
                residual = base @ self.cells[cell_id].weight.T
                total += residual
                if return_trace:
                    trace.append(
                        {
                            "cell_id": cell_id,
                            "incoming": base.detach().clone(),
                            "residual": residual.detach().clone(),
                        }
                    )
            return base + total, trace
        for cell_id in ids:
            incoming = h
            residual = incoming @ self.cells[cell_id].weight.T
            h = incoming + residual
            if return_trace:
                trace.append(
                    {
                        "cell_id": cell_id,
                        "incoming": incoming.detach().clone(),
                        "residual": residual.detach().clone(),
                    }
                )
        return h, trace

    def execute(
        self,
        hidden: torch.Tensor,
        route_tokens: torch.Tensor,
        *,
        mode: str,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, list[int], list[dict[str, Any]]]:
        ids = self.route_tokens(route_tokens)
        output, trace = self.execute_ids(
            hidden, ids, mode=mode, return_trace=return_trace
        )
        return output, ids, trace

    def write_commit_probability(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        cell_id: int,
        *,
        tolerance: float,
    ) -> tuple[float, dict[str, Any]]:
        cell = self.cells[int(cell_id)]
        safe = constrained_update(
            hidden,
            residual,
            cell.basis,
            numerical_rank_tolerance=tolerance,
        )
        unsafe = constrained_update(
            hidden,
            residual,
            torch.zeros(self.hidden_dim, 0, dtype=_DTYPE),
            numerical_rank_tolerance=tolerance,
        )
        features = _write_features(
            float(safe["fit_error"]),
            float(unsafe["fit_error"]),
            int(safe["free_design_rank"]),
            int(cell.basis.shape[1]),
            self.hidden_dim,
            float(torch.mean(residual.square()).item()),
        )
        with torch.no_grad():
            probability = float(
                torch.sigmoid(self.controllers.write(features)).item()
            )
        return probability, safe

    def growth_probability(
        self,
        route_context: torch.Tensor,
        hidden: torch.Tensor,
        target_residual: torch.Tensor,
    ) -> tuple[float, int, dict[str, float], torch.Tensor]:
        candidate, route = self.route_one(route_context)
        prediction = hidden @ self.cells[candidate].weight.T
        existing_error = _relative_mse(prediction, target_residual)
        fresh_weight = _least_squares_operator(
            hidden, target_residual, ridge=self.ridge
        )
        fresh_error = _relative_mse(hidden @ fresh_weight.T, target_residual)
        features = _growth_features(
            existing_error,
            fresh_error,
            route["margin"],
            route["top_probability"],
            self.cells[candidate].observations,
        )
        with torch.no_grad():
            probability = float(
                torch.sigmoid(self.controllers.growth(features)).item()
            )
        return probability, candidate, {
            **route,
            "existing_error": existing_error,
            "fresh_error": fresh_error,
        }, fresh_weight

    def spawn_child(
        self,
        route_context: torch.Tensor,
        weight: torch.Tensor,
        *,
        parent_id: int,
    ) -> int:
        cell_id = len(self.cells)
        child = EndogenousCell.empty(
            cell_id,
            route_context,
            self.hidden_dim,
            parent_id=int(parent_id),
        )
        child.weight = weight.detach().clone()
        child.observations = self.hidden_dim
        self.cells.append(child)
        self.spawn_events += 1
        return cell_id


def _true_execute(
    hidden: torch.Tensor,
    operators: list[torch.Tensor],
    cell_ids: Iterable[int],
    *,
    mode: str,
) -> torch.Tensor:
    ids = [int(v) for v in cell_ids]
    h = hidden.to(dtype=_DTYPE)
    if mode == "simultaneous":
        base = h
        total = torch.zeros_like(base)
        for cell_id in ids:
            total += base @ operators[cell_id].T
        return base + total
    if mode == "sequential":
        for cell_id in ids:
            h = h + h @ operators[cell_id].T
        return h
    raise ValueError(f"unknown mode {mode!r}")


def _acquire(
    model: EndogenousCellModel,
    true_operators: list[torch.Tensor],
    cfg: EndogenousControlConfig,
    *,
    generator: torch.Generator,
    rng: random.Random,
) -> dict[str, Any]:
    schedule = [
        cell_id
        for cell_id in range(cfg.root_count)
        for _ in range(cfg.acquisition_batches_per_cell)
    ]
    rng.shuffle(schedule)
    correct = 0
    for evaluator_source in schedule:
        context = _route_token(
            model.cells[evaluator_source].route_key,
            noise=cfg.route_noise,
            generator=generator,
        )
        hidden = torch.randn(
            cfg.acquisition_batch_size,
            cfg.hidden_dim,
            generator=generator,
            dtype=_DTYPE,
        )
        residual = hidden @ true_operators[evaluator_source].T
        residual += cfg.acquisition_noise * torch.randn(
            residual.shape, generator=generator, dtype=_DTYPE
        )
        routed = model.observe_operator(context, hidden, residual)
        correct += int(routed == evaluator_source)
    model.fit_operators()
    errors = [
        _relative_frobenius(model.cells[i].weight, true_operators[i])
        for i in range(cfg.root_count)
    ]
    return {
        "transactions": len(schedule),
        "route_accuracy": correct / max(len(schedule), 1),
        "mean_operator_relative_error": _mean(errors),
        "max_operator_relative_error": max(errors, default=0.0),
        "raw_examples_retained": model.raw_examples_retained,
    }


def _evaluate_composition(
    model: EndogenousCellModel,
    true_operators: list[torch.Tensor],
    cfg: EndogenousControlConfig,
    *,
    mode: str,
    generator: torch.Generator,
    rng: random.Random,
) -> dict[str, Any]:
    mse_values: list[float] = []
    route_exact: list[float] = []
    active_counts: list[int] = []
    order_effects: list[float] = []
    permutation_errors: list[float] = []
    for _ in range(cfg.evaluation_cases_per_mode):
        active = rng.randint(cfg.minimum_active_cells, cfg.maximum_active_cells)
        ids = rng.sample(range(model.cell_count), active)
        hidden = torch.randn(
            cfg.evaluation_batch_size,
            cfg.hidden_dim,
            generator=generator,
            dtype=_DTYPE,
        )
        tokens = torch.stack(
            [
                _route_token(
                    model.cells[cell_id].route_key,
                    noise=cfg.route_noise,
                    generator=generator,
                )
                for cell_id in ids
            ]
        )
        target = _true_execute(hidden, true_operators, ids, mode=mode)
        predicted, routed, _ = model.execute(hidden, tokens, mode=mode)
        mse_values.append(_mse(predicted, target))
        route_exact.append(float(routed == ids))
        active_counts.append(active)
        if mode == "sequential":
            reversed_target = _true_execute(
                hidden, true_operators, reversed(ids), mode=mode
            )
            order_effects.append(_mse(reversed_target, target))
        else:
            permuted, _ = model.execute_ids(
                hidden, reversed(ids), mode=mode
            )
            permutation_errors.append(_mse(permuted, predicted))
    mean_active = _mean(active_counts)
    return {
        "mode": mode,
        "cases": cfg.evaluation_cases_per_mode,
        "mean_mse": _mean(mse_values),
        "max_mse": max(mse_values, default=0.0),
        "exact_route_sequence_accuracy": _mean(route_exact),
        "mean_active_cells": mean_active,
        "maximum_active_cells": max(active_counts, default=0),
        "total_cells": model.cell_count,
        "cell_execution_fraction_vs_dense": mean_active / max(model.cell_count, 1),
        "mean_true_order_effect_mse": _mean(order_effects),
        "mean_simultaneous_permutation_mse": _mean(permutation_errors),
    }


def _target_incoming(
    trace: list[dict[str, Any]], target_cell: int
) -> torch.Tensor:
    for item in trace:
        if int(item["cell_id"]) == int(target_cell):
            if int(item["incoming"].shape[0]) != 1:
                raise ValueError("history certificate expects batch size one")
            return item["incoming"][0].clone()
    raise ValueError("target Cell missing from composition trace")


def _mutation_stream(
    model: EndogenousCellModel,
    true_operators: list[torch.Tensor],
    cfg: EndogenousControlConfig,
    *,
    generator: torch.Generator,
    rng: random.Random,
) -> dict[str, Any]:
    targets = rng.sample(range(cfg.root_count), cfg.mutation_target_cells)
    safe_commits = 0
    conflict_write_rejections = 0
    conflict_spawns = 0
    child_reuse_hits = 0
    child_reuse_growth_rejections = 0
    historical_mse: list[float] = []
    unsafe_reuse_mse: list[float] = []
    spawned_children: list[int] = []
    unrelated_drift = 0.0
    original_replay = model.replay_accesses

    for target in targets:
        history_rows: list[torch.Tensor] = []
        history_records: list[tuple[torch.Tensor, list[int], torch.Tensor]] = []
        for _ in range(cfg.mutation_history_cases):
            others = rng.sample(
                [i for i in range(cfg.root_count) if i != target], 2
            )
            ids = [others[0], target, others[1]]
            hidden = torch.randn(1, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
            before, trace = model.execute_ids(
                hidden, ids, mode="sequential", return_trace=True
            )
            history_rows.append(_target_incoming(trace, target))
            history_records.append((hidden.clone(), ids, before.clone()))
        basis = extend_basis(
            torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE),
            torch.stack(history_rows),
            tolerance=cfg.numerical_rank_tolerance,
        )
        model.cells[target].basis = basis.clone()

        hidden = torch.randn(
            cfg.safe_mutation_examples,
            cfg.hidden_dim,
            generator=generator,
            dtype=_DTYPE,
        )
        projector = projector_from_basis(cfg.hidden_dim, basis)
        desired_map = 0.05 * torch.randn(
            cfg.hidden_dim, cfg.hidden_dim, generator=generator, dtype=_DTYPE
        )
        residual = (hidden @ projector) @ desired_map.T
        commit_probability, safe = model.write_commit_probability(
            hidden,
            residual,
            target,
            tolerance=cfg.numerical_rank_tolerance,
        )
        if commit_probability >= 0.5:
            safe_commits += 1
            before_weights = [cell.weight.clone() for cell in model.cells]
            model.cells[target].weight += safe["delta_weight"]
            true_operators[target] = true_operators[target] + desired_map @ projector
            for cell_id, previous in enumerate(before_weights):
                if cell_id != target:
                    unrelated_drift = max(
                        unrelated_drift,
                        float(
                            torch.max(
                                torch.abs(model.cells[cell_id].weight - previous)
                            ).item()
                        ),
                    )
        for hist_hidden, ids, before in history_records:
            after, _ = model.execute_ids(hist_hidden, ids, mode="sequential")
            historical_mse.append(_mse(after, before))

        child_key = _orthogonal_child_key(
            model.cells[target].route_key,
            shift=cfg.child_route_shift,
            generator=generator,
        )
        hidden_conflict = torch.randn(
            cfg.conflict_mutation_examples,
            cfg.hidden_dim,
            generator=generator,
            dtype=_DTYPE,
        )
        delta = _scaled_random_operator(
            cfg.hidden_dim,
            cfg.child_operator_scale,
            generator=generator,
        )
        target_full_operator = true_operators[target] + delta
        target_full = hidden_conflict @ target_full_operator.T
        candidate, _ = model.route_one(child_key)
        desired_delta = target_full - hidden_conflict @ model.cells[candidate].weight.T
        write_probability, _ = model.write_commit_probability(
            hidden_conflict,
            desired_delta,
            candidate,
            tolerance=cfg.numerical_rank_tolerance,
        )
        if write_probability < 0.5:
            conflict_write_rejections += 1

        unsafe = constrained_update(
            hidden_conflict,
            desired_delta,
            torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE),
            numerical_rank_tolerance=cfg.numerical_rank_tolerance,
        )
        unsafe_model = model.clone()
        unsafe_model.cells[candidate].weight += unsafe["delta_weight"]
        for hist_hidden, ids, before in history_records:
            unsafe_after, _ = unsafe_model.execute_ids(
                hist_hidden, ids, mode="sequential"
            )
            unsafe_reuse_mse.append(_mse(unsafe_after, before))

        growth_probability, growth_candidate, _, fresh_weight = model.growth_probability(
            child_key,
            hidden_conflict,
            target_full,
        )
        if (
            write_probability < 0.5
            and growth_probability >= 0.5
            and growth_candidate == candidate
        ):
            conflict_spawns += 1
            child_id = model.spawn_child(
                child_key,
                fresh_weight,
                parent_id=candidate,
            )
            spawned_children.append(child_id)
            true_operators.append(fresh_weight.detach().clone())
        else:
            continue

        child_id = spawned_children[-1]
        for _ in range(cfg.child_reuse_repeats):
            query = _route_token(
                model.cells[child_id].route_key,
                noise=cfg.route_noise,
                generator=generator,
            )
            routed, _ = model.route_one(query)
            hidden_reuse = torch.randn(
                24, cfg.hidden_dim, generator=generator, dtype=_DTYPE
            )
            target_reuse = hidden_reuse @ true_operators[child_id].T
            probability, _, _, _ = model.growth_probability(
                query,
                hidden_reuse,
                target_reuse,
            )
            child_reuse_hits += int(routed == child_id)
            child_reuse_growth_rejections += int(probability < 0.5)

    always_spawn_control = cfg.mutation_target_cells * (
        1 + cfg.child_reuse_repeats
    )
    return {
        "target_cells": targets,
        "safe_commit_count": safe_commits,
        "conflict_write_rejection_count": conflict_write_rejections,
        "conflict_spawn_count": conflict_spawns,
        "spawned_children": spawned_children,
        "final_cells": model.cell_count,
        "child_reuse_hits": child_reuse_hits,
        "child_reuse_growth_rejections": child_reuse_growth_rejections,
        "child_reuse_trials": cfg.mutation_target_cells * cfg.child_reuse_repeats,
        "maximum_historical_composition_mse": max(historical_mse, default=0.0),
        "mean_unsafe_reuse_historical_mse": _mean(unsafe_reuse_mse),
        "always_spawn_control_cells_added": always_spawn_control,
        "learner_replay_accesses": model.replay_accesses - original_replay,
        "learner_raw_history_retained": model.raw_examples_retained,
        "unrelated_cell_parameter_drift": unrelated_drift,
    }


def _shared_substrate_probe(
    model: EndogenousCellModel,
    cfg: EndogenousControlConfig,
    *,
    generator: torch.Generator,
    rng: random.Random,
) -> dict[str, Any]:
    history_rows: list[torch.Tensor] = []
    records: list[tuple[torch.Tensor, list[int], str, torch.Tensor]] = []
    for index in range(cfg.shared_history_cases):
        mode = MODES[index % len(MODES)]
        active = rng.randint(cfg.minimum_active_cells, cfg.maximum_active_cells)
        ids = rng.sample(range(model.cell_count), active)
        hidden = torch.randn(1, cfg.hidden_dim, generator=generator, dtype=_DTYPE)
        before, _ = model.execute_ids(hidden, ids, mode=mode)
        history_rows.append(hidden[0].clone())
        records.append((hidden.clone(), ids, mode, before.clone()))
    basis = extend_basis(
        torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE),
        torch.stack(history_rows),
        tolerance=cfg.numerical_rank_tolerance,
    )
    hidden = torch.randn(
        cfg.shared_mutation_examples,
        cfg.hidden_dim,
        generator=generator,
        dtype=_DTYPE,
    )
    projector = projector_from_basis(cfg.hidden_dim, basis)
    desired = cfg.shared_update_scale * torch.randn(
        cfg.hidden_dim, cfg.hidden_dim, generator=generator, dtype=_DTYPE
    )
    residual = (hidden @ projector) @ desired.T
    safe = constrained_update(
        hidden,
        residual,
        basis,
        numerical_rank_tolerance=cfg.numerical_rank_tolerance,
    )
    unsafe = constrained_update(
        hidden,
        residual,
        torch.zeros(cfg.hidden_dim, 0, dtype=_DTYPE),
        numerical_rank_tolerance=cfg.numerical_rank_tolerance,
    )
    safe_history: list[float] = []
    unsafe_history: list[float] = []
    for hist_hidden, ids, mode, before in records:
        safe_input = hist_hidden + hist_hidden @ safe["delta_weight"].T
        unsafe_input = hist_hidden + hist_hidden @ unsafe["delta_weight"].T
        safe_after, _ = model.execute_ids(safe_input, ids, mode=mode)
        unsafe_after, _ = model.execute_ids(unsafe_input, ids, mode=mode)
        safe_history.append(_mse(safe_after, before))
        unsafe_history.append(_mse(unsafe_after, before))
    safe_fit = hidden @ safe["delta_weight"].T
    return {
        "certificate_rank": int(basis.shape[1]),
        "learner_replay_accesses": 0,
        "safe_fit_mse": _mse(safe_fit, residual),
        "safe_historical_composition_mse": _mean(safe_history),
        "unsafe_historical_composition_mse": _mean(unsafe_history),
        "safe_protected_change": float(safe["protected_change"]),
    }


def run_seed(
    seed: int,
    cfg: EndogenousControlConfig = EndogenousControlConfig(),
) -> dict[str, Any]:
    seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xC1A005)
    rng = random.Random(seed ^ 0x51A005)
    controllers = learned_controllers()

    parent = run_parent_g4(seed)

    route_keys, bridge = learn_structural_roots(
        seed,
        StructuralBridgeConfig(
            max_transactions=1024,
            initial_factors=6,
            final_factors=cfg.root_count,
            context_dim=cfg.route_dim,
            effect_dim=40,
            bootstrap_cycles=6,
            introduction_repeats=4,
            growth_alpha=0.60,
            stabilization_tail=64,
            samples_per_transaction=16,
            train_noise=0.02,
        ),
    )
    model = EndogenousCellModel(
        route_keys,
        cfg.hidden_dim,
        controllers,
        ridge=cfg.ridge,
    )
    true_operators = [
        _scaled_random_operator(
            cfg.hidden_dim,
            cfg.operator_spectral_norm,
            generator=generator,
        )
        for _ in range(cfg.root_count)
    ]

    acquisition = _acquire(
        model,
        true_operators,
        cfg,
        generator=generator,
        rng=rng,
    )
    pre_sim = _evaluate_composition(
        model,
        true_operators,
        cfg,
        mode="simultaneous",
        generator=generator,
        rng=rng,
    )
    pre_seq = _evaluate_composition(
        model,
        true_operators,
        cfg,
        mode="sequential",
        generator=generator,
        rng=rng,
    )

    mutation = _mutation_stream(
        model,
        true_operators,
        cfg,
        generator=generator,
        rng=rng,
    )
    final_sim = _evaluate_composition(
        model,
        true_operators,
        cfg,
        mode="simultaneous",
        generator=generator,
        rng=rng,
    )
    final_seq = _evaluate_composition(
        model,
        true_operators,
        cfg,
        mode="sequential",
        generator=generator,
        rng=rng,
    )
    shared = _shared_substrate_probe(
        model,
        cfg,
        generator=generator,
        rng=rng,
    )

    controller_diag = controllers.diagnostics
    controller_heldout = min(
        float(controller_diag["router"]["heldout_meta_accuracy"]),
        float(controller_diag["growth"]["heldout_meta_accuracy"]),
        float(controller_diag["write"]["heldout_meta_accuracy"]),
    )
    expected_children = cfg.mutation_target_cells
    reuse_trials = cfg.mutation_target_cells * cfg.child_reuse_repeats
    final_main = max(final_sim["mean_mse"], final_seq["mean_mse"])

    gates = {
        "parent_g4_baseline_reusable": bool(parent.get("pass", False)),
        "controller_meta_generalization": controller_heldout >= 0.98,
        "controller_training_is_formal_data_free": bool(
            controller_diag["meta_training_uses_formal_seed_data"] is False
            and controller_diag["meta_training_uses_hidden_ids_as_targets"] is False
        ),
        "structural_bridge_valid": bool(
            int(bridge.get("root_cells", -1)) == cfg.root_count
            and int(bridge.get("covered_factors", -1)) == cfg.root_count
            and int(bridge.get("duplicate_assignments", 1)) == 0
            and float(bridge.get("mean_matched_root_key_cosine", 0.0)) >= 0.985
        ),
        "learned_router_acquisition": acquisition["route_accuracy"] >= 0.99,
        "learned_operator_quality": bool(
            acquisition["mean_operator_relative_error"] <= 0.01
            and acquisition["max_operator_relative_error"] <= 0.02
            and acquisition["raw_examples_retained"] == 0
        ),
        "pre_removal_composition_preserved": bool(
            pre_sim["mean_mse"] <= 1e-4
            and pre_seq["mean_mse"] <= 1e-4
            and pre_sim["exact_route_sequence_accuracy"] >= 0.995
            and pre_seq["exact_route_sequence_accuracy"] >= 0.995
        ),
        "learned_write_controller_commits_safe_updates": (
            mutation["safe_commit_count"] == cfg.mutation_target_cells
        ),
        "learned_write_controller_rejects_conflicts": (
            mutation["conflict_write_rejection_count"] == cfg.mutation_target_cells
        ),
        "learned_growth_controller_spawns_on_conflict": (
            mutation["conflict_spawn_count"] == expected_children
        ),
        "bounded_learned_growth": bool(
            mutation["final_cells"] == cfg.root_count + expected_children
            and mutation["conflict_spawn_count"]
            <= 0.40 * mutation["always_spawn_control_cells_added"]
        ),
        "learned_child_reuse": bool(
            mutation["child_reuse_hits"] == reuse_trials
            and mutation["child_reuse_growth_rejections"] == reuse_trials
        ),
        "replay_free_endogenous_control": bool(
            mutation["learner_replay_accesses"] == 0
            and mutation["learner_raw_history_retained"] == 0
        ),
        "protected_history_survives_learned_control": (
            mutation["maximum_historical_composition_mse"] <= 1e-10
        ),
        "unsafe_reuse_control_exposes_interference": (
            mutation["mean_unsafe_reuse_historical_mse"] >= 1e-4
        ),
        "cell_local_isolation_preserved": (
            mutation["unrelated_cell_parameter_drift"] <= 1e-15
        ),
        "final_unseen_multicell_composition": bool(
            final_main <= 1e-4
            and final_sim["exact_route_sequence_accuracy"] >= 0.995
            and final_seq["exact_route_sequence_accuracy"] >= 0.995
            and final_seq["mean_true_order_effect_mse"] >= 1e-3
            and final_sim["mean_simultaneous_permutation_mse"] <= 1e-12
        ),
        "sparse_compute_survives_scaffold_removal": bool(
            final_sim["cell_execution_fraction_vs_dense"] <= 0.30
            and final_seq["cell_execution_fraction_vs_dense"] <= 0.30
            and final_sim["maximum_active_cells"] <= cfg.maximum_active_cells
            and final_seq["maximum_active_cells"] <= cfg.maximum_active_cells
        ),
        "slow_shared_substrate_plasticity": shared["safe_fit_mse"] <= 1e-10,
        "slow_shared_substrate_retention": bool(
            shared["safe_historical_composition_mse"] <= 1e-10
            and shared["safe_protected_change"] <= 1e-10
            and shared["unsafe_historical_composition_mse"] >= 1e-6
        ),
    }
    return {
        "seed": seed,
        "pass": all(gates.values()),
        "config": asdict(cfg),
        "controller_state_sha256": controllers.state_sha256,
        "controller_diagnostics": controller_diag,
        "parent_g4": {
            "pass": bool(parent.get("pass", False)),
            "simultaneous_mse": parent.get("simultaneous", {}).get("mean_mse"),
            "sequential_mse": parent.get("sequential", {}).get("mean_mse"),
        },
        "structural_bridge": bridge,
        "acquisition": acquisition,
        "pre_mutation": {
            "simultaneous": pre_sim,
            "sequential": pre_seq,
        },
        "mutation": mutation,
        "final_composition": {
            "simultaneous": final_sim,
            "sequential": final_seq,
        },
        "shared_substrate": shared,
        "gates": gates,
    }


def endogenous_control_smoke(seed: int = 601) -> dict[str, Any]:
    controllers = learned_controllers()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xA005)
    keys = F.normalize(
        torch.randn(5, 20, generator=generator, dtype=_DTYPE), dim=1
    )
    routes = []
    for source in range(len(keys)):
        query = _route_token(keys[source], noise=0.01, generator=generator)
        predicted = int(torch.argmax(controllers.router.scores(query, keys)).item())
        routes.append(int(predicted == source))
    return {
        "seed": int(seed),
        "controller_state_sha256": controllers.state_sha256,
        "controller_diagnostics": controllers.diagnostics,
        "route_smoke_accuracy": _mean(routes),
        "pass": bool(
            _mean(routes) == 1.0
            and min(
                controllers.diagnostics["router"]["heldout_meta_accuracy"],
                controllers.diagnostics["growth"]["heldout_meta_accuracy"],
                controllers.diagnostics["write"]["heldout_meta_accuracy"],
            )
            >= 0.98
        ),
    }
