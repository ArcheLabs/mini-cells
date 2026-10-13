"""CPU-trained order-sensitive Gate 1; isolated from all TinyStories evidence."""
from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from .routing import StateAwareRouter, hard_with_top2_backup
from .topology import make_initial_graph, refresh_edges

GATE_SEED = 50713041


def transform(state, ids):
    a = torch.stack((state[:, 0] + state[:, 1], state[:, 1]), -1)
    b = torch.stack((state[:, 0], state[:, 0] + state[:, 1]), -1)
    c = torch.stack((-state[:, 1], state[:, 0]), -1)
    return torch.stack((a, b, c), 1)[torch.arange(len(state)), ids]


def trained_composition_gate(steps: int = 400) -> dict[str, Any]:
    """Train identical routers on AB/BA tasks with fixed or refreshed adjacency.

    Both arms execute two frozen noncommuting Cells. The first is specified by
    the task instruction; the learned second router is source/state aware.
    Refresh scores use training-only soft-label NLL. Validation inputs are never
    used to select edges or train routers. This is a mechanism gate, not LM proof.
    """
    torch.set_num_threads(1)
    rng = np.random.default_rng(GATE_SEED)
    train_x = torch.tensor(rng.normal(size=(512, 2)), dtype=torch.float32)
    valid_x = torch.tensor(rng.normal(size=(256, 2)), dtype=torch.float32)
    train_source = torch.arange(512) % 2
    valid_source = torch.arange(256) % 2
    def target(x, source):
        return transform(transform(x, source), 1 - source).softmax(-1)
    train_target, valid_target = target(train_x, train_source), target(valid_x, valid_source)
    initial = make_initial_graph(3, 2, GATE_SEED)
    outcomes = {}
    for name in ('fixed-random', 'learned-adjacency'):
        torch.manual_seed(GATE_SEED)
        router = StateAwareRouter(4, 3, 8)
        optimizer = torch.optim.Adam(router.parameters(), lr=0.02)
        graph = initial.clone()
        changes = []
        history = []
        def logits(x, source, train):
            state = transform(x, source)
            inputs = torch.cat((state, F.one_hot(source, 2).float()), -1)
            route = router(inputs, source, graph)
            values, slots = route.logits.topk(2, -1)
            ids = route.candidate_ids.gather(1, slots)
            hard = transform(state, ids[:, 0])
            if train:
                backup = torch.stack((hard, transform(state, ids[:, 1])), 1)
                return hard_with_top2_backup(hard, backup, values)
            return hard
        for step in range(1, steps + 1):
            if step % 50 == 0:
                losses = torch.full(graph.shape, float('nan'), dtype=torch.float64)
                with torch.no_grad():
                    for source in [0, 1]:
                        mask = train_source == source
                        state = transform(train_x[mask], train_source[mask])
                        for cell in range(3):
                            value = transform(state, torch.full((len(state),), cell))
                            losses[source, cell] = -(train_target[mask] * value.log_softmax(-1)).sum(-1).mean()
                updated, event = refresh_edges(graph, losses, step=step, max_replacements=2)
                if name == 'learned-adjacency':
                    graph = updated
                    changes.extend(event)
            optimizer.zero_grad()
            prediction = logits(train_x, train_source, True)
            loss = -(train_target * prediction.log_softmax(-1)).sum(-1).mean()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                history.append({'step':step,'training_soft_label_nll':float(loss.detach())})
        with torch.no_grad():
            predicted = logits(valid_x, valid_source, False)
            nll = -(valid_target * predicted.log_softmax(-1)).sum(-1).mean()
            entropy = -(valid_target * valid_target.clamp_min(1e-12).log()).sum(-1).mean()
            kl = float(nll - entropy)
        outcomes[name] = {'validation_soft_label_nll':float(nll), 'validation_kl':kl,
                         'edge_changes':changes, 'final_adjacency':graph.int().tolist(),
                         'training_curve':history, 'steps':steps, 'cell_executions_per_example':2}
    fixed = outcomes['fixed-random']['validation_kl']
    learned = outcomes['learned-adjacency']['validation_kl']
    passed = bool(fixed > 1e-3 and learned <= .75 * fixed and outcomes['learned-adjacency']['edge_changes'])
    return {'status':'TRAINED_GATE_1_PASSED' if passed else 'TRAINED_GATE_1_FAILED',
            'scientific_decision':False, 'seed':GATE_SEED, 'predeclared_steps':steps,
            'predeclared_min_kl_relative_improvement':.25,
            'train_input_sha256':hashlib.sha256(train_x.numpy().tobytes()).hexdigest(),
            'validation_input_sha256':hashlib.sha256(valid_x.numpy().tobytes()).hexdigest(),
            'train_examples':512,'validation_examples':256,
            'initial_adjacency':initial.int().tolist(),'arms':outcomes,
            'scope':'trained router mechanism with two frozen affine Cells and an instruction-fixed first Cell; does not establish the full LM graph claim'}
