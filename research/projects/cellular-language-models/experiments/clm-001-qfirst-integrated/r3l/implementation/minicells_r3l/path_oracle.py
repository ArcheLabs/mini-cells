"""Target-aware finite beam diagnostic using the actual model transition operation."""

import copy
import time
import torch
import torch.nn.functional as F
from minicells_r3h.evaluation import windows
from minicells_r3e.common import fingerprint


@torch.no_grad()
def run(model, runtime, data, count=16, width=8, seed=2026102305):
    model = copy.deepcopy(model)
    model.eval()
    before = fingerprint(model.state_dict())
    before_work = dict(model.work)
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records = []
    for story, x, y in windows(data, runtime, "development", next(model.parameters()).device):
        initial, _ = model.initial(x)
        for position in range(x.numel()):
            if len(records) >= count:
                break
            target = y.reshape(-1)[position : position + 1]
            h0 = initial[position : position + 1]

            def loss(h):
                return float(F.cross_entropy(model.head(h).float(), target))

            baseline = h0.clone()
            previous = torch.full((1,), 6, device=h0.device, dtype=torch.long)
            baseline_path = []
            random = h0.clone()
            rp = previous.clone()
            random_path = []
            beams = [(loss(h0), h0, previous, [])]
            for stage in range(model.steps):
                action, _ = model.select(baseline, stage, previous, epsilon=0)
                baseline, previous = model.transition(baseline, stage, previous, action)
                baseline_path.append(int(action))
                ra = torch.randint(6 if model.graph else 15, (1,), generator=generator).to(
                    h0.device
                )
                random, rp = model.transition(random, stage, rp, ra)
                random_path.append(int(ra))
                expanded = []
                for _, state, pred, path in beams:
                    for action in range(6 if model.graph else 15):
                        nxt, npred = model.transition(
                            state, stage, pred, torch.tensor([action], device=h0.device)
                        )
                        expanded.append((loss(nxt), nxt, npred, path + [action]))
                beams = sorted(expanded, key=lambda item: (item[0], item[3]))[:width]
            records.append(
                dict(
                    story=story,
                    position=position,
                    baseline_nll=loss(baseline),
                    random_nll=loss(random),
                    beam_nll=beams[0][0],
                    baseline_path=baseline_path,
                    random_path=random_path,
                    beam_path=beams[0][3],
                    cell_calls_per_complete_path=8,
                    beam_width=width,
                )
            )
        if len(records) >= count:
            break
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("ORACLE_MUTATED_MODEL")
    return dict(
        scope="TARGET_AWARE_NON_DEPLOYABLE_DIAGNOSTIC",
        global_upper_bound=False,
        deployed_router=False,
        records=records,
        seconds=time.perf_counter() - started,
        operator_rows={k: v - before_work.get(k, 0) for k, v in model.work.items()},
    )
