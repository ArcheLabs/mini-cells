"""Sidecar for the fixed-length, externally L2-normalized FLA 0.2.2 recurrence."""
import torch
from torch.nn import functional as F


FIELDS = ("prediction_norm", "value_norm", "raw_residual_norm", "relative_residual_norm",
          "delta_norm", "relative_delta_norm", "prediction_cosine", "read_output_norm",
          "beta", "g")


def reference(q, k, v, g, beta, initial_state=None):
    """FP32 recurrence using FLA's [batch, head, key, value] layout."""
    batch, _, heads, key = k.shape
    state = (torch.zeros((batch, heads, key, v.shape[-1]), device=q.device, dtype=torch.float32)
             if initial_state is None else initial_state.float().clone())
    output, prediction, residual = [], [], []
    for t in range(q.shape[1]):
        kt, vt = k[:, t].float(), v[:, t].float()
        state = state * g[:, t].float().exp()[..., None, None]
        predicted = (state * kt[..., None]).sum(-2)
        error = vt - predicted
        strength = beta[:, t].float()
        if beta.ndim == 3:
            strength = strength.unsqueeze(-1)
        state = state + kt[..., None] * (error * strength).unsqueeze(-2)
        output.append((state * (q[:, t].float() * key ** -0.5).unsqueeze(-1)).sum(-2))
        prediction.append(predicted)
        residual.append(error)
    return torch.stack(output, 1).to(q.dtype), state, torch.stack(prediction, 1), torch.stack(residual, 1)


def statistics(output, prediction, residual, value, beta, decay):
    value = value.float()
    norm = value.norm(dim=-1)
    error = residual.norm(dim=-1)
    strength = beta.float()
    if beta.ndim == 3:
        strength = strength.unsqueeze(-1)
    delta = (residual * strength).norm(dim=-1)
    return torch.stack((prediction.norm(dim=-1), norm, error, error / (norm + 1e-6),
                        delta, delta / (norm + 1e-6), F.cosine_similarity(prediction, value, dim=-1, eps=1e-6),
                        output.float().norm(dim=-1), strength.mean(-1), decay.float()), dim=-1)


def parity_report(actual, expected, *, atol, rtol):
    delta = (actual.float() - expected.float()).abs()
    limits = atol + rtol * expected.float().abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return dict(passed=finite and bool((delta <= limits).all()), max_abs=float(delta.max()),
                max_relative=float((delta / expected.float().abs().clamp_min(atol)).max()),
                atol=atol, rtol=rtol)


# FP16 output and FP32 final state have separate error budgets. The original
# and sidecar use one Triton warp and the same reduction tile.
OUTPUT_TOLERANCE = dict(atol=5e-4, rtol=2e-3)
STATE_TOLERANCE = dict(atol=5e-5, rtol=5e-4)


class ResidualTrace:
    def __init__(self, kernel, verify=False, capture=False):
        self.kernel = kernel
        self.verify = verify
        self.routers = []
        self.banks = []
        self.parity = {}
        self.inputs = {} if capture else None
        self.positions = {} if capture else None

    def begin_layer(self, layer, probabilities, selected, weights, packed_tokens):
        self.layer = layer
        self.packed = packed_tokens.cpu()
        topk_mass = probabilities.gather(-1, selected).sum(-1)
        self.routers.append(dict(layer=layer, probabilities=probabilities[0].tolist(),
                                 topk_indices=selected[0].tolist(), topk_mass=topk_mass[0].tolist(),
                                 weights=weights[0].tolist()))

    def for_cell(self, slot, shared=False):
        return BankTrace(self, slot, shared)


class BankTrace:
    def __init__(self, target, slot, shared):
        self.target, self.slot, self.shared = target, slot, shared

    def before(self, bank, q, k, v, g, beta, state):
        if self.target.inputs is not None and self.slot == 0:
            bank = 4 if self.shared else bank
            self.target.inputs[f"{self.target.layer}.{bank}"] = tuple(
                tensor.cpu() for tensor in (q, k, v, g, beta))
        return self.target.kernel(q, k, v, g, beta, state.clone() if state is not None else None)

    def after(self, bank, probe, native_output, native_state, value, beta, decay):
        output, state, prediction, residual = probe
        if self.target.verify:
            for label, actual, expected, tolerance in (
                ("output", output, native_output, OUTPUT_TOLERANCE),
                ("state", state, native_state, STATE_TOLERANCE),
            ):
                report = parity_report(actual, expected, **tolerance)
                current = self.target.parity.setdefault(label, dict(calls=0, **report))
                current["calls"] += 1
                current["passed"] &= report["passed"]
                for key in ("max_abs", "max_relative"):
                    current[key] = max(current[key], report[key])
                if not report["passed"]:
                    raise AssertionError(f"FLA {label} parity at layer {self.target.layer}, "
                                         f"Cell slot {self.slot}, bank {bank}: {report}")
        values = statistics(native_output, prediction, residual, value, beta, decay)[0]
        if self.shared:
            tokens = torch.arange(values.shape[0])
            bank = self.target.packed.shape[0]
        else:
            tokens = self.target.packed[bank]
        mask = tokens >= 0
        if self.target.positions is not None and self.slot == 0:
            self.target.positions[f"{self.target.layer}.{bank}"] = mask.nonzero().flatten()
        self.target.banks.append((self.slot, self.target.layer, bank, tokens[mask],
                                  values.detach().cpu()[mask]))
