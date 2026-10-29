"""CLM003-only affine interpolation audit; production Trust metrics stay Float32."""

import copy
from types import SimpleNamespace

import torch
from minicells_fast001.engine import Engine as OriginalEngine
from minicells_fast_r1 import scan, stageb_engine
from minicells_r3e.graph import GraphConnection
from minicells_r3f.evaluation import states, trust
from minicells_r3k.reuse import bind


@torch.no_grad()
def audited_scan(engine, package, refs, journal, directory, config, identity):
    def checked_trust(
        old, new, inputs, *, fixed=None, expected_delta=None, scale=None, rtol=1e-5, atol=1e-7
    ):
        if (
            type(old.connection) is not GraphConnection
            or type(new.connection) is not GraphConnection
        ):
            raise RuntimeError("CLM003_AFFINE_AUDIT_UNSUPPORTED_CONNECTION")
        if scale is None or expected_delta is None:
            raise RuntimeError("CLM003_AFFINE_AUDIT_MISSING_ENDPOINTS")
        original = package["original_connection"]
        full = package["candidate_connection"]
        # Verify the actual deployed values, not just closeness of their outputs.
        for name, actual in new.connection.named_parameters():
            expected = original[name].to(actual) + scale * (
                full[name].to(actual) - original[name].to(actual)
            )
            if not torch.equal(actual, expected):
                raise RuntimeError("CLM003_SCALED_PARAMETER_IDENTITY_MISMATCH")
        first = copy.deepcopy(old.connection).double()
        last = copy.deepcopy(old.connection).double()
        last.load_state_dict({k: v.double() for k, v in full.items()}, strict=True)
        ideal = copy.deepcopy(first)
        for name, p in ideal.named_parameters():
            p.copy_(
                dict(first.named_parameters())[name]
                + scale
                * (dict(last.named_parameters())[name] - dict(first.named_parameters())[name])
            )
        records = states(old, inputs) if fixed is None else fixed
        affine_error = 0.0
        legacy_error = 0.0
        legacy_pass = True
        for i, row in enumerate(records):
            h, t, previous = row["state"], row["stage"], row["previous"]
            zero = torch.zeros(h.shape[0], 6, device=h.device, dtype=torch.float64)
            d0 = first(h.double(), t, zero, previous)[1]
            d1 = last(h.double(), t, zero, previous)[1]
            di = ideal(h.double(), t, zero, previous)[1]
            expected64 = d0 + scale * (d1 - d0)
            affine_error = max(affine_error, float((di - expected64).abs().max()))
            if not torch.allclose(di, expected64, rtol=rtol, atol=atol):
                raise RuntimeError("CLM003_FP64_AFFINE_INTERPOLATION_MISMATCH")
            do = old.scores(h, t, previous)[1]
            dn = new.scores(h, t, previous)[1]
            legacy = do + scale * (expected_delta[i] - do)
            legacy_error = max(legacy_error, float((dn - legacy).abs().max()))
            legacy_pass = legacy_pass and torch.allclose(dn, legacy, rtol=rtol, atol=atol)
        # No numeric tolerance or routing metric is changed here. The old endpoint
        # Float32 affine check is replaced only for this known affine Connection.
        metrics = trust(old, new, inputs, fixed=records)
        metrics.update(
            interpolation_max_abs_error=affine_error,
            interpolation_check_scope="FP64_AFFINE_LAW_AND_EXACT_DEPLOYED_FP32_PARAMETERS",
            legacy_fp32_interpolation_max_abs_error=legacy_error,
            legacy_fp32_interpolation_pass=bool(legacy_pass),
            production_metric_dtype="UNCHANGED_FLOAT32_ROUTING_RAW_DOUBLE_SOFTMAX_KL",
        )
        return metrics

    return bind(getattr(scan.run, "__wrapped__", scan.run), trust=checked_trust)(
        engine, package, refs, journal, directory, config, identity
    )


class Engine(OriginalEngine):
    def update(self, stream, checkpoint, references, failure_hook=None):
        proposal = bind(stageb_engine.propose, scan=SimpleNamespace(run=audited_scan))
        return bind(
            OriginalEngine.update,
            policy_proposal=SimpleNamespace(collect=stageb_engine.collect, optimize=proposal),
        )(self, stream, checkpoint, references, failure_hook)
