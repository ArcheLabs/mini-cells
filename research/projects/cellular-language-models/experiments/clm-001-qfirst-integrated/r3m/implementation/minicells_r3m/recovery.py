"""Cross-container next-update comparison using a pinned immutable checkpoint and batch."""

import copy
import socket
from pathlib import Path
import torch
from minicells_r3e.common import canonical, fingerprint
from .training import restore, advance, semantic, initial_aux


def session_identity():
    return canonical(
        dict(
            host=socket.gethostname(),
            boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
    )


def reference(tr, batches, c, identity, sampled_rng_state=None):
    state = copy.deepcopy(tr.state())
    aux = initial_aux(tr, c, identity)
    copied = restore(tr.c, state, tr.model.seed, next(tr.model.parameters()).device)
    if sampled_rng_state is not None:
        copied.data_rng.bit_generator.state = copy.deepcopy(sampled_rng_state)
    row = advance(copied, batches, c, aux["schedule"])
    return dict(
        sampled_rng_state=sampled_rng_state,
        writer_session=session_identity(),
        identity=identity,
        anchor=state,
        batches=[(x.cpu(), y.cpu()) for x, y in batches],
        schedule=aux["schedule"],
        expected=copy.deepcopy(semantic(copied.state())),
        ce=row["ce"],
        scope="ONE_UPDATE_ENGINEERING_NOT_SCIENTIFIC_CONTINUATION",
    )


def verify(payload, runtime, c, device, reader_session=None, batch_factory=None):
    reader_session = reader_session or session_identity()
    if reader_session == payload["writer_session"]:
        raise RuntimeError("CROSS_SESSION_RESTORE_REQUIRES_NEW_KAGGLE_SESSION")
    tr = restore(runtime, payload["anchor"], payload["identity"]["seed"], device)
    batches = [(x.to(device), y.to(device)) for x, y in payload["batches"]]
    if payload.get("sampled_rng_state") is not None:
        if batch_factory is None:
            raise RuntimeError("CROSS_SESSION_RESTORE_REQUIRES_REAL_SAMPLER")
        drawn = batch_factory(tr)
        if fingerprint([(x.cpu(), y.cpu()) for x, y in drawn]) != fingerprint(payload["batches"]):
            raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_BATCH_STREAM")
        if fingerprint(tr.data_rng.bit_generator.state) != fingerprint(
            payload["sampled_rng_state"]
        ):
            raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_DATA_RNG")
        batches = drawn
    row = advance(tr, batches, c, payload["schedule"])
    actual = semantic(tr.state())
    expected = payload["expected"]
    for k in actual:
        if k in ("model", "cell_optimizer"):

            def compare(a, b):
                if isinstance(a, torch.Tensor):
                    if not torch.allclose(a.cpu(), b.cpu(), rtol=1e-5, atol=1e-7):
                        raise RuntimeError("CROSS_SESSION_RESTORE_INVALID:" + k)
                elif isinstance(a, dict):
                    if set(a) != set(b):
                        raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_KEYS")
                    for key in a:
                        compare(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    if len(a) != len(b):
                        raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_LENGTH")
                    for x, y in zip(a, b):
                        compare(x, y)
                elif a != b:
                    raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_VALUE")

            compare(actual[k], expected[k])
        elif fingerprint(actual[k]) != fingerprint(expected[k]):
            raise RuntimeError("CROSS_SESSION_RESTORE_INVALID:" + k)
    if abs(row["ce"] - payload["ce"]) > 1e-5:
        raise RuntimeError("CROSS_SESSION_RESTORE_INVALID_NEXT_CE")
    return dict(
        status="CROSS_SESSION_RESTORE_PASS",
        writer_session=payload["writer_session"],
        reader_session=reader_session,
        model_optimizer="RTOL_1E-5_ATOL_1E-7",
        stream_cursor_identity="EXACT",
        production_updates_added=0,
    )
