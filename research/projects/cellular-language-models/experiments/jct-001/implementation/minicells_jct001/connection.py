"""Source-bound production Q optimizer with explicit, unapproved synthetic task adapters."""

import types
import torch
from minicells_r3e import policy_proposal as production
from minicells_r3k.reuse import bind


def references(model, c, data):
    # No calibration labels, report targets, correct actions or Oracle routes enter admission.
    return [data["reference_features"]]


@torch.no_grad()
def trust(old, candidate, refs):
    kl = []
    changed = []
    paths = []
    for features in refs:
        before = old(features, keep_trace=True)
        after = candidate(features)
        changed.append((before.actions != after.actions).float())
        paths.append((before.actions != after.actions).any(-1).float())
        for row in before.trace:
            p = old.scores(row["state"], row["stage"], row["previous"])[0].double().softmax(-1)
            q = (
                candidate.scores(row["state"], row["stage"], row["previous"])[0]
                .double()
                .softmax(-1)
            )
            kl.append((p * (p.log() - q.log())).sum(-1))
    return dict(
        kl=float(torch.cat(kl).mean()),
        action_change=float(torch.cat(changed).mean()),
        path_change=float(torch.cat(paths).mean()),
    )


def within(metrics, c):
    q = c["engineering_trust"]
    return (
        metrics["kl"] <= q["kl_max"]
        and metrics["action_change"] <= q["action_change_max"]
        and metrics["path_change"] <= q["path_change_max"]
    )


class TorchAdapter:
    def __init__(self, journal):
        def optimizer(parameters, **kwargs):
            parameters = list(parameters)
            opt = torch.optim.AdamW(parameters, **kwargs)
            step = opt.step
            opt.step = lambda *a, **kw: journal.optimizer_call(
                "connection", lambda: step(*a, **kw), parameters
            )
            return opt

        self.optim = types.SimpleNamespace(AdamW=optimizer)

    def __getattr__(self, name):
        return getattr(torch, name)


def propose(trainer, data):
    def current_training_batch(dataset, engine, rng, device):
        if (
            dataset.get("role") not in ("train", "UNIT_ONLY")
            or "current_training_batch" not in dataset
        ):
            raise RuntimeError("JCT_COUNTERFACTUALS_REQUIRE_THE_CURRENT_TRAIN_BATCH")
        return dataset["current_training_batch"]

    collector = torch.no_grad()(
        bind(production.collect.__wrapped__, train_batches=current_training_batch)
    )
    optimizer = bind(
        production.optimize,
        collect=collector,
        reference=references,
        trust=trust,
        within=within,
        torch=TorchAdapter(trainer.journal),
    )
    with trainer.journal.phase("connection_proposal_counterfactual_trust"):
        candidate, info = optimizer(trainer, data)
    if candidate is not None:
        trainer.model.connection.load_state_dict(candidate.connection.state_dict(), strict=True)
    trainer.connection_optimizer_state = info["proposal_optimizer_final_state"]
    info["algorithm"] = "SOURCE_BOUND_R3E_ALL_ACTION_TERMINAL_Q_EXPECTED_CENTERED_COST"
    info["scientific_mechanism_review"] = "PENDING_TASK_ADAPTER_AND_TRUST_CONFIG_REVIEW"
    info["data_semantics"] = "CURRENT_MAIN_TRAIN_BATCH_REUSED; NO_NEW_EXAMPLES_FROM_NEXT_BATCH"
    trainer.journal.append(
        "AUXILIARY_LABEL_WORK",
        dict(
            target_evaluations=info["counterfactual_states"] * info["actions_per_state"],
            distinct_extra_examples=0,
        ),
    )
    return info
