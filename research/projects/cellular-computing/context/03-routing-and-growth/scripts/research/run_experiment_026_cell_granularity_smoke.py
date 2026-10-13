"""CPU smoke test for Experiment 026: developmental cell granularity.

This is not the formal scientific run.  It verifies that granularity can be
changed without changing the initial model function, parameter count, or root
routing architecture, and that one micro-cell division is function-preserving.
"""

from __future__ import annotations

import json

import torch
from minicells.developmental_tissue import (
    TissueConfig,
    convert_model_experts_to_tissues,
    count_module_parameters,
)
from minicells.language_models import TextNCALM, count_parameters
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def main() -> None:
    torch.manual_seed(26031)
    source = TextNCALM(
        vocab_size=31,
        max_context=8,
        dim=8,
        heads=2,
        ffn_dim=12,
        windows=(2, 3, 4),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )
    base = convert_textnca_to_upcycled(
        source,
        config=UpcyclingConfig(num_experts=4, top_k=1),
    )
    inputs = torch.randint(0, 31, (3, 8))
    expected = base(inputs).logits.detach()
    base_parameters = count_parameters(base)

    rows: list[dict[str, object]] = []
    for granularity in (1, 2, 3, 4, 6, 12):
        model = convert_model_experts_to_tissues(
            base,
            config=TissueConfig(cells_per_tissue=granularity),
        )
        logits = model(inputs).logits.detach()
        rows.append(
            {
                "cells_per_tissue": granularity,
                "total_microcells": sum(
                    expert.cell_count
                    for stage in model.stages
                    for expert in stage.program_bank.experts
                ),
                "parameters": count_parameters(model),
                "parameter_delta": count_parameters(model) - base_parameters,
                "max_logits_abs_diff": float((logits - expected).abs().max().item()),
            }
        )

    division_model = convert_model_experts_to_tissues(
        base,
        config=TissueConfig(cells_per_tissue=3, juvenile_plasticity=4.0),
    )
    tissue = division_model.stages[1].program_bank.experts[0]
    before = division_model(inputs).logits.detach().clone()
    before_parameters = count_module_parameters(tissue)
    event = tissue.divide_cell(1)
    after = division_model(inputs).logits.detach()
    division = {
        **event,
        "tissue_parameter_delta": count_module_parameters(tissue) - before_parameters,
        "max_logits_abs_diff": float((after - before).abs().max().item()),
    }

    result = {
        "format": "minicells.experiment-026-granularity-smoke.v0",
        "status": "PASS"
        if all(row["parameter_delta"] == 0 for row in rows)
        and max(float(row["max_logits_abs_diff"]) for row in rows) <= 2e-6
        and division["max_logits_abs_diff"] <= 2e-6
        else "FAIL",
        "granularity": rows,
        "division": division,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
