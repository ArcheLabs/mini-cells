"""Six read-only configurations; never replace frozen weights to hide a mismatch."""

import copy
import torch
from .common import fingerprint
from .evaluation import measure


def effects(values):
    oo, on, no, nn = (values[k] for k in ("OO", "ON", "NO", "NN"))
    return {
        "route_effect_old_cells": on - oo,
        "route_effect_new_cells": nn - no,
        "cell_effect_old_connection": no - oo,
        "cell_effect_new_connection": nn - on,
        "interaction": nn - no - on + oo,
        "immediate_route_effect": values["0N"] - values["0O"],
        "old_branch_total_change": oo - values["0O"],
        "new_branch_total_change": nn - values["0N"],
    }


def evaluate(c, arm, models, data):
    runtime = c["parent_runtime"]
    zero, old, new = (models[k] for k in ("official", "old", "new"))
    configurations = {
        "OO": (old, old),
        "ON": (old, new),
        "NO": (new, old),
        "NN": (new, new),
        "0O": (zero, old),
        "0N": (zero, new),
    }
    runs = {}
    cache = {}
    for name, (cell_source, connection_source) in configurations.items():
        m = copy.deepcopy(cell_source)
        m.connection.load_state_dict(connection_source.connection.state_dict(), strict=True)
        digest = fingerprint(m.state_dict())
        if digest not in cache:
            cache[digest] = measure(m, runtime, data, c["quality_repeats"])
        runs[name] = {
            **cache[digest],
            "identical_model_evaluation_reused": any(
                r["model_hash"] == digest for r in runs.values()
            ),
        }
    values = {k: r["nll"] for k, r in runs.items()}
    historical = c["historical_nll"][arm]
    errors = {k: abs(values[k] - historical[k]) for k in ("OO", "NN")}
    reproduced = all(v <= c["historical_nll_atol"] for v in errors.values())
    dtype_floor = torch.finfo(torch.float32).eps * max(abs(v) for v in values.values())
    resolution = max(
        c["numerical_floor"], dtype_floor, max(r["max_repeat_error"] for r in runs.values())
    )
    result = {
        "arm": arm,
        "seed": c["seed"],
        **{"L_" + k: v for k, v in values.items()},
        "evaluations": runs,
        "numerical_resolution": resolution,
        "dtype_resolution_floor": dtype_floor,
        "interaction_error_bound": 4 * resolution,
        "historical_nll_absolute_error": errors,
        "historical_nll_reproduced": reproduced,
        "status": "HISTORICAL_CALIBRATION_REPRODUCED"
        if reproduced
        else "HISTORICAL_CALIBRATION_MISMATCH",
    }
    if reproduced:
        result.update(effects(values))
    return result


def attribution(cross, parameters, trajectory):
    if not cross["historical_nll_reproduced"]:
        return {
            "attribution_status": "INVALID_OR_INCOMPLETE",
            "reason": "HISTORICAL_CALIBRATION_MISMATCH",
        }
    tolerance = cross["numerical_resolution"]
    error = 2 * tolerance
    route = cross["route_effect_old_cells"]
    cell = cross["cell_effect_new_connection"]
    interaction = cross["interaction"]
    identical = parameters["all_cells_bitwise_equal"]
    if identical and (
        abs(cross["cell_effect_old_connection"]) > error
        or abs(cell) > error
        or abs(interaction) > 4 * tolerance
    ):
        raise RuntimeError("IDENTICAL_CELLS_CROSSOVER_INCONSISTENT")
    nondetect = identical or (
        abs(cell) <= error and abs(cross["cell_effect_old_connection"]) <= error
    )
    favorable = interaction < -4 * tolerance
    route_only = route < -error and nondetect
    benefit = cell < -error
    tags = []
    if route_only:
        tags.append("ROUTE_DOMINANT")
    if benefit:
        tags.append("CELL_ADAPTATION_DETECTED")
    if favorable:
        tags.append("FAVORABLE_INTERACTION_DETECTED")
    if nondetect:
        tags.append("NO_DETECTABLE_CELL_ADAPTATION")
    if not tags:
        tags = ["MIXED_OR_INCONCLUSIVE"]
    return {
        "route_only_benefit": route_only,
        "cell_adaptation_benefit": benefit,
        "interaction_effect": interaction,
        "favorable_interaction": favorable,
        "training_route_changed": trajectory["any_training_path_changed"],
        "cell_parameters_changed": not identical,
        "quality_change_detected": abs(cross["L_NN"] - cross["L_OO"]) > error,
        "numerical_resolution": tolerance,
        "cell_effect_status": "BELOW_NUMERICAL_RESOLUTION" if abs(cell) <= error else "RESOLVED",
        "attribution_status": tags,
        "raw_effects": {
            k: cross[k]
            for k in effects({k: cross["L_" + k] for k in ("0O", "0N", "OO", "ON", "NO", "NN")})
        },
        "limits": "One seed; Calibration selected the parent organization. No independent confirmation or specialization claim.",
    }
