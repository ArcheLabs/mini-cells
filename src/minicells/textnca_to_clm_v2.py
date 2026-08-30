from __future__ import annotations

from .language_models import TextNCALM
from .overcomplete_cellular_textnca import CLMv2Config, OvercompleteCellularTextNCA


def convert_textnca_to_clm_v2(
    model: TextNCALM,
    *,
    config: CLMv2Config | None = None,
) -> OvercompleteCellularTextNCA:
    converted = OvercompleteCellularTextNCA(model, config or CLMv2Config())
    converted.freeze_scaffold()
    converted.train(model.training)
    return converted
