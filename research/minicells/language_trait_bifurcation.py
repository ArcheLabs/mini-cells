from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .language_conflict_differentiation import DOMAINS, IDENTITY_NORMALIZED_MARGIN_MIN, ROUTING_PURITY_MIN, summarize_identity

ARMS = ("unified", "stratified-capacity-fork", "geometry-bifurcation-fork")
BIFURCATION_GAIN_MIN = 0.20
BIFURCATION_SPLIT_BALANCE_MIN = 0.25
BIFURCATION_AXIS_STABILITY_MIN = 0.50
BIFURCATION_WINDOWS_MIN = 2
GEOMETRY_MARGIN_ADVANTAGE_MIN = 0.05
POSITIVE_REPLICATES_MIN = 2
KMEANS_STEPS = 32


@dataclass(frozen=True)
class BifurcationGeometry:
    one_mode_centroid: torch.Tensor
    centroids: torch.Tensor
    axis: torch.Tensor
    residual_k1: float
    residual_k2: float
    bifurcation_gain: float
    split_balance: float
    centroid_separation: float


@dataclass(frozen=True)
class PersistentBifurcationSummary:
    windows_passed: int
    axis_stability: float
    combined_gain: float
    combined_split_balance: float
    persistent: bool
