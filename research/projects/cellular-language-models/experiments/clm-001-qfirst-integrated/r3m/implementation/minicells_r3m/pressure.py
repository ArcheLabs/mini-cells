"""Task-blind prequential detector: current observations vs strictly past statistics."""

import math
import numpy as np


class Detector:
    def __init__(self, config, state=None):
        self.config = dict(config)
        self.state = state or dict(
            count=0,
            mean=0.0,
            variance=0.0,
            gradient=None,
            streak=0,
            last_proposal=-1000,
            proposals=0,
        )

    def observe(self, loss, gradient, step):
        s, c = self.state, self.config
        g = np.asarray(gradient, dtype=np.float64)
        if not math.isfinite(loss) or not np.isfinite(g).all():
            raise ValueError("NONFINITE_PRESSURE_INPUT")
        prior = s["gradient"]
        conflict = (
            0.0
            if prior is None
            else 1 - float(np.dot(g, prior) / max(1e-12, np.linalg.norm(g) * np.linalg.norm(prior)))
        )
        surprise = (
            0.0 if s["count"] < 2 else (loss - s["mean"]) / math.sqrt(max(1e-12, s["variance"]))
        )
        score = None
        if c.get("threshold") is not None:
            score = max(
                (surprise - c["loss_mean"]) / c["loss_std"],
                (conflict - c["gradient_mean"]) / c["gradient_std"],
            )
        # Observe first, update history afterwards. No environment/task argument exists.
        delta = loss - s["mean"]
        s["variance"] = 0.95 * (s["variance"] + 0.05 * delta * delta)
        s["mean"] += 0.05 * delta
        s["gradient"] = (
            g.tolist() if prior is None else (0.95 * np.asarray(prior) + 0.05 * g).tolist()
        )
        s["count"] += 1
        trigger = False
        if step % c["interval"] == 0:
            s["streak"] = s["streak"] + 1 if score is not None and score > c["threshold"] else 0
            trigger = (
                s["streak"] >= c["consecutive"]
                and step - s["last_proposal"] >= c["cooldown"]
                and s["proposals"] < c["maximum_proposals"]
            )
            if trigger:
                s["last_proposal"], s["proposals"], s["streak"] = step, s["proposals"] + 1, 0
        return dict(
            loss_surprise=surprise,
            gradient_conflict=conflict,
            standardized_score=score,
            triggered=trigger,
        )
