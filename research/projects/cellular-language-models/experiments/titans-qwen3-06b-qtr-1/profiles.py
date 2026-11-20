"""Architecture-independent frozen QTR-1 optimizer profiles."""
import math

ARMS = {"E0": (.0025, "Q"), "E1": (1., "Q"), "E2": (.0025, "T"), "E3": (1., "T")}
SEEDS = (42, 43, 44)
STEPS = 500
PROFILES = {
    "Q": dict(lr=1e-4, betas=(.9,.999), weight_decay=0., clip=1.),
    "T": dict(lr=5e-4, betas=(.9,.95), weight_decay=.01, clip=10.),
}


def learning_rate(profile, step):
    """One-based rate used for the actual update, not the following update."""
    if not 1 <= step <= STEPS:
        raise ValueError("Invalid optimizer step")
    peak = PROFILES[profile]["lr"]
    if profile == "Q":
        return peak * min(step / 10, 1)
    floor = 1e-5
    if step <= 10:
        return floor + (peak-floor)*(step-1)/9
    return floor + (peak-floor)*(1+math.cos(math.pi*(step-10)/(STEPS-10)))/2


def make_optimizer(profile, named, backbone, upstream):
    import subprocess
    import sys
    import torch
    from pathlib import Path
    params = list(named.values())
    if not params or len({id(p) for p in params}) != len(params) or not all(p.requires_grad for p in params):
        raise ValueError("Invalid trainable parameter allowlist")
    if {id(p) for p in params} & {id(p) for p in backbone}:
        raise ValueError("Backbone in optimizer")
    cls = torch.optim.AdamW
    if profile == "T":
        path = Path(upstream).resolve()
        sha = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
        if sha != "c60b4efaa3a3c6df4deebd57cbccf75947b87cad":
            raise ValueError("Wrong FP32AdamW source")
        sys.path.insert(0,str(path))
        from fp32_adamw import FP32AdamW
        import inspect
        if Path(inspect.getfile(FP32AdamW)).resolve() != path / "fp32_adamw.py":
            raise ValueError("FP32AdamW imported from another checkout")
        cls = FP32AdamW
    cfg = PROFILES[profile]
    return cls(params,lr=learning_rate(profile,1),betas=cfg["betas"],weight_decay=cfg["weight_decay"])
