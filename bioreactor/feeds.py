import numpy as np
from params import BioreactorParams

from .model_types import FeedFunc


def make_feed(mode: str, p: BioreactorParams) -> tuple[FeedFunc, bool]:
    """
    Return (F_of_t, has_outflow) for the requested operating mode.

    F_of_t(t, V) -> volumetric feed rate [L/h]; smoothly squashed toward 0 as
    V approaches p.V_max (keep the RHS continuous, per the design note).
    has_outflow is True only for chemostat (constant-volume) operation.
    """
    clamp_steepness = 20.0  # 1/L; numerical smoothness of the V_max feed cap.

    # 1. Define mathematical closures upfront
    def _batch(t: float, V: float) -> float:
        return 0.0

    def _const(t: float, V: float) -> float:
        return p.F

    def _exp(t: float, V: float) -> float:
        return p.F * np.exp(p.alpha * t)

    # 2. Select the strategy and state
    base_feed_func: FeedFunc
    outflow: bool
    match mode:
        case "batch":
            base_feed_func = _batch
            outflow = False
        case "fedbatch_const":
            base_feed_func = _const
            outflow = False
        case "fedbatch_exp":
            base_feed_func = _exp
            outflow = False
        case "chemostat":
            base_feed_func = _const
            outflow = True
        case _:
            raise NotImplementedError(f"Unknown operating mode: {mode}")

    # 3. Use the chosen strategy inside the final wrapper
    def F_of_t(t: float, V: float) -> float:
        clamp: float = 0.5 * (1 - np.tanh(clamp_steepness * (V - p.V_max)))
        return base_feed_func(t, V) * clamp

    return F_of_t, outflow
