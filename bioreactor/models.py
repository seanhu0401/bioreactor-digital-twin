import numpy as np

from ._model_types import FeedFunc, StateVector
from .kinetics import monod_mu, product_rate
from .params import BioreactorParams


def fed_batch_rhs(
    t: float, y: StateVector, p: BioreactorParams, feed: FeedFunc, outflow: bool
) -> StateVector:
    """
    4-state fed-batch RHS: y = [X, S, P, V] -> dy/dt.

    feed:    F_of_t(t, V) callable from make_feed
    outflow: True for chemostat (dilution removes product/biomass at F/V)
    """
    X, S, P, V = y

    F = feed(t, V)
    D = F / V
    mu = monod_mu(S, p)
    qp = product_rate(S, p)

    dX = mu * X - p.k_d * X - D * X
    dS = -mu * X / p.Y_xs + D * (p.S_f - S)
    dP = qp * X - D * P
    dV = 0.0 if outflow else F

    return np.array([dX, dS, dP, dV], dtype=float)
