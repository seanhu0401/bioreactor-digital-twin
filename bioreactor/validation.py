import numpy as np
import numpy.typing as npt

from .params import BioreactorParams
from .simulate import SimulationResult
from .steady_state import ChemostatAnalysis


def fed_batch_const_mass_balance_error(
    result: SimulationResult,
    params: BioreactorParams,
    y0: npt.NDArray[np.float64],
) -> float:
    lhs = result.S * result.V + result.X * result.V / params.Y_xs
    rhs = (
        y0[1] * params.V_0
        + y0[0] * params.V_0 / params.Y_xs
        + params.F * params.S_f * result.t
    )
    return float(np.max(np.abs(lhs - rhs) / np.abs(rhs)))


def fed_batch_const_qss_substrate(
    result: SimulationResult,
    params: BioreactorParams,
) -> npt.NDArray[np.float64]:
    D = params.F / result.V
    if np.any(D >= params.mu_max):
        raise ValueError("QSS substrate requires D(t) < mu_max.")
    return params.K_S * D / (params.mu_max - D)


def chemostat_steady_state_error(
    result: SimulationResult, analysis: ChemostatAnalysis
) -> float:
    """
    Compute the steady-state error for a chemostat simulation result.
    """
    return max(
        abs(result.S[-1] - analysis.S_star),
        abs(result.X[-1] - analysis.X_star),
    )
