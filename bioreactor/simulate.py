from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .feeds import make_feed
from .models import fed_batch_rhs
from .params import BioreactorParams
from .state import validate_initial_volume
from .steady_state import ChemostatAnalysis, analyze_chemostat


class SolverResult(Protocol):
    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    success: bool
    message: str


@dataclass(frozen=True)
class SimulationResult:
    t: npt.NDArray[np.float64]
    X: npt.NDArray[np.float64]
    S: npt.NDArray[np.float64]
    P: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]
    raw: SolverResult
    mode: str
    max_volume: float
    near_volume_cap: bool
    chemostat: ChemostatAnalysis | None = None


def simulate(
    mode: str,
    params: BioreactorParams,
    y0: npt.NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    volume_cap_margin: float = 0.5,
    rtol: float = 1e-3,
    atol: float | npt.NDArray[np.float64] = 1e-6,
    dense_output: bool = False,
    max_step: float = np.inf,
) -> SimulationResult:
    """
    Simulate the bioreactor dynamics for a given mode and parameter set.

    Parameters
    ----------
    mode : str
        The feed mode to use.
    params : BioreactorParams
        The bioreactor parameters.
    y0 : npt.NDArray[np.float64]
        The initial conditions.
    t_span : tuple[float, float]
        The time span to simulate over.
    volume_cap_margin : float, optional
        The margin to use when checking for volume cap, by default 0.5.
    rtol : float, optional
        The relative tolerance for the solver, by default 1e-3.
    atol : float | npt.NDArray[np.float64], optional
        The absolute tolerance for the solver, by default 1e-6.
    dense_output : bool, optional
        Whether to return dense output, by default False.
    max_step : float, optional
        The maximum step size for the solver, by default np.inf.

    Returns
    -------
    SimulationResult
        The simulation result.
    """

    validate_initial_volume(y0, params)

    feed, outflow = make_feed(mode, params)
    solution = solve_ivp(
        fed_batch_rhs,
        t_span,
        y0,
        args=(params, feed, outflow),
        method="LSODA",
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        max_step=max_step,
    )
    X, S, P, V = solution.y
    max_volume = float(np.max(V))

    chemostat = None
    if mode == "chemostat":
        chemostat = analyze_chemostat(params)

    return SimulationResult(
        t=solution.t,
        X=X,
        S=S,
        P=P,
        V=V,
        raw=solution,
        mode=mode,
        max_volume=max_volume,
        near_volume_cap=max_volume >= params.V_max - volume_cap_margin,
        chemostat=chemostat,
    )
