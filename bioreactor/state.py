import numpy as np
import numpy.typing as npt

from .params import BioreactorParams


def initial_state(
    X0: float,
    S0: float,
    P0: float = 0.0,
    *,
    params: BioreactorParams,
) -> npt.NDArray[np.float64]:
    return np.array([X0, S0, P0, params.V_0], dtype=float)


def validate_initial_volume(
    y0: npt.NDArray[np.float64],
    params: BioreactorParams,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    if not np.isclose(float(y0[3]), params.V_0, rtol=rtol, atol=atol):
        raise ValueError(
            f"Initial volume mismatch: y0[3]={float(y0[3]):.6g} "
            f"but params.V_0={params.V_0:.6g}. "
            "Use initial_state(..., params=params) or update BioreactorParams(V_0=...)."
        )
