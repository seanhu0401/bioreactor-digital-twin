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
