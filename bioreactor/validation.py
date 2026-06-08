import numpy as np
import numpy.typing as npt

from .params import BioreactorParams


def validate_initial_volume(
    y0: npt.NDArray[np.float64], params: BioreactorParams
) -> bool:
    return y0[3] == params.V_0
