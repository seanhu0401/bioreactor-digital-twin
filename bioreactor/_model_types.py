from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type FeedFunc = Callable[[float, float], float]
type StateVector = npt.NDArray[np.float64]
