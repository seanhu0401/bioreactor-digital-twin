from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .kinetics import monod_mu
from .params import BioreactorParams


@dataclass(frozen=True)
class ChemostatAnalysis:
    dilution_rate: float
    washout_threshold: float
    washout_expected: bool
    S_star: float
    X_star: float


def analyze_chemostat(params: BioreactorParams) -> ChemostatAnalysis:
    D = params.F / params.V_0
    D_crit = monod_mu(params.S_f, params) - params.k_d
    if D >= D_crit:
        return ChemostatAnalysis(
            dilution_rate=D,
            washout_threshold=D_crit,
            washout_expected=True,
            S_star=params.S_f,
            X_star=0.0,
        )
    S_star = (D + params.k_d) * params.K_S / (params.mu_max - params.k_d - D)
    X_star = params.Y_xs * (params.S_f - S_star)
    return ChemostatAnalysis(
        dilution_rate=D,
        washout_threshold=D_crit,
        washout_expected=False,
        S_star=S_star,
        X_star=X_star,
    )


def chemostat_bifurcation_curve(
    params: BioreactorParams,
    dilution_rates: npt.NDArray[np.float64],
) -> list[ChemostatAnalysis]:
    return [
        analyze_chemostat(replace(params, F=D * params.V_0)) for D in dilution_rates
    ]
