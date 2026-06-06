from dataclasses import dataclass


@dataclass(frozen=True)
class BioreactorParams:
    mu_max: float = 0.46  # maximum specific growth rate [h^-1]
    K_S: float = 33.5  # Monod half-saturation constant [g/L]
    Y_xs: float = 0.08  # biomass yield from substrate [g X / g S]
    Y_ps: float = 0.05  # product yield from substrate [g P / g S] (placeholder)
    k_d: float = 0.0  # first-order death/maintenance [h^-1] (off by default)
    S_f: float = 50.0  # feed substrate concentration [g/L]
    V_0: float = 1.0  # initial volume [L]
    V_max: float = 5.0  # maximum reactor capacity [L]
    F: float = 0.1  # baseline constant feed rate [L/h]
    alpha: float = 0.1  # exponential-feed rate constant [h^-1] (fedbatch_exp)
