from params import BioreactorParams


def monod_mu(S: float, p: BioreactorParams) -> float:
    """Specific growth rate mu(S) = mu_max * S / (K_S + S)  [1/h]."""
    mu = p.mu_max * S / (p.K_S + S)
    return mu


def product_rate(S: float, p: BioreactorParams) -> float:
    """Growth-coupled product formation rate q_p(S) = Y_ps * mu(S)  [1/h]."""
    q_p = p.Y_ps * monod_mu(S, p)
    return q_p
