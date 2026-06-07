from dataclasses import replace

import numpy as np
import pytest

from bioreactor.params import BioreactorParams
from bioreactor.simulate import simulate


@pytest.mark.parametrize("feed_rate", [0.025, 0.1, 0.2])
def test_constant_feed_fed_batch_closes_substrate_biomass_mass_balance(feed_rate):
    params = replace(BioreactorParams(), F=feed_rate, V_max=30.0)
    y0 = np.array([0.1, 20.0, 0.0, params.V_0])
    result = simulate(
        "fedbatch_const",
        params,
        y0,
        (0.0, 80.0),
        rtol=1e-9,
        atol=1e-11,
    )

    assert result.raw.success, result.raw.message
    assert not result.near_volume_cap, (
        f"mass-balance oracle assumes unclamped feed, but volume reached "
        f"{result.max_volume:.6g} near V_max={params.V_max:.6g}"
    )

    lhs = result.S * result.V + result.X * result.V / params.Y_xs
    rhs = (
        y0[1] * params.V_0
        + y0[0] * params.V_0 / params.Y_xs
        + params.F * params.S_f * result.t
    )
    rel_err = np.max(np.abs(lhs - rhs) / np.abs(rhs))

    assert rel_err < 1e-6, (
        f"mass-balance relative error {rel_err:.3e} exceeded 1e-6 "
        f"for feed_rate={feed_rate}; "
        f"final volume={result.V[-1]:.6g}, "
        f"final biomass={result.X[-1]:.6g}, "
        f"final substrate={result.S[-1]:.6g}"
    )
