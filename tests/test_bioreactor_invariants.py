from dataclasses import replace

import numpy as np
import pytest

from bioreactor.params import BioreactorParams
from bioreactor.simulate import simulate
from bioreactor.state import initial_state, validate_initial_volume
from bioreactor.steady_state import analyze_chemostat, chemostat_bifurcation_curve
from bioreactor.validation import (
    chemostat_steady_state_error,
    fed_batch_const_mass_balance_error,
    fed_batch_const_qss_substrate,
)


@pytest.mark.parametrize("feed_rate", [0.025, 0.1, 0.2])
def test_constant_feed_fed_batch_closes_substrate_biomass_mass_balance(feed_rate):
    params = BioreactorParams(F=feed_rate, V_max=30.0)
    y0 = initial_state(0.1, 20.0, 0.0, params=params)
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

    err = fed_batch_const_mass_balance_error(result, params, y0)

    assert err < 1e-6, (
        f"mass-balance relative error {err:.3e} exceeded 1e-6 for feed_rate={feed_rate}; "
        f"final volume={result.V[-1]:.6g}, "
        f"final biomass={result.X[-1]:.6g}, "
        f"final substrate={result.S[-1]:.6g}"
    )


def test_initial_volume_validation_rejects_mismatched_state_volume():
    params = BioreactorParams(V_0=1.0)
    y0 = np.array([0.1, 20.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="Initial volume mismatch"):
        validate_initial_volume(y0, params)


@pytest.mark.parametrize(
    "dilution, expected_washout, s_target, x_target",
    [
        (0.1, False, 9.31, 3.26),  # D < mu(S_f), washout not expected
        (0.6, True, 50.0, 0.0),  # D > mu(S_f), washout expected
    ],
)
def test_chemostat_analysis_matches_expected_regimes(
    dilution, expected_washout, s_target, x_target
):
    base = BioreactorParams()
    params = replace(base, F=dilution * base.V_0)
    analysis = analyze_chemostat(params)
    assert analysis.washout_expected is expected_washout
    assert analysis.S_star == pytest.approx(s_target, abs=1e-2)
    assert analysis.X_star == pytest.approx(x_target, abs=1e-2)


@pytest.mark.parametrize(
    "dilution, t_end, tolerance",
    [
        (0.1, 300.0, 1e-4),
        (0.2, 300.0, 1e-4),
        (0.3, 1000.0, 1e-4),
        (0.4, 300.0, 1e-4),
    ],
)
def test_chemostat_simulation_converges_to_analysis(dilution, t_end, tolerance):
    base = BioreactorParams()
    params = replace(base, F=dilution * base.V_0)
    y0 = initial_state(X0=0.1, S0=20.0, P0=0.0, params=params)
    analysis = analyze_chemostat(params)
    result = simulate(
        "chemostat",
        params,
        y0,
        (0.0, t_end),
        rtol=1e-9,
        atol=1e-11,
    )
    assert result.raw.success, result.raw.message
    assert result.chemostat == analysis

    err = chemostat_steady_state_error(result, analysis)
    assert err < tolerance


def test_constant_feed_fed_batch_tracks_quasi_steady_substrate():
    params = BioreactorParams(F=0.1, V_max=30.0)
    y0 = initial_state(X0=0.1, S0=20.0, P0=0.0, params=params)
    result = simulate(
        "fedbatch_const",
        params,
        y0,
        (0.0, 80.0),
        rtol=1e-9,
        atol=1e-11,
    )
    assert result.raw.success, result.raw.message
    assert not result.near_volume_cap

    S_qss = fed_batch_const_qss_substrate(result, params)
    discrepancy = np.abs(result.S - S_qss)
    early = (result.t >= 20.0) & (result.t <= 30.0)
    late = (result.t >= 60.0) & (result.t <= 80.0)
    assert np.median(discrepancy[late]) < np.median(discrepancy[early])
    assert discrepancy[-1] < discrepancy[early][0]


def test_constant_feed_qss_substrate_rejects_invalid_dilution_regime():
    params = BioreactorParams(F=0.5, V_max=30.0)
    y0 = initial_state(X0=0.1, S0=20.0, P0=0.0, params=params)
    result = simulate(
        "fedbatch_const",
        params,
        y0,
        (0.0, 1.0),
        rtol=1e-9,
        atol=1e-11,
    )

    assert result.raw.success, result.raw.message
    with pytest.raises(ValueError, match=r"QSS substrate requires D\(t\) < mu_max"):
        fed_batch_const_qss_substrate(result, params)


@pytest.mark.parametrize(
    "params",
    [
        BioreactorParams(),
        BioreactorParams(S_f=30.0),
        BioreactorParams(mu_max=0.6, K_S=20.0),
    ],
)
def test_chemostat_bifurcation_curve_has_washout_threshold(params):
    D_values = np.linspace(0.01, params.mu_max * 1.2, 200)
    curve = chemostat_bifurcation_curve(params, D_values)
    X_star = np.array([point.X_star for point in curve])
    washout = np.array([point.washout_expected for point in curve])
    D_crit = curve[0].washout_threshold

    assert np.array_equal(washout, D_values >= D_crit)
    assert np.all(X_star[D_values >= D_crit] == pytest.approx(0.0))
    assert np.all(X_star[D_values < D_crit] > 0.0)
    assert np.all(np.diff(X_star[D_values < D_crit]) < 0.0)
