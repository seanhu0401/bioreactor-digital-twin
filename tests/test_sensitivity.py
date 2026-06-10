import pytest

from bioreactor.sensitivity import (
    SensitivityResult,
    oat_sensitivity_analysis,
    tornado_ranking,
)


def test_oat_sensitivity_returns_one_row_per_parameter():
    rows = oat_sensitivity_analysis(
        sweep_params=["mu_max", "K_S"],
        perturbation_fraction=0.2,
        initial_conditions=[0.1, 20.0, 0.0],
        metric="X",
    )

    assert [row.parameter_name for row in rows] == ["mu_max", "K_S"]


def test_oat_sensitivity_records_low_and_high_parameter_values():
    rows = oat_sensitivity_analysis(
        sweep_params=["mu_max"],
        perturbation_fraction=0.2,
        initial_conditions=[0.1, 20.0, 0.0],
        metric="X",
    )

    row = rows[0]
    assert row.parameter_low_value == pytest.approx(0.46 * 0.8)
    assert row.parameter_high_value == pytest.approx(0.46 * 1.2)


@pytest.mark.parametrize("metric", ["X", "P"])
def test_oat_sensitivity_supports_metrics(metric):
    rows = oat_sensitivity_analysis(
        sweep_params=["mu_max"],
        perturbation_fraction=0.2,
        initial_conditions=[0.1, 20.0, 0.0],
        metric=metric,
    )

    assert len(rows) == 1
    assert rows[0].raw_low_value >= 0.0
    assert rows[0].raw_high_value >= 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": "S"}, "Unsupported metric"),
        ({"perturbation_fraction": 0.0}, "perturbation_fraction must be greater than 0"),
        ({"perturbation_fraction": 1.0}, "perturbation_fraction must be greater than 0"),
        ({"initial_conditions": [0.1, 20.0]}, "initial_conditions must have 3 elements"),
        ({"sweep_params": ["not_a_param"]}, "Invalid sweep parameter"),
        ({"sweep_params": []}, "sweep_params must not be empty"),
    ],
)
def test_oat_sensitivity_rejects_invalid_inputs(kwargs, match):
    base_kwargs = {
        "sweep_params": ["mu_max"],
        "perturbation_fraction": 0.2,
        "initial_conditions": [0.1, 20.0, 0.0],
        "metric": "X",
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match):
        oat_sensitivity_analysis(**base_kwargs)


def test_tornado_ranking_sorts_by_largest_absolute_response():
    rows = [
        SensitivityResult("low", 0.8, 1.2, 1.0, 1.1, -0.05, 0.10),
        SensitivityResult("high", 0.8, 1.2, 1.0, 1.1, -0.40, 0.20),
        SensitivityResult("mid", 0.8, 1.2, 1.0, 1.1, -0.15, 0.12),
    ]

    ranked = tornado_ranking(rows)

    assert [row.parameter_name for row in ranked] == ["high", "mid", "low"]
