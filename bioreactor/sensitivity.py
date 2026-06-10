from dataclasses import dataclass, fields, replace
from typing import Sequence

import numpy as np

from .params import BioreactorParams
from .simulate import simulate
from .state import initial_state


@dataclass(frozen=True)
class SensitivityResult:
    parameter_name: str
    parameter_low_value: float
    parameter_high_value: float
    raw_low_value: float
    raw_high_value: float
    normalized_low_value: float
    normalized_high_value: float


def oat_sensitivity_analysis(
    sweep_params: Sequence[str],
    perturbation_fraction: float,
    initial_conditions: Sequence[float],
    metric: str,
) -> list[SensitivityResult]:
    """
    Perform OAT sensitivity analysis on the bioreactor model.

    Currently it only supports one level of perturbation across all parameters.
    The perturbations are relative to the current parameter values.


    Parameters
    ----------
    sweep_params : Sequence[str]
        List of parameter names to perturb.
    perturbation_fraction : float
        The fraction of the current parameter value(s) to use as the perturbation magnitude.
    initial_conditions : Sequence[float]
        The initial conditions for the simulation. Should always be in a shape and order of [X0, S0, P0].
    metric : str
        The metric to use for sensitivity analysis. Currently only max biomass overtime and max product are supported. ("X" or "P" for input)

    Returns
    -------
    list[SensitivityResult]
        A list of sensitivity results for each parameter.
    """
    if metric not in ["X", "P"]:
        raise ValueError(
            f"Unsupported metric: {metric}. Supported metrics are 'X' or 'P', got {metric}"
        )
    if not 0 < perturbation_fraction < 1:
        raise ValueError(
            f"perturbation_fraction must be greater than 0 and less than 1, got {perturbation_fraction}"
        )
    if len(initial_conditions) != 3:
        raise ValueError(
            f"initial_conditions must have 3 elements in the order of [X0, S0, P0], got {len(initial_conditions)}"
        )

    valid_params = {f.name for f in fields(BioreactorParams)}
    invalid_params = [name for name in sweep_params if name not in valid_params]
    if invalid_params:
        raise ValueError(
            f"Invalid sweep parameter(s): {invalid_params}. Must be one of {sorted(valid_params)}"
        )

    if not sweep_params:
        raise ValueError("sweep_params must not be empty")

    baseline_params = BioreactorParams()

    y0 = initial_state(*initial_conditions, params=baseline_params)
    baseline_result = simulate(
        mode="fedbatch_const", params=baseline_params, y0=y0, t_span=(0, 80.0)
    )
    baseline_metric: float = max(getattr(baseline_result, metric))
    if np.isclose(baseline_metric, 0):
        raise ValueError(
            "Baseline metric is 0, cannot perform sensitivity analysis. Cannot divide by zero."
        )

    sensitivity_rows: list[SensitivityResult] = []

    for parameter_name in sweep_params:
        baseline_value: float = getattr(baseline_params, parameter_name)
        low_params = replace(
            baseline_params,
            **{parameter_name: baseline_value * (1 - perturbation_fraction)},
        )
        high_params = replace(
            baseline_params,
            **{parameter_name: baseline_value * (1 + perturbation_fraction)},
        )
        low_result = simulate(
            mode="fedbatch_const", params=low_params, y0=y0, t_span=(0, 80.0)
        )
        high_result = simulate(
            mode="fedbatch_const", params=high_params, y0=y0, t_span=(0, 80.0)
        )
        low_metric: float = max(getattr(low_result, metric))
        high_metric: float = max(getattr(high_result, metric))
        sensitivity_rows.append(
            SensitivityResult(
                parameter_name=parameter_name,
                parameter_low_value=getattr(low_params, parameter_name),
                parameter_high_value=getattr(high_params, parameter_name),
                raw_low_value=low_metric,
                raw_high_value=high_metric,
                normalized_low_value=(low_metric - baseline_metric) / baseline_metric,
                normalized_high_value=(high_metric - baseline_metric) / baseline_metric,
            )
        )

    return sensitivity_rows


def tornado_ranking(
    sensitivity_results: list[SensitivityResult],
) -> list[SensitivityResult]:
    return sorted(
        sensitivity_results,
        key=lambda row: max(
            abs(row.normalized_low_value),
            abs(row.normalized_high_value),
        ),
        reverse=True,
    )
