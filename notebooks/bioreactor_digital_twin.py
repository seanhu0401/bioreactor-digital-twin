import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    from bioreactor.params import BioreactorParams
    from bioreactor.sensitivity import oat_sensitivity_analysis, tornado_ranking
    from bioreactor.simulate import simulate
    from bioreactor.state import initial_state
    from bioreactor.steady_state import chemostat_bifurcation_curve

    SAVE_FIGURES = False

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-colorblind")

    params = {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.formatter.use_mathtext": True,
        "axes.grid": True,
        "errorbar.capsize": 1.7,
        "xtick.labelsize": 12,
        "xtick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.labelsize": 12,
        "ytick.major.size": 4,
        "ytick.major.width": 1,
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "grid.color": "#000000",
        "grid.alpha": 0.1,
        "legend.fontsize": "small",
    }

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))

    def save_figure(figure, filename: str):
        output_path = figures_dir / filename
        if SAVE_FIGURES:
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
        return output_path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bioreactor Digital Twin

    This notebook reproduces the stable project figures for a mechanistic bioreactor digital twin. It uses the tested package API for simulation, chemostat steady-state analysis, and one-at-a-time sensitivity analysis.

    The outputs are deterministic and saved under `figures/` for README and publication use. A companion interactive notebook is still under development, which will expose the same model through sliders and controls for interactivity.
    """)
    return


@app.cell(hide_code=True)
def _():
    fedbatch_params = BioreactorParams(F=0.1, V_max=30.0)
    fedbatch_y0 = initial_state(X0=0.1, S0=20.0, P0=0.0, params=fedbatch_params)
    fedbatch_result = simulate(
        "fedbatch_const",
        fedbatch_params,
        fedbatch_y0,
        (0.0, 80.0),
        rtol=1e-9,
        atol=1e-11,
    )
    return fedbatch_params, fedbatch_result


@app.cell(hide_code=True)
def _(fedbatch_params, fedbatch_result):
    mo.md(f"""
    ## Fed-Batch Trajectory

    The baseline run uses a constant feed rate of
    `{fedbatch_params.F:.2f} L/h` and a large volume cap of
    `{fedbatch_params.V_max:.1f} L`, so the trajectory stays in the
    analytical mass-balance regime used by the package tests.

    Solver status: `{fedbatch_result.raw.message}`
    """)
    return


@app.cell(hide_code=True)
def _(fedbatch_result):
    with plt.rc_context(params):
        fedbatch_fig, fedbatch_axes = plt.subplots(
            3, 1, figsize=(6.5, 6.6), sharex=True
        )

        fedbatch_axes[0].plot(fedbatch_result.t, fedbatch_result.X, label="Biomass X")
        fedbatch_axes[0].plot(fedbatch_result.t, fedbatch_result.S, label="Substrate S")
        product_line = fedbatch_axes[0].plot(
            fedbatch_result.t, fedbatch_result.P, label="Product P"
        )[0]
        fedbatch_axes[0].set_ylabel("Concentration [g/L]")
        fedbatch_axes[0].legend(loc="best")
        fedbatch_axes[0].set_title("Constant-Feed Fed-Batch Trajectory")

        # Product on its own axis: P is tiny against X/S on the shared scale above,
        # so a dedicated panel makes its monotonic accumulation legible.
        fedbatch_axes[1].plot(
            fedbatch_result.t, fedbatch_result.P, color=product_line.get_color()
        )
        fedbatch_axes[1].set_ylabel("Product P [g/L]")

        fedbatch_axes[2].plot(fedbatch_result.t, fedbatch_result.V, color="tab:purple")
        fedbatch_axes[2].set_xlabel("Time [h]")
        fedbatch_axes[2].set_ylabel("Volume [L]")

        fedbatch_fig.tight_layout()
        fedbatch_path = save_figure(fedbatch_fig, "fedbatch_trajectory.png")
    fedbatch_fig
    return (fedbatch_path,)


@app.cell(hide_code=True)
def _():
    chemostat_params = BioreactorParams()
    chemostat_probe = chemostat_bifurcation_curve(chemostat_params, np.array([0.01]))[0]
    dilution_rates = np.linspace(0.001, chemostat_probe.washout_threshold * 1.25, 240)
    chemostat_curve = chemostat_bifurcation_curve(chemostat_params, dilution_rates)
    washout_threshold = chemostat_curve[0].washout_threshold
    biomass_steady_state = np.array([point.X_star for point in chemostat_curve])
    substrate_steady_state = np.array([point.S_star for point in chemostat_curve])
    return (
        biomass_steady_state,
        chemostat_params,
        dilution_rates,
        substrate_steady_state,
        washout_threshold,
    )


@app.cell(hide_code=True)
def _(chemostat_params, washout_threshold):
    mo.md(rf"""
    ## Chemostat Washout Boundary

    The chemostat steady-state calculation maps the dilution rate to analytical expressions for substrate and biomass equilibria. For the default feed substrate concentration of `{chemostat_params.S_f:.1f} g/L`, washout begins at approximately `{washout_threshold:.3f}` $\mathtt h^{{-1}}$.
    """)
    return


@app.cell(hide_code=True)
def _(
    biomass_steady_state,
    dilution_rates,
    substrate_steady_state,
    washout_threshold,
):
    with plt.rc_context(params):
        chemostat_fig, chemostat_axis = plt.subplots(figsize=(6.5, 4.0))
        chemostat_axis.plot(
            dilution_rates,
            biomass_steady_state,
            label="Biomass steady state (X*)",
            color="tab:green",
        )

        chemostat_axis.set_xlabel(r"Dilution rate (D) [h$^{-1}$]")
        chemostat_axis.set_ylabel("Biomass (X*) [g/L]", color="tab:green")
        chemostat_axis.tick_params(axis="y", labelcolor="tab:green")
        chemostat_axis.axvline(
            washout_threshold,
            color="tab:red",
            linestyle="--",
            label="Washout threshold",
        )
        chemostat_axis.grid(
            True,
            axis="both",
            color="#d0d0d0",
            linestyle="-",
            linewidth=0.7,
            alpha=0.65,
        )

        substrate_axis = chemostat_axis.twinx()
        substrate_axis.plot(
            dilution_rates,
            substrate_steady_state,
            label="Substrate steady state (S*)",
            color="tab:blue",
        )
        substrate_axis.set_ylabel("Substrate (S*) [g/L]", color="tab:blue")
        substrate_axis.tick_params(axis="y", labelcolor="tab:blue")
        substrate_axis.spines["right"].set_visible(True)
        substrate_axis.grid(
            True,
            axis="y",
            color="tab:blue",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        handles, labels = chemostat_axis.get_legend_handles_labels()
        substrate_handles, substrate_labels = substrate_axis.get_legend_handles_labels()
        substrate_axis.legend(
            handles + substrate_handles, labels + substrate_labels, loc=6
        )
        chemostat_axis.set_title("Chemostat Bifurcation and Washout")
        chemostat_fig.tight_layout()

        chemostat_path = save_figure(chemostat_fig, "chemostat_bifurcation.png")
    chemostat_fig
    return (chemostat_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One-at-a-Time Sensitivity

    The tornado plot perturbs each selected parameter by $\pm$20% around the package defaults and reports the normalized response in maximum product concentration for the curated constant-feed fed-batch scenario.
    """)
    return


@app.cell(hide_code=True)
def _():
    sensitivity_rows = oat_sensitivity_analysis(
        sweep_params=["mu_max", "K_S", "Y_xs", "Y_ps", "F"],
        perturbation_fraction=0.2,
        initial_conditions=[0.1, 20.0, 0.0],
        metric="P",
    )
    ranked_sensitivity = tornado_ranking(sensitivity_rows)

    def parameter_display_name(parameter_name: str) -> str:
        if parameter_name == "mu_max":
            return r"$\mu_{max}$"
        if "_" not in parameter_name:
            return parameter_name
        base, subscript = parameter_name.split("_", 1)
        return rf"${base}_{{{subscript}}}$"

    sensitivity_names = [
        parameter_display_name(row.parameter_name) for row in ranked_sensitivity
    ]
    low_effects = np.array(
        [100.0 * row.normalized_low_value for row in ranked_sensitivity]
    )
    high_effects = np.array(
        [100.0 * row.normalized_high_value for row in ranked_sensitivity]
    )
    sensitivity_positions = np.arange(len(ranked_sensitivity))
    return high_effects, low_effects, sensitivity_names, sensitivity_positions


@app.cell(hide_code=True)
def _(high_effects, low_effects, sensitivity_names, sensitivity_positions):
    with plt.rc_context(params):
        tornado_fig, tornado_axis = plt.subplots(figsize=(6.5, 3.8))
        tornado_axis.barh(
            sensitivity_positions,
            low_effects,
            color="tab:orange",
            label="Parameter -20%",
        )
        tornado_axis.barh(
            sensitivity_positions,
            high_effects,
            color="tab:cyan",
            label="Parameter +20%",
        )
        tornado_axis.axvline(0.0, color="black", linewidth=0.8)
        tornado_axis.set_yticks(sensitivity_positions)
        tornado_axis.set_yticklabels(sensitivity_names)
        tornado_axis.invert_yaxis()
        tornado_axis.set_xlabel("Change in max product [% from baseline]")
        tornado_axis.set_title("OAT Sensitivity Tornado Ranking")
        tornado_axis.legend(loc="best")
        tornado_fig.tight_layout()

        tornado_path = save_figure(tornado_fig, "oat_sensitivity_tornado.png")
    tornado_fig
    return (tornado_path,)


@app.cell(hide_code=True)
def _(chemostat_path, fedbatch_path, tornado_path):
    mo.md(f"""
    ## Exported Figures

    Static figure files are stored under `{figures_dir}/` for public rendering.
    This demo does not overwrite generated assets unless `SAVE_FIGURES` is set to
    `True` in the setup cell.

    - `{fedbatch_path}`
    - `{chemostat_path}`
    - `{tornado_path}`
    """)
    return


if __name__ == "__main__":
    app.run()
