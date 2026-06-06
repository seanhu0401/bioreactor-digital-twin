# Bioreactor Digital Twin

Mechanistic fed-batch bioreactor simulation. A reactive [marimo](https://marimo.io) notebook drives a small, tested `bioreactor/` model package supporting batch, fed-batch, and chemostat modes with mass-balance and washout testing. The notebook is currently under development and polishing. 

## Layout

- `bioreactor/` — model package: `kinetics`, `models` (RHS callables), `feeds`, `sensitivity`, `validation`, `params`, `model_types`. 
- `notebooks/` — the marimo notebook 
- `tests/` — invariant tests (`test_fed_batch_invariants.py`).

## Under Development
- Demo Marimo notebook
- `sensitivity.py` and `validation.py`
- `test_fed_batch_invariants.py`

## Develop

~~~bash
uv sync
uv run marimo edit notebooks/        # interactive notebook
uv run pytest                        # invariant tests
~~~
