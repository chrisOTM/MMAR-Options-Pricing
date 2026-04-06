# MMAR Options Pricing

This repository contains a research-style Python workflow for simulating asset dynamics with fractional Brownian motion (fBM) and the Multifractal Model of Asset Returns (MMAR), then pricing European call and put options from the simulated paths.

It is a lightweight script project rather than a packaged library. The main workflow lives in `main.py`, and the reusable numerical routines live in `mmar.py`.

## What The Script Does

Running `python main.py` performs a full end-to-end experiment:

1. Downloads historical price data for a hardcoded ticker with `yfinance`
2. Estimates Hurst exponents across rolling segments of the historical series
3. Simulates and visualizes fractional Brownian motion paths using the empirical Hurst estimate
4. Computes multifractal scaling statistics such as `tau(q)`
5. Estimates the multifractal spectrum `f(alpha)`
6. Builds a log-normal multiplicative cascade and trading-time transformation
7. Calibrates an fBM magnitude parameter to historical return volatility
8. Simulates MMAR return paths and converts them to price paths
9. Prices ATM and strike-grid European options from the simulated terminal prices

The script also produces several plots that help inspect intermediate steps of the workflow.

## Repository Structure

```text
.
|-- main.py            # End-to-end workflow, plotting, experiment parameters
|-- mmar.py            # MMAR/fBM simulation, scaling, and option pricing helpers
|-- requirements.txt   # Runtime dependencies
|-- AGENTS.md          # Repo-specific guidance for coding agents
|-- plots/             # Generated plot outputs from script runs
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in dependency pins target modern Python versions. If installation fails, verify the interpreter in the active venv with `python --version` before debugging package issues.

Optional development tools used in the README checks:

```bash
python -m pip install ruff black mypy pytest
```

## Dependencies

The runtime code depends on:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `nolds`
- `hurst`
- `fbm`
- `scikit-learn`
- `tqdm`

These are listed in `requirements.txt`.

## How To Run

Run the full workflow:

```bash
python main.py
```

Override the main experiment inputs from the command line:

```bash
python main.py \
  --ticker-symbol COIN \
  --start-date 2020-01-01 \
  --end-date 2025-03-24 \
  --risk-free-rate 0.043 \
  --days-to-expiration 45
```

Important runtime notes:

- `main.py` downloads historical market data, so network access is required.
- The workflow uses Monte Carlo simulation and repeated calibration loops, so runtime can be noticeable.
- The script exposes the main market and option inputs as CLI flags; other simulation settings remain in `main.py`.

## Experiment Parameters

The main script accepts these core experiment inputs as optional CLI parameters:

- `--ticker-symbol` with default `COIN`
- `--start-date` with default `2020-01-01`
- `--end-date` with default `2025-03-24`
- `--risk-free-rate` with default `0.043`
- `--days-to-expiration` with default `45`

If you run `python main.py` with no arguments, these defaults are used.

The script still sets other simulation parameters directly in code, such as:

- the q-moment grid used for multifractal estimation
- the number of calibration paths
- the number of simulated MMAR paths
- the strike spacing used for option pricing

If you want to change those deeper simulation settings, `main.py` is still the first place to edit.

## How The Workflow Is Wired

### 1. Historical Data Download

`main.py` downloads OHLCV data with `yfinance` and works primarily with the `Close` series.

The first plot saved or displayed is the historical close-price chart.

### 2. Hurst Estimation

The script divides the price history into equal-length segments and computes a Hurst exponent for each segment using `nolds.hurst_rs`.

This produces:

- a histogram of segment-level Hurst exponents
- an empirical mean Hurst estimate used later in both fBM and MMAR steps

### 3. Fractional Brownian Motion Simulation

The workflow uses the empirical Hurst estimate to:

- simulate a sample fBM series with `nolds.fbm`
- generate a separate price path with `generate_fbm_path()` from `mmar.py`

These plots are meant as diagnostic comparisons before the multifractal modeling stage.

### 4. Multifractal Scaling Analysis

The script converts historical close prices to log returns and evaluates scaling behavior over a range of q-moments and time windows.

Key functions in `mmar.py`:

- `define_time_window()` builds logarithmically spaced window sizes
- `calculate_scaling_exponent()` computes partition values `Fq` and estimates `tau(q)`
- `estimate_multifractal_spectrum()` fits a quadratic approximation and derives spectrum parameters

This stage generates plots for:

- q-value distribution
- `tau(q)` values

### 5. MMAR Time Change

Using the estimated multifractal parameters, the script:

- derives log-normal cascade parameters
- builds a dyadic multiplicative cascade with `calculate_lognormal_cascade()`
- converts the cascade into a cumulative trading-time process with `calculate_trading_time()`

The trading-time series is then truncated to the chosen option days-to-expiration horizon.

### 6. Magnitude Calibration

Before generating MMAR paths, the script calibrates an fBM magnitude parameter so that simulated volatility better matches the historical return standard deviation.

This happens in `calculate_magnitude_parameter()` and can take noticeable time because it repeatedly simulates multiple paths until the target mismatch falls below a tolerance.

### 7. MMAR Path Simulation

`calculate_mmar_returns()`:

- simulates fBM paths with the calibrated magnitude parameter
- samples them along the MMAR trading-time clock
- treats the sampled process as log returns
- converts returns into simulated price paths via `S0 * exp(return)`

The script then visualizes:

- simulated MMAR returns
- simulated MMAR prices
- mean MMAR-generated returns
- prices implied by the mean return path

### 8. Option Pricing

European options are priced from the simulated MMAR terminal prices using `option_pricer()`.

`main.py` prints:

- the fair ATM call price
- the fair ATM put price
- a strike grid of call and put prices around the current spot level

## Generated Outputs

When the script runs in a non-interactive or headless environment, plots are saved to `plots/*.png`.

Typical outputs include:

- `historical_close_prices.png`
- `hurst_distribution.png`
- `simulated_fbm_with_empirical_hurst.png`
- `simulated_fbm_price_path.png`
- `q_values_distribution.png`
- `tau_q_values.png`
- `trading_time_theta_t.png`
- `simulated_mmar_returns.png`
- `simulated_mmar_prices.png`
- `mean_of_mmar_generated_returns.png`
- `prices_derived_from_the_mean_return.png`

In an interactive environment, Matplotlib may display figures instead of writing them to disk.

## Matplotlib Backend Behavior

`main.py` configures the backend automatically:

- If `MMAR_MPL_BACKEND` is set, that backend is used.
- If a display is available, it tries `TkAgg`.
- Otherwise it falls back to `Agg`.

Examples:

```bash
MMAR_MPL_BACKEND=Agg python main.py
MMAR_MPL_BACKEND=TkAgg python main.py
```

Using `Agg` is useful for servers, CI, or other headless environments.

## Fast Verification Commands

If you do not want to run the full experiment, use a fast syntax check:

```bash
python -m py_compile main.py mmar.py
```

Additional optional checks:

```bash
ruff check main.py mmar.py
black --check main.py mmar.py
mypy main.py mmar.py
```

## Tests

There is currently no committed `tests/` directory in this repository.

If a test suite is added later, use `pytest` in the normal way.

## Notes And Constraints

- This codebase is research/prototyping oriented rather than productionized.
- The workflow is controlled by script-level constants rather than config files or command-line arguments.
- Plot files under `plots/` are generated artifacts from running the script.
- Because the script downloads live market data, results may vary over time.
- Monte Carlo outputs also vary unless you add explicit random seeding.
