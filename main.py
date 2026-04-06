import argparse
import os
import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nolds  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]
from mmar import (
    calculate_hurst_for_segments,
    calculate_lognormal_cascade,
    calculate_magnitude_parameter,
    calculate_mmar_returns,
    calculate_scaling_exponent,
    calculate_trading_time,
    define_time_window,
    estimate_multifractal_spectrum,
    generate_fbm_path,
    option_pricer,
)
from sklearn.exceptions import UndefinedMetricWarning  # type: ignore[import-untyped]

import math


def _configure_matplotlib_backend() -> None:
    forced_backend = os.getenv("MMAR_MPL_BACKEND")
    if forced_backend:
        plt.switch_backend(forced_backend)
        return

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            plt.switch_backend("Agg")
    else:
        plt.switch_backend("Agg")


_configure_matplotlib_backend()


NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}
IS_NON_INTERACTIVE_BACKEND = (
    matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS
)
PLOTS_DIR = Path("plots")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "plot"


def finalize_plot(filename_stem: str) -> None:
    if IS_NON_INTERACTIVE_BACKEND:
        PLOTS_DIR.mkdir(exist_ok=True)
        output_path = PLOTS_DIR / f"{filename_stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MMAR options pricing workflow."
    )
    parser.add_argument("--ticker-symbol", default="COIN")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2025-03-24")
    parser.add_argument("--risk-free-rate", type=float, default=0.043)
    parser.add_argument("--days-to-expiration", type=int, default=45)
    return parser.parse_args()


warnings.simplefilter(action="ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore", message=".*RANSAC did not reach consensus.*", category=RuntimeWarning
)


def plot_mmar_paths(paths, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))

    for path in paths:
        plt.plot(path)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    finalize_plot(_slugify(title))


def price_option_map_for_strikes(
    paths,
    center: float = 80,
    step: float = 5,
    num_strikes: int = 5,
    r: float = 0.05,
    T: float = 1,
) -> dict[float, float]:

    option_prices: dict[float, float] = {}

    # Calculate prices for many strikes
    for i in range(1, num_strikes + 1):
        strike = center - i * step
        option_prices[strike] = option_pricer(paths, strike, r, T, option_type="put")

    # Calculate prices for strikes at and above the center
    for i in range(num_strikes + 1):
        strike = center + i * step
        option_prices[strike] = option_pricer(paths, strike, r, T, option_type="call")

    return option_prices


# 1) Data import and base parameters

args = parse_args()

ticker_symbol = args.ticker_symbol
start_date = args.start_date
end_date = args.end_date

r = args.risk_free_rate
days_to_expiration = args.days_to_expiration
T = days_to_expiration / 365


# Keep historical close handling stable across yfinance releases.
data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False)

prices = data["Close"]
# print(prices.describe())
prices.plot()
finalize_plot("historical_close_prices")


prices_fixed = prices.copy()


# 2) fBM simulation and Monte Carlo option pricing

# H < 0.5: anti-persistent (mean-reverting tendency).
# H = 0.5: standard Brownian motion.
# H > 0.5: persistent (trend-following tendency).


# num_segments = 625
num_segments = round(len(prices_fixed) / 10)
hurst_values = calculate_hurst_for_segments(prices, num_segments)

plt.figure(figsize=(10, 6))
plt.hist(hurst_values, bins=10, edgecolor="black", alpha=0.7)
plt.title(f"Distribution {ticker_symbol}'s Hurst Exponent (H) across segments")
plt.xlabel("Hurst Exponent (H)")
plt.ylabel("Frequency")
plt.axvline(np.mean(hurst_values), color="red")
plt.grid(True, which="both", ls="--")
# plt.savefig('dist_oil_h.png')
finalize_plot("hurst_distribution")


print(f"{ticker_symbol}'s Hurst Value: {np.mean(hurst_values)}")
h_underlying = np.mean(hurst_values)

fbm_underlying = nolds.fbm(100, H=h_underlying)
fbm_underlying = pd.Series(fbm_underlying)

plt.figure(figsize=(10, 6))
plt.plot(fbm_underlying)
# plt.axhline(y=0.0, color='grey', linestyle='-.', linewidth=1)
plt.title(f"Simulated Fractional Brownian Motion with {ticker_symbol}'s Hurst")
# plt.legend(loc=0)
plt.xlabel("Days")
plt.ylabel("Price")
finalize_plot("simulated_fbm_with_empirical_hurst")


n = 1000  # Number of simulated time steps.
hurst_mean = h_underlying  # Use the empirical H estimate.
prices = generate_fbm_path(n, hurst_mean, s0=80)

# Enforce non-negative prices.
prices = np.where(prices > 0, prices, 0)

# Plot one simulated fBM price path.
plt.figure(figsize=(10, 6))
plt.plot(prices)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(
    f"Simulated {ticker_symbol} Price Progression using fBm with H={hurst_mean:.2f} and downside limit"
)
plt.grid(True)
# plt.savefig('oil_fbm_downside_s0.png')
finalize_plot("simulated_fbm_price_path")


# 3) Multifractal Model of Asset Returns (MMAR)

# 3.1 Compute log returns from historical close prices.


close_return = prices_fixed.copy()
# print(close_return)
# NOTE: Keep raw returns unchanged; avoid manual clipping.
close_return = np.log(close_return / close_return.shift())
close_return = close_return.dropna()
# print(f"Close Returns: {close_return}")


# 3.2 Define q moments for the partition function.

# Select the q-moment range used for multifractal estimation.

power = 3  # Keeps tau(q) curvature stable for this setup.
q = np.linspace(0.01, 3, 120)
# print(q[:10])

# Plot q values used in the partition function.
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.title("The distribution of q_values for Partition Function")
plt.xlabel(f"Index of q\n(Total {len(q)})")
plt.ylabel("raw moment")
plt.plot(q, marker="o")
# plt.savefig('dist_q_values.png')
finalize_plot("q_values_distribution")


# 3.3 Define candidate window sizes.

min_window = 10
max_window = len(prices_fixed)
print("length of the training prices data: ", len(prices_fixed))

window_sizes = define_time_window(
    min_window=min_window, max_window=max_window, base=10, interval=0.25
)
print(f"Window sizes: {window_sizes}")

# 3.4 Compute partition values Fq and scaling exponents tau(q).

# print("Contents of q:", q)

Fq, tau_q_list = calculate_scaling_exponent(
    window_sizes, prices_fixed, q, ticker_symbol
)
# print(f"{Fq.head()}")

# Plot tau(q) against q.
# Monofractal signals are approximately linear in tau(q).
# Multifractal signals show nonlinear curvature in tau(q).

plt.figure(figsize=(10, 6))
y = np.array(Fq.index)
plt.plot(y, tau_q_list)
plt.title("Tau(q) Values")
# plt.savefig('tau_values.png')
finalize_plot("tau_q_values")


# 3.5 Set the Hurst component used by MMAR.

H = h_underlying


# 3.6 Estimate the multifractal spectrum f(alpha).

# Apply the Legendre transform from tau(q) to f(alpha).
# Sweeping q traces the full multifractal spectrum.


F_A, parameters_of_spectrum = estimate_multifractal_spectrum(
    tau_q_list, q.tolist(), 0, len(q) - 1
)


# 3.7 Extract the most probable Holder exponent.

# Second returned parameter is the Holder exponent at the spectrum peak.
a0 = parameters_of_spectrum[1]


# 3.8 Derive log-normal cascade parameters.

simulated_lambda = a0 / H
# print(simulated_lambda)

# Variance relation for base-2 branching: sigma^2 = 2 * (lambda - 1) / ln(2)
simulated_variance = 2 * (simulated_lambda - 1) / np.log(2)
simulated_sigma = np.sqrt(simulated_variance)


# 3.9 Determine cascade depth from the simulation horizon.

# Compute K for dyadic branching (base 2).
days_for_simulation = 252
K = math.ceil(np.log2(days_for_simulation))  # k value


# 3.10 Generate a log-normal multiplicative cascade.

new_cascade = calculate_lognormal_cascade(
    layers=K, v=1, ln_lambda=np.log(simulated_lambda), ln_sigma=np.log(simulated_sigma)
)
new_cascade = list(np.array(new_cascade).flat)


# 3.11 Compute the trading-time transformation.

trading_time = calculate_trading_time(layers=K, lognormal_cascade=new_cascade)
# trading_time = trading_time[:252]
trading_time = trading_time[:days_to_expiration]  # Align with option DTE.

plt.plot(trading_time)
plt.title("Trading time (theta_t)")
plt.xlabel("Days")
plt.ylabel("CDF values")
plt.grid(True)
finalize_plot("trading_time_theta_t")


# 3.12 Calibrate fBM magnitude to the observed return volatility.


print(f"Real Std-Dev: {np.std(close_return[ticker_symbol])}")
magnitude_parameter = calculate_magnitude_parameter(
    initial_value=0.5,
    eps=0.01,
    steps=0.5,
    number_of_path=100,
    real_std=np.std(close_return[ticker_symbol]),
    layers=K,
    hurst_exponent=h_underlying,
)


# 3.13 Simulate MMAR returns and price paths.

number_of_path = 1000
s0 = prices_fixed.iloc[-1][ticker_symbol]
print(f"Latest price: {s0}")

mmar_returns, prices_paths = calculate_mmar_returns(
    S0=s0,
    number_of_path=number_of_path,
    layers=K,
    hurst_exponent=h_underlying,
    trading_time=trading_time,
    magnitude_parameter=magnitude_parameter,
)

plot_mmar_paths(mmar_returns, "Simulated MMAR returns", "Days", "Returns")
plot_mmar_paths(prices_paths, "Simulated MMAR Prices", "Days", f"{ticker_symbol} Price")

mean_return = np.mean(mmar_returns, axis=0)
plot_mmar_paths(
    [mean_return], "Mean of MMAR generated returns", "Time\n(days)", "Returns"
)

mean_prices = s0 * np.exp(mean_return)
plot_mmar_paths(
    [mean_prices], "Prices derived from the mean return", "Time\n(days)", "Prices"
)


# 3.14 Price options from simulated terminal distributions.


strike_p = s0  # Strike price

print(
    f"Fair Call option price ATM, {days_to_expiration} DTE: ",
    option_pricer(prices_paths, strike_p, r, T, option_type="call"),
)
print(
    f"Fair Put option price ATM:, , {days_to_expiration} DTE: ",
    option_pricer(prices_paths, strike_p, r, T, option_type="put"),
)


steps = int(s0) * 0.01
option_prices = price_option_map_for_strikes(
    prices_paths, center=s0, step=steps, num_strikes=10, r=r, T=T
)

print(f"{'Side':<6} {'Strike':>10} {'Price':>12}")
print("-" * 32)

for strike, price in sorted(option_prices.items()):
    side = "Put" if float(strike) < float(s0) else "Call"
    print(f"{side:<6} {float(strike):>10.2f} {float(price):>12.4f}")
