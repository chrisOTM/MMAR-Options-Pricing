# Core numeric and simulation dependencies.
import math
from typing import Any, Sequence

import nolds  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from fbm import FBM  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

# from main import ticker_symbol


def segment_data(data, num_segments):
    """
    Split a sequence into equal-length segments.
    """
    len_segment = len(data) // num_segments
    return [
        data[i : i + len_segment]
        for i in range(0, len(data), len_segment)
        if len(data[i : i + len_segment]) == len_segment
    ]


def calculate_hurst_for_segments(data, num_segments):
    """
    Calculate one Hurst exponent per segment.
    """
    segments = segment_data(data, num_segments)
    hurst_values = [nolds.hurst_rs(seg) for seg in segments]
    return hurst_values


def define_time_window(
    min_window: int, max_window: int, base: float = 10, interval: float = 0.25
):
    """
    Build logarithmically spaced window sizes for scaling analysis.

    Args:
        min_window: Minimum window length.
        max_window: Maximum window length.
        base: Logarithmic base for spacing.
        interval: Step size in log-space.

    Returns:
        List of integer window sizes.
    """
    window_sizes = list(
        map(
            lambda x: int(base**x),
            np.arange(math.log10(min_window), math.log10(max_window), interval),
        )
    )

    return window_sizes


def calculate_scaling_exponent(delta, x_t, q, ticker):
    """
    Compute partition values Fq and scaling exponents tau(q).

    Args:
        delta: Iterable of time windows.
        x_t: Time series DataFrame.
        q: Iterable of moment exponents.
        ticker: Column name in x_t to analyze.

    Returns:
        Tuple of:
        - Fq: DataFrame of partition values indexed by q and window size.
        - tau_q_list: Estimated scaling exponents.
    """
    # Fq[k][j] stores the partition value for q[k] and delta[j].
    Fq = [[0 for x in range(len(delta))] for y in range(len(q))]

    # print(f"Fq: {Fq}, type: {type(Fq)}") # --> list
    # print(f"x_t: {x_t}, type: {type(x_t)}") # --> Dataframe
    # print(f"q: {q}, type: {type(q)}") # --> numpy array

    # For each q and delta, aggregate absolute increments raised to q.
    for k in range(0, len(q)):
        if k % 30 == 0:  # Progress indicator every 30 q values.
            print("calculating q=" + str(k) + " out of " + str(len(q) - 1))
        for j in range(0, len(delta)):
            # print(f"j: {j}")
            for i in range(0, len(x_t) - 1):
                # print(f"i: {i}")
                if i < int((len(x_t) - 1) / delta[j]):
                    Fq[k][j] = (
                        Fq[k][j]
                        + abs(
                            x_t[ticker].iloc[i * delta[j] + delta[j]]
                            - x_t[ticker].iloc[i * delta[j]]
                        )
                        ** q[k]
                    )
                    # print(f"Fq: {Fq[k][j]}")

    Fq = pd.DataFrame(Fq)

    for i in range(0, len(q)):
        Fq.rename(index={Fq.index[i]: q[i]}, inplace=True)
    for i in range(len(delta) - 1, -1, -1):
        Fq.rename(columns={Fq.columns[i]: delta[i]}, inplace=True)

    print("Finished calculating the partition values Fq")

    # Regress log(Fq) against log(delta) to estimate tau(q).
    tau_q_list = []
    for i, row in Fq.iterrows():
        Fq_matrix = np.vstack([np.log10(row.values), np.ones(len(row))]).T
        tau_q, c = np.linalg.lstsq(Fq_matrix, np.log10(delta), rcond=-1)[0]
        tau_q_list.append(tau_q)

    return Fq, tau_q_list


def estimate_multifractal_spectrum(
    tau_q_list: Sequence[float],
    q: Sequence[float],
    start_of_list: int,
    end_of_list: int,
):
    """
    Estimate the multifractal spectrum f(alpha) from tau(q).

    Args:
        tau_q_list: Scaling exponents tau(q).
        q: Statistical moments.
        start_of_list: Start index used for fitting.
        end_of_list: End index used for fitting.

    Returns:
        Tuple of:
        - F_A: DataFrame with f(alpha) and alpha proxy values.
        - parameters_of_spectrum: (width, holder_exponent, asymmetry).
    """
    tau_q_estimated = np.polyfit(
        q[start_of_list:end_of_list], tau_q_list[start_of_list:end_of_list], 2
    )

    F_A = [0 for x in range(len(q) - 10)]
    p = [0 for x in range(len(q) - 10)]

    a = tau_q_estimated[0]
    b = tau_q_estimated[1]
    c = tau_q_estimated[2]

    for i in range(0, len(q) - 10):
        p[i] = 2 * a * q[i] + b
        F_A[i] = ((p[i] - b) / (2 * a)) * p[i] - (
            a * ((p[i] - b) / (2 * a)) ** 2 + b * ((p[i] - b) / (2 * a)) + c
        )

    F_A = pd.DataFrame(F_A)
    F_A.rename(columns={F_A.columns[0]: "f(a)"}, inplace=True)
    F_A["p"] = p

    print(
        "Using the range of q's from "
        + str(q[start_of_list])
        + " to "
        + str(q[end_of_list])
        + ":"
    )
    # tau_q_estimated = quadratic coefficients (a, b, c) for tau(q).
    print("The estimated parameters for tau(q) are: \n" + str(tau_q_estimated))

    # Parameters derived from the Legendre-transform quadratic form.
    # 1 / (4a): width of the f(alpha) spectrum.
    width_of_spectrum = 1 / (4 * a)
    # (-2b) / (4a): peak location alpha0 (most probable Holder exponent).
    holder_exponent = (-2 * b) / (4 * a)
    # (-4ac + b^2) / (4a): asymmetry-related term of f(alpha).
    asymmetry_of_spectrum = (-4 * a * c + b**2) / (4 * a)
    # Together these summarize the geometry of the estimated spectrum.
    print(
        "\nThus, the estimated parameters for f(a) are: \n width_of_spectrum: "
        + str(width_of_spectrum)
        + ", \n holder_exponent: "
        + str(holder_exponent)
        + ", \n asymmetry_of_spectrum: "
        + str(asymmetry_of_spectrum)
    )

    return F_A, (width_of_spectrum, holder_exponent, asymmetry_of_spectrum)


def calculate_lognormal_cascade(
    layers: int,
    v: float,
    ln_lambda: float,
    ln_sigma: float,
) -> float | list[Any]:
    """
    Generate a recursive log-normal multiplicative cascade.

    Args:
        layers: Remaining cascade depth.
        v: Current branch weight.
        ln_lambda: Mean parameter of the log-normal draw.
        ln_sigma: Standard deviation parameter of the log-normal draw.

    Returns:
        Nested list/scalar structure representing branch weights.
    """
    layers = layers - 1

    m0 = np.random.lognormal(ln_lambda, ln_sigma)
    m1 = np.random.lognormal(ln_lambda, ln_sigma)
    total_weight = m0 + m1
    m0 = m0 / total_weight
    m1 = m1 / total_weight

    M = [m0, m1]

    if layers >= 0:
        d: list[Any] = [0 for x in range(0, 2)]
        for i in range(0, 2):
            d[i] = calculate_lognormal_cascade(layers, (M[i] * v), ln_lambda, ln_sigma)

        return d

    return v


def calculate_trading_time(layers: int, lognormal_cascade: list):
    """
    Convert cascade weights into cumulative trading time.

    Args:
        layers: Cascade depth.
        lognormal_cascade: Flattened cascade weights.

    Returns:
        Trading-time process (normalized cumulative sum).
    """
    trading_time = 2**layers * np.cumsum(lognormal_cascade) / sum(lognormal_cascade)
    return trading_time


def calculate_magnitude_parameter(
    initial_value: float,
    eps: float,
    steps: float,
    number_of_path: int,
    real_std: float,
    layers: int,
    hurst_exponent: float,
):
    """
    Calibrate FBM path length to match observed return volatility.

    Args:
        initial_value: Initial guess for FBM length.
        eps: Convergence tolerance for volatility mismatch.
        steps: Update scale applied to the mismatch.
        number_of_path: Number of simulated FBM paths per iteration.
        real_std: Target standard deviation from historical returns.
        layers: Cascade depth.
        hurst_exponent: Estimated Hurst exponent.

    Returns:
        Calibrated magnitude parameter.
    """
    diff = np.inf
    magnitude_parameter = initial_value

    while abs(diff) > eps:
        std_list = []
        for nb in range(number_of_path):  # Keep loop quiet during calibration.
            new_fbm_class = FBM(
                n=10 * 2**layers + 1,
                hurst=hurst_exponent,
                length=magnitude_parameter,
                method="daviesharte",
            )
            new_fbm_simulation = new_fbm_class.fbm()
            std_list.append(np.std(new_fbm_simulation))
        diff = real_std - np.median(std_list)
        print("Diff: ", diff)
        if abs(diff) > eps:
            magnitude_parameter += diff * steps
            print("new magnitude_parameter:", magnitude_parameter)

    return magnitude_parameter


def calculate_mmar_returns(
    S0: float,
    number_of_path: int,
    layers: int,
    hurst_exponent: float,
    trading_time: list,
    magnitude_parameter: float,
    time_window_base: float = 10,
):
    """
    Simulate MMAR returns and convert them to price paths.

    Args:
        S0: Initial asset price.
        number_of_path: Number of simulated paths.
        layers: Cascade depth.
        hurst_exponent: Estimated Hurst exponent.
        trading_time: Time-change process used to sample FBM.
        magnitude_parameter: FBM length parameter.
        time_window_base: Unused legacy parameter kept for compatibility.

    Returns:
        Tuple of (mmar_returns, mmar_prices).
    """
    mmar_returns = []
    mmar_prices = []

    for nb in tqdm(range(number_of_path)):
        new_fbm_class = FBM(
            n=10 * 2**layers + 1,
            hurst=hurst_exponent,
            length=magnitude_parameter,
            method="daviesharte",
        )
        new_fbm_simulation = new_fbm_class.fbm()
        new_fbm_simulation = new_fbm_simulation[1:]

        # Sample FBM along the MMAR trading-time clock.
        simulated_xt_array = [0 for x in range(0, len(trading_time))]
        for i in range(0, len(trading_time)):
            simulated_xt_array[i] = new_fbm_simulation[int(trading_time[i] * 10)]
        mmar_returns.append(simulated_xt_array)

        # Convert simulated log-returns to price levels.
        simulated_prices_array = S0 * np.exp(simulated_xt_array)
        mmar_prices.append(simulated_prices_array)

    return mmar_returns, mmar_prices


def option_pricer(paths, strike, r, T, option_type):
    """
    Price a European option from simulated asset paths.

    Parameters:
    - paths: An array of simulated asset paths.
    - strike: Strike price of the option.
    - r: Risk-free rate.
    - T: Time to maturity in years.
    - option_type: call or put.

    Returns:
    - Discounted Monte Carlo option price.
    """

    if isinstance(paths, list):
        paths = np.array(paths)

    # Terminal asset prices at maturity.
    S_T = paths[:, -1]

    # Pathwise terminal payoff.
    if option_type == "call":
        payoffs = np.maximum(S_T - strike, 0)
    elif option_type == "put":
        payoffs = np.maximum(strike - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Discount the average payoff to present value.
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price


def option_pricer_half_time(paths, strike, r, T, option_type):
    """
    Price a European option at an intermediate day index.

    Parameters:
    - paths: An array of simulated asset paths.
    - strike: Strike price of the option.
    - r: Risk-free rate.
    - T: Time to maturity in years.
    - option_type: call or put.

    Returns:
    - Discounted Monte Carlo option price.
    """

    if isinstance(paths, list):
        paths = np.array(paths)

    # Convert year fraction to an integer day index.
    half_time_dte = round(T * 365)
    # print(f"Half Time DTE: {half_time_dte}")

    # Asset prices at the selected intermediate index.
    S_T = paths[:, half_time_dte]

    # Pathwise payoff at the selected time point.
    if option_type == "call":
        payoffs = np.maximum(S_T - strike, 0)
    elif option_type == "put":
        payoffs = np.maximum(strike - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Discount the average payoff to present value.
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price


def generate_fbm_path(n, hurst, dt=1, s0=1):
    """Generate a fractional Brownian motion path with n steps and a given Hurst exponent."""
    # Draw Gaussian innovations.
    dW = np.random.randn(n)

    # Scale increments according to dt and Hurst exponent.
    increments = dW * (dt ** (hurst))

    # Accumulate increments into a path.
    fbm_path = np.cumsum(increments)

    # Shift so the first value equals s0.
    fbm_path = fbm_path - fbm_path[0] + s0

    return fbm_path


def generate_multiple_paths(num_paths, n, hurst, dt=1, s0=1):
    """Generate multiple non-negative fBM-like price paths."""
    paths = []
    # Simulate each path independently.
    for _ in range(num_paths):
        prices = generate_fbm_path(n, hurst, dt, s0)
        # Enforce non-negative prices.
        prices = np.where(prices > 0, prices, 0)
        paths.append(prices)

    return paths


def price_options_for_strikes(paths, center, step, num_strikes, r, T):
    """Price call/put options on a strike grid and return a table."""

    # option_prices = {}

    strikes = []
    dte = []
    option_prices = []
    side = []

    # Put side: strikes below the center.
    for i in range(1, num_strikes + 1):
        strike = center - i * step
        # option_prices[strike] = option_pricer(paths, strike, r, T, option_type='put')
        option_price = option_pricer(paths, strike, r, T, option_type="put")
        print(option_price)
        if option_price > 0:
            strikes.append(strike)
            dte.append(T * 365)
            option_prices.append(option_price)
            side.append("Put")

    # Call side: strikes at/above the center.
    for i in range(num_strikes + 1):
        strike = center + i * step
        # option_prices[strike] = option_pricer(paths, strike, r, T, option_type='call')
        option_price = option_pricer(paths, strike, r, T, option_type="call")
        if option_price > 0:
            strikes.append(strike)
            dte.append(T * 365)
            option_prices.append(option_price)
            side.append("Call")

    option_prices_df = pd.DataFrame(
        {"Strike": strikes, "DTE": dte, "Option Prices": option_prices, "Side": side}
    )

    return option_prices_df
