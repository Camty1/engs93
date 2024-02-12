import numpy as np
from scipy.stats import norm, t, chi2
from typing import Tuple
from math import sqrt, ceil


def mean_ci(
    sample_mean: float,
    std: float,
    num_samples: int,
    interval: float = 0.95,
    bound: str = "centered",
    known_variance: bool = True,
) -> Tuple[float, ...]:
    valid_bounds = ["upper", "centered", "lower"]
    assert bound in valid_bounds

    if known_variance:
        if bound == "upper":
            z = norm.ppf(interval)
            upper_bound = sample_mean + std * z / sqrt(num_samples)

            return (-np.inf, upper_bound)

        elif bound == "centered":
            interval = 1 - ((1 - interval) / 2)
            z = norm.ppf(interval)
            upper_bound = sample_mean + std * z / sqrt(num_samples)
            lower_bound = sample_mean - std * z / sqrt(num_samples)

            return (lower_bound, upper_bound)

        else:
            z = norm.ppf(interval)
            lower_bound = sample_mean - std * z / sqrt(num_samples)

            return (lower_bound, np.inf)

    else:
        if bound == "upper":
            t_val = t.ppf(interval, num_samples - 1)
            upper_bound = sample_mean + std * t_val / sqrt(num_samples)

            return (-np.inf, upper_bound)

        elif bound == "centered":
            interval = 1 - ((1 - interval) / 2)
            t_val = t.ppf(interval, num_samples - 1)
            upper_bound = sample_mean + std * t_val / sqrt(num_samples)
            lower_bound = sample_mean - std * t_val / sqrt(num_samples)

            return (lower_bound, upper_bound)

        else:
            t_val = t.ppf(interval, num_samples - 1)
            lower_bound = sample_mean - std * t_val / sqrt(num_samples)

            return (lower_bound, np.inf)


def proportion_ci(
    proportion: float,
    num_samples: int,
    interval: float = 0.95,
    bound: str = "centered",
) -> Tuple[float, ...]:
    valid_bounds = ["upper", "centered", "lower"]
    assert bound in valid_bounds

    if bound == "upper":
        z = norm.ppf(interval)
        upper_bound = proportion + z * sqrt(
            proportion * (1 - proportion) / num_samples
        )

        return (-np.inf, upper_bound)

    elif bound == "centered":
        interval = 1 - ((1 - interval) / 2)
        z = norm.ppf(interval)
        upper_bound = proportion + z * sqrt(
            proportion * (1 - proportion) / num_samples
        )
        lower_bound = proportion - z * sqrt(
            proportion * (1 - proportion) / num_samples
        )

        return (lower_bound, upper_bound)

    else:
        z = norm.ppf(interval)
        lower_bound = proportion - z * sqrt(
            proportion * (1 - proportion) / num_samples
        )

        return (lower_bound, np.inf)


def num_samples_given_width(
    std: float, width: float, interval: float = 0.95
) -> int:
    interval = 1 - ((1 - interval) / 2)
    z = norm.ppf(interval)

    return ceil((2 * std * z / width) ** 2)


def num_samples_given_error(
    std: float, error: float, interval: float = 0.95
) -> int:
    interval = 1 - ((1 - interval) / 2)
    z = norm.ppf(interval)

    return ceil((std * z / error) ** 2)


def proportion_num_samples(
    error: float, proportion: float = 0.5, interval: float = 0.95
) -> int:
    interval = 1 - (1 - interval) / 2
    z = norm.ppf(interval)

    return (z / error) ** 2 * proportion * (1 - proportion)


def std_ci(
    std: float,
    num_samples: int,
    interval: float = 0.95,
    bound: str = "centered",
) -> Tuple[float, ...]:
    valid_bounds = ["upper", "centered", "lower"]
    assert bound in valid_bounds

    if bound == "upper":
        interval = 1 - interval
        upper_chi_squared = chi2.ppf(interval, num_samples - 1)
        upper_bound = sqrt((num_samples - 1) * std**2 / upper_chi_squared)

        return (-np.inf, upper_bound)

    elif bound == "centered":
        interval = (1 - interval) / 2
        upper_chi_squared = chi2.ppf(interval, num_samples - 1)

        interval = 1 - interval
        lower_chi_squared = chi2.ppf(interval, num_samples - 1)

        lower_bound = sqrt((num_samples - 1) * std**2 / lower_chi_squared)
        upper_bound = sqrt((num_samples - 1) * std**2 / upper_chi_squared)

        return (lower_bound, upper_bound)

    else:
        lower_chi_squared = chi2.ppf(interval, num_samples - 1)
        lower_bound = sqrt((num_samples - 1) * std**2 / lower_chi_squared)

        return (lower_bound, np.inf)
