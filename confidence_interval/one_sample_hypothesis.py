import numpy as np
import ci
from scipy.stats import norm, t, chi2
from typing import Tuple
from math import sqrt, ceil
import utils_93


def mean_test(
    sample_mean: float,
    pop_mean: float,
    std: float,
    num_samples: int,
    alpha: float = 0.05,
    direction: str = "centered",
    known_variance: bool = True,
) -> Tuple[float, float, bool]:
    valid_directions = ["above", "centered", "below"]
    assert direction in valid_directions

    if known_variance:
        z_0 = (sample_mean - pop_mean) / std * sqrt(num_samples)

        if direction == "below":
            z = norm.ppf(alpha)
            p = norm.cdf(z_0)

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "upper",
                known_variance,
            )

            return (z_0, confidence_interval, p, z_0 < z)

        elif direction == "centered":
            z = abs(norm.ppf(alpha / 2))
            p = 2 * (1 - norm.cdf(abs(z_0)))

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "centered",
                known_variance,
            )

            return (z_0, confidence_interval, p, abs(z_0) > z)

        else:
            z = abs(norm.ppf(alpha))
            p = 1 - norm.cdf(z_0)

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "lower",
                known_variance,
            )

            return (z_0, confidence_interval, p, z_0 > z)
    else:
        t_0 = (sample_mean - pop_mean) / std * sqrt(num_samples)

        if direction == "lower":
            t_val = t.ppf(alpha, num_samples - 1)
            p = t.cdf(t_0, num_samples - 1)

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "upper",
                known_variance,
            )

            return (t_0, confidence_interval, p, t_0 < t_val)

        elif direction == "centered":
            t_val = abs(t.ppf(alpha / 2, num_samples - 1))
            p = 2 * (1 - t.cdf(abs(t_0), num_samples - 1))

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "centered",
                known_variance,
            )

            return (t_0, confidence_interval, p, abs(t_0) > t_val)

        else:
            t_val = abs(t.ppf(alpha, num_samples - 1))
            p = 1 - t.cdf(t_0, num_samples - 1)

            confidence_interval = ci.mean_ci(
                sample_mean,
                std,
                num_samples,
                1 - alpha,
                "lower",
                known_variance,
            )

            return (t_0, confidence_interval, p, t_0 > t_val)


def calc_beta(
    std: float,
    delta: float,
    num_samples: int,
    alpha: float = 0.05,
    bound: str = "centered",
    known_variance: bool = True,
) -> float:
    valid_bounds = ["lower", "centered", "upper"]
    assert bound in valid_bounds

    if known_variance:
        if bound == "centered":
            z_alpha = abs(norm.ppf(alpha / 2))

            return norm.cdf(
                z_alpha - delta * sqrt(num_samples) / std
            ) - norm.cdf(-z_alpha - delta * sqrt(num_samples) / std)
        else:
            delta = abs(delta)
            z_alpha = abs(norm.ppf(alpha))

            return norm.cdf(
                z_alpha - delta * sqrt(num_samples) / std
            ) - norm.cdf(-z_alpha - delta * sqrt(num_samples) / std)
    else:
        if bound == "centered":
            t_alpha = abs(t.ppf(alpha / 2, num_samples - 1))

            return t.cdf(
                t_alpha - delta * sqrt(num_samples) / std, num_samples - 1
            ) - t.cdf(
                -t_alpha - delta * sqrt(num_samples) / std, num_samples - 1
            )

        else:
            delta = abs(delta)
            t_alpha = abs(t.ppf(alpha, num_samples - 1))

            return t.cdf(
                t_alpha - delta * sqrt(num_samples) / std, num_samples - 1
            ) - t.cdf(
                -t_alpha - delta * sqrt(num_samples) / std, num_samples - 1
            )


def mean_sample_size(
    std: float,
    delta: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    bound: str = "centered",
) -> int:
    valid_bounds = ["lower", "centered", "upper"]
    assert bound in valid_bounds
    z_beta = norm.ppf(beta)
    if bound == "upper" or bound == "lower":
        z_alpha = norm.ppf(alpha)
    else:
        z_alpha = norm.ppf(alpha / 2)
    return ceil(((z_alpha + z_beta) * std / delta) ** 2)


def variance_test(
    sample_std: float,
    pop_std: float,
    num_samples: int,
    alpha: float = 0.05,
    direction: str = "above",
) -> Tuple[float, bool]:
    valid_directions = ["above", "centered", "below"]
    assert direction in valid_directions
    chi_0 = (num_samples - 1) * sample_std**2 / pop_std**2

    if direction == "above":
        confidence_interval = ci.std_ci(
            sample_std, num_samples, 1 - alpha, bound="lower"
        )

        p = 1 - chi2.cdf(chi_0, num_samples - 1)

        return (
            chi_0,
            confidence_interval,
            p,
            pop_std < confidence_interval[0],
        )

    elif direction == "centered":
        confidence_interval = ci.std_ci(sample_std, num_samples, 1 - alpha)

        p = chi2.sf(chi_0, num_samples - 1) * 2

        return (
            chi_0,
            confidence_interval,
            p,
            confidence_interval[0] > pop_std
            or pop_std > confidence_interval[1],
        )

    else:
        confidence_interval = ci.std_ci(
            sample_std, num_samples, 1 - alpha, bound="upper"
        )

        p = chi2.cdf(chi_0, num_samples - 1)

        return (
            chi_0,
            confidence_interval,
            p,
            pop_std > confidence_interval[1],
        )


def proportion_test(
    sample_proportion: float,
    pop_proportion: float,
    num_samples: int,
    alpha: float = 0.05,
    direction: str = "centered",
) -> Tuple[float, Tuple[float, ...], float, bool]:
    valid_directions = ["above", "centered", "below"]
    assert direction in valid_directions
    z_0 = (sample_proportion - pop_proportion) / sqrt(
        pop_proportion * (1 - pop_proportion) / num_samples
    )
    if direction == "above":
        confidence_interval = ci.proportion_ci(
            sample_proportion, num_samples, 1 - alpha, "lower"
        )

        p = norm.sf(z_0)

        return (
            z_0,
            confidence_interval,
            p,
            pop_proportion < confidence_interval[0],
        )

    elif direction == "centered":
        confidence_interval = ci.proportion_ci(
            sample_proportion, num_samples, 1 - alpha, "centered"
        )

        p = 2 * norm.sf(abs(z_0))

        return (
            z_0,
            confidence_interval,
            p,
            pop_proportion < confidence_interval[0]
            or pop_proportion > confidence_interval[1],
        )

    else:
        confidence_interval = ci.proportion_ci(
            sample_proportion, num_samples, 1 - alpha, "upper"
        )

        p = norm.cdf(z_0)

        return (
            z_0,
            confidence_interval,
            p,
            pop_proportion > confidence_interval[1],
        )


print(mean_test(1.15, 1.2, 0.4, 64, direction="centered", known_variance=False))

print(t.ppf(0.025, 63), t.ppf(0.975, 63))

print(ci.std_ci(13.41, 101, bound="lower"))

print(chi2.ppf(0.05, 100))

print(ci.mean_ci(13.41, sqrt(10**2/(2*101)), 101, bound="lower"))

print(norm.ppf(0.05))
