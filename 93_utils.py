import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def normal_probability_plot(data_array: np.ndarray) -> None:
    sorted_array = np.sort(data_array.flatten())

    n = len(sorted_array)

    z = np.zeros(n)

    for i in range(n):
        z[i] = norm.ppf((i + 0.5) / n)

    plt.scatter(sorted_array, z)
    plt.title("Normal Probability Plot")
    plt.xlabel("Values")
    plt.ylabel("Z value")
    plt.show()


rng = np.random.default_rng()

gaussian = rng.standard_normal(50) * 2 + 10

normal_probability_plot(gaussian)
