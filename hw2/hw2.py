# ENGS 93 HW 2
# Cameron Wolfe, 1/31/24

# Imports
import numpy as np
from scipy.stats import norm, t, chi2
import matplotlib.pyplot as plt
from math import sqrt

# Question 1
q_1_file = open("datafiles/q1.dat", "r")
q_1_dat = [int(x) for x in q_1_file.read().split(",")]
num_data_points = len(q_1_dat)

# 1a
print("Question 1a")
frequency_dict = {}
for val in q_1_dat:
    if val in frequency_dict:
        frequency_dict[val] += 1
    else:
        frequency_dict[val] = 1
sorted_items = sorted(frequency_dict.items())
sorted_frequencies = {}
sorted_relative_frequencies = {}
for key, value in sorted_items:
    sorted_frequencies[key] = value
    sorted_relative_frequencies[key] = value / num_data_points
print("Frequencies:\n", sorted_frequencies)
print("Relative Frequencies:\n", sorted_relative_frequencies)

# 1b
print("\nQuestion 1b")
at_most_5 = 0
less_than_5 = 0
more_than_5 = 0
for key, value in sorted_items:
    if key < 5:
        at_most_5 += value
        less_than_5 += value
    elif key == 5:
        at_most_5 += value
    else:
        more_than_5 += value
print(
    "Proportion of batches with at most 5 nonconforming transducers:\n",
    at_most_5 / num_data_points,
)
print(
    "Proportion of batches with less than 5 nonconforming transducers:\n",
    less_than_5 / num_data_points,
)
print(
    "Proportion of batches with more than 5 nonconforming transducers:\n",
    more_than_5 / num_data_points,
)

# Question 1c
print("\nQuestion 1c")
q_1_np = np.array(q_1_dat, dtype=int)
bins = np.array(list(range(max(q_1_dat) + 2))) - 0.5
plt.relfreq(q_1_np, bins=bins)
plt.title("Histogram of Nonconforming Transducers")
plt.xlabel("Count of Nonconforming Transducers in Batch")
plt.ylabel("Frequency")
plt.show()

# Question 1d
print("\nQuestion 1d")
print(
    "This appears to be a binomial distribution centered around 2 or 3 transducers."
)

# Question 2
q_2_file = open("datafiles/q2.dat")
q_2_dat = [int(x) for x in q_2_file.read().split(",")]
q_2_np = np.array(q_2_dat)

# Question 2a
print("\nQuestion 2a")
sample_mean = q_2_np.mean()
sample_std = np.std(q_2_np, ddof=1)
sample_var = sample_std**2
print(
    "Sample mean:",
    sample_mean,
    "\nSample variance:",
    sample_var,
    "\nSample std:",
    sample_std,
)

# Question 2b
print("\nQuestion 2b")


def find_median(data_arr):
    sorted_arr = sorted(data_arr)
    while len(sorted_arr) > 2:
        sorted_arr = sorted_arr[1:-1]

    if len(sorted_arr) == 2:
        return (sorted_arr[0] + sorted_arr[1]) / 2.0
    else:
        return sorted_arr[0]


print("Median:", find_median(q_2_dat))
print(
    "The largest temperature measurement could increase indefinitly and not increase the median"
)

# Question 2c
print("\nQuestion 2c")
plt.boxplot(q_2_np, vert=False, labels=[""])
plt.title("Box Plot of Temperature Data")
plt.xlabel("Temperature in Fahrenheit")
plt.show()

# Question #3
q_3_file = open("datafiles/q3.dat", "r")
q_3_dat = [int(x) for x in q_3_file.read().split(",")]
q_3_np = np.array(q_3_dat)

# Question 3a
print("\nQuestion 3a")
sample_mean = q_3_np.mean()
print("Sample mean:", sample_mean)

# Question 3b
print("\nQuestion 3b")
sample_std = np.std(q_3_np, ddof=1)
print("Standard error:", sample_std)
print(
    "Standard error is an estimation of the standard deviation of the population."
)

# Question 3c
print("\nQuestion 3c")
print("Median:", find_median(q_3_dat))

# Question 3d
print("\nQuestion 3d")
print("Proportion greater than 430:", len(q_3_np[q_3_np > 430]) / len(q_3_np))

# Question 4
q_4_file = open("datafiles/q4.dat", "r")
q_4_dat = [int(x) for x in q_4_file.read().split(",")]
q_4_np = np.array(q_4_dat)

# Question 4a
print("\nQuestion 4a")
sample_mean = q_4_np.mean()
print("Sample mean:", sample_mean)

# Question 4b
print("\nQuestion 4b")
sample_std = np.std(q_4_np, ddof=1)
print("Standard error:", sample_std)

# Question 4c
print("\nQuestion 4c")
print(
    "By reducing the sample size, you reduce the accuracy of point estimates.  The sample mean is less likely to be close to the population mean, and the standard error has the potential to be either larger or smaller based on the samples collected."
)

# Question 5
q_5_file = open("datafiles/q5.dat", "r")
q_5_dat = sorted([float(x) for x in q_5_file.read().split(",")])
q_5_np = np.array(q_5_dat)
n = len(q_5_dat)
ppf = np.zeros(n, dtype=float)
z = np.zeros(n, dtype=float)
for i in range(n):
    ppf[i] = (i + 0.5) / n
    z[i] = norm.ppf(ppf[i])

plt.plot(q_5_np[[0, -1]], z[[0, -1]])
plt.scatter(q_5_np, z)
plt.title("Probability Plot of Octane Ratings")
plt.xlabel("Octane Rating")
plt.ylabel("Z")
plt.show()

# Question 6

# Question 6a
print("\nQuestion 6a")
pop_mean = 34
pop_std = 15
n = 100
sample_std = 15 / sqrt(n)
print("Mean:", pop_mean)
print("Standard deviation:", sample_std)

# Question 6b
print("\nQuestion 6b")
print(
    "The distribution of the sample mean of smartphone user's aage should also be a normal distribution."
)

# Question 6c
print("\nQuestion 6c")
z = (30 - pop_mean) / sample_std
prob = 1 - norm.cdf(z)
print("Probability that sample mean is greater than 30:", prob)

# Question 6d
print("\nQuestion 6d")
z = norm.ppf(0.95)
sample_mean_95 = z * sample_std + pop_mean
print("95th percentile for sample mean age:", sample_mean_95)

# Question 7
q_7_dat = [76, 78, 80, 82, 84]
n = len(q_7_dat)
q_7_np = np.array(q_7_dat)

# Question 7a
print("\nQuestion 7a")
sample_std = np.std(q_7_np, ddof=1)
sample_mean = q_7_np.mean()
t_val = t.ppf(0.975, n - 1)
mean_ci = [
    sample_mean - t_val * sample_std / sqrt(n),
    sample_mean + t_val * sample_std / sqrt(n),
]
print("Two-sided 95% CI for population mean:", mean_ci)

# Question 7b
print("\nQuestion 7b")
upper_chi_val = chi2.ppf(0.025, n - 1)
lower_chi_val = chi2.ppf(0.975, n - 1)
std_ci = [
    sqrt((n - 1) * sample_std**2 / lower_chi_val),
    sqrt((n - 1) * sample_std**2 / upper_chi_val),
]
print("Two-sided 95% CI for population standard deviation:", std_ci)

# Question 7c
print("\nQuestion 7c")
pop_std = sample_std
z_val = norm.ppf(0.975)
mean_ci = [
    sample_mean - z_val * pop_std / sqrt(n),
    sample_mean + z_val * pop_std / sqrt(n),
]
print(
    "Two-sided 95% CI for population mean given standard deviation:",
    mean_ci,
)

# Question 7d
print("\nQuestion 7d")
print(
    "The confidence interval in C is tighter than in A because there is some uncertainty in the sample standard deviation from the data, meaning that in order to account for this, a wider range was needed."
)

# Question 7e
print("\nQuestion 7e")
n_needed = (z_val * pop_std / 1) ** 2
print("Sample size needed for width of 95% CI to be 2:", n_needed)

# Question 8
print("\nQuestion 8")
sample_std = 5
z_val = norm.ppf(0.95)
n_needed = (z_val * sample_std / 2) ** 2
print("Number of trees needed for 90% CI to be +- 2:", n_needed)

# Question 9
print("\nQuestion 9")
mean_ci = [34.64, 35.37]
n = 225
t_val = t.ppf(0.975, n - 1)
sample_std = (mean_ci[1] - (mean_ci[0] + mean_ci[1]) / 2) * sqrt(n) / t_val
print("Sample standard deviation:", sample_std)

# Question 10

# Question 10a
print("\nQuestion 10a")
print(
    "Null hypothesis: The temperature being below the threshold is due to random variation in the measurements"
)
print(
    "Alternative hypothesis: The temperature is below the threshold because of actions taken by the power plant"
)

# Question 10b
print("\nQuestion 10b")
print("A type 1 error would be if the temperature just happened to be below the threshold due the variability, whereas the true mean temperature is above the threshold")


# Question 10c
print("\nQuestion 10c")
print("A type 2 error would be if the temperature were above the threshold because of random chance, whereas the cooling system is actually working and keeps the true mean temperature below the threshold")
