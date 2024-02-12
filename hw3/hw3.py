import numpy as np
from scipy.stats import norm, t, chi2
from math import sqrt
import ci
import utils_93

# Q1
q1_dat = open("datafiles/q1.dat", "r")
q1_mu = 11.5
q1_np = np.array([float(x) for x in q1_dat.read().split(",")]) + q1_mu
q1_xbar = q1_np.mean()
q1_s = np.std(q1_np, ddof=1)
q1_n = len(q1_np)
print("Sample mean:", q1_xbar, "| Sample std:", q1_s, "| Num samples:", q1_n)

t0 = (q1_xbar - q1_mu) / (q1_s / sqrt(q1_n))

p = 2 * (1 - t.cdf(abs(t0), q1_n - 1))

print("P-value:", p)

q1_d = abs(11.4 - q1_mu) / q1_s
print("d used in OC chart", q1_d)
print("Power: 1")

q1_d = abs(11.45 - q1_mu) / q1_s
print("d used in OC chart", q1_d)
print("Num samples:", 7)

q1_ci = ci.mean_ci(q1_xbar, q1_s, q1_n, bound="centered", known_variance=False)

print("95% CI:", q1_ci)

# utils_93.normal_probability_plot(q1_np)

# Q2
q2_mu = 1000
q2_xbar = 1050
q2_std = 50
q2_n = 25

z0 = (q2_xbar - q2_mu) / (q2_std / sqrt(q2_n))
p = norm.sf(z0)

print("P-value:", p)

q2_ci = ci.mean_ci(q2_xbar, q2_std, q2_n)

print("95% CI:", q2_ci)

# You would expect it to be 1050

# Q3

# H0: sigma = 0.01, H1: sigma > 0.01, assuming normal dist
q3_n = 15
q3_s = 0.008
q3_std = 0.01
q3_chi2 = (q3_n - 1) * q3_s**2 / q3_std**2
p = chi2.sf(q3_chi2, q3_n - 1)
print("P-value:", p)
# No this does not provide strong evidence

q3_lambda = 1.5
print("Probability of correct diagnosis is roughly 50% from OC chart")

print("It looks like you need either 75-100 or 100+ for a power of 80%")

# Q4
q4_x1 = 70
q4_n1 = 100
q4_x2 = 75
q4_n2 = 100
q4_p1 = q4_x1 / q4_n1
q4_p2 = q4_x2 / q4_n2
q4_p = (q4_x1 + q4_x2) / (q4_n1 + q4_n2)

z0 = (q4_p1 - q4_p2) / sqrt(q4_p * (1 - q4_p) * (1 / q4_n1 + 1 / q4_n2))

p = 2 * norm.sf(abs(z0))

print("P-value:", p)
# Nope, cannot conclude they're different
