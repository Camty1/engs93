import numpy as np
from scipy.stats import norm, t, chi2
from math import sqrt, floor
import matplotlib.pyplot as plt
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
print("Num samples:", 5)

q1_ci = ci.mean_ci(q1_xbar, q1_s, q1_n, bound="centered", known_variance=False)

print("95% CI:", q1_ci)

utils_93.normal_probability_plot(q1_np)

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

# Q5
q5_xbar_1 = 290
q5_xbar_2 = 321

q5_s1 = 12
q5_s2 = 22

q5_n1 = 10
q5_n2 = 16

t0 = (q5_xbar_1 - q5_xbar_2) / sqrt(q5_s1**2 / q5_n1 + q5_s2**2 / q5_n2)
nu = floor(
    (q5_s1**2 / q5_n1 + q5_s2**2 / q5_n2) ** 2
    / (
        (q5_s1**2 / q5_n1) ** 2 / (q5_n1 - 1)
        + (q5_s2**2 / q5_n2) ** 2 / (q5_n2 - 1)
    )
)

p = t.sf(abs(t0), nu)
print("P-value:", p)
# Can reject H0

mean_difference_ci = ci.mean_ci(
    q5_xbar_1 - q5_xbar_2,
    sqrt(q5_s1**2 / q5_n1 + q5_s2**2 / q5_n2),
    nu + 1,
    bound="lower",
    known_variance=False,
)

print("95% CI for difference:", mean_difference_ci)

# Q6
q6_xbar = 5.91
q6_ybar = 21.47
q6_s_x = 2.88
q6_s_y = 6.75
q6_rho = 0.9
q6_n = 10

beta_1 = q6_rho * q6_s_y / q6_s_x
beta_0 = q6_ybar - beta_1 * q6_xbar

print("Regression Parameters:", (beta_0, beta_1))

sigma_squared = (
    1
    / (q6_n - 2)
    * (
        q6_n * q6_ybar**2
        + (q6_n - 1) * q6_s_y**2
        - 2
        * (
            beta_0 * q6_n * q6_ybar
            + beta_1
            * (
                q6_n * q6_xbar * q6_ybar
                + (q6_n - 1) * q6_rho * q6_s_x * q6_s_y
            )
        )
        + q6_n * beta_0**2
        + 2 * q6_n * beta_0 * beta_1 * q6_xbar
        + beta_1**2 * (q6_n * q6_xbar**2 + (q6_n - 1) * q6_s_x**2)
    )
)

se_0 = sqrt(
    sigma_squared * (1 / q6_n + q6_xbar**2 / ((q6_n - 1) * q6_s_x**2))
)
se_1 = sqrt(sigma_squared / ((q6_n - 1) * q6_s_x**2))

print("Standard error of beta 0 and beta 1:", (se_0, se_1))

x0 = 5.5
t_val = t.ppf(0.975, q6_n - 2)

mu_hat = beta_0 + beta_1 * x0

lower_ci = mu_hat - t_val * sqrt(
    sigma_squared
    * (1 / q6_n + (x0 - q6_xbar) ** 2 / ((q6_n - 1) * q6_s_x**2))
)

upper_ci = mu_hat + t_val * sqrt(
    sigma_squared
    * (1 / q6_n + (x0 - q6_xbar) ** 2 / ((q6_n - 1) * q6_s_x**2))
)

print("95% CI at x = 5.5:", (lower_ci, upper_ci))

# Q7
# Variance is sigma^2/Sxx, error variance is sigma^2,
# R^2 = 1 - SSE / Syy, Eqn 11-31
ybar = 3.9
s_y = sqrt(2.7)
xbar = 10.6
s_x = sqrt(34.2)
n = 100
beta_1 = (0.23 + 0.28) / 2
beta_0 = ybar - beta_1 * xbar
t_val = t.ppf(0.975, n - 2)
slope_variance = ((0.28 - beta_1) / t_val)**2
print("Slope variance:", slope_variance)

s_xx = (n - 1) * s_x
error_variance = slope_variance * s_xx
print("Error variance:", error_variance)

sse = error_variance * (n - 2)
s_yy = (n - 1) * s_y
R_squared = 1 - sse / s_yy
print("Coefficient of determination:", R_squared)

x0 = 15
t_val = t.ppf(0.9, n - 2)
std = sqrt(error_variance * (1 / n + (x0 - xbar) ** 2 / ((n - 1) * s_x**2)))
fit_val = beta_0 + beta_1 * x0
confidence_interval = (fit_val - t_val * std, fit_val + t_val * std)
print("80% CI:", confidence_interval)


# Q8
q8_np = np.genfromtxt("datafiles/streams.csv", delimiter=",")[1:, :]
x = q8_np[:, 1]
y = q8_np[:, 0]
n = len(q8_np)


x_aug = np.concatenate(
    (np.ones((n, 1), dtype=float), np.reshape(x, (-1, 1))), axis=1
)

# X'X b = X'y
beta = np.linalg.solve(
    np.matmul(np.transpose(x_aug), x_aug),
    np.matmul(np.transpose(x_aug), np.reshape(y, (-1, 1))),
)
print("Beta 0:", float(beta[0]), "Beta 1:", float(beta[1]))
y_line = np.matmul(x_aug, beta)

plt.scatter(x, y, label="Measurements")
plt.plot(x, y_line, color="tab:orange", label="Trendline")
plt.title("Chloride Concentration vs Roadway Area")
plt.xlabel("Roadway Area (%)")
plt.ylabel("Chloride Concentration (mg/L)")
plt.legend(loc="upper left")
plt.show()

s_xy = np.sum((x - x.mean()) * (y - y.mean()))
s_yy = np.sum((y - y.mean()) * (y - y.mean()))
sse = s_yy - beta[1] * s_xy
sigma_squared = sse / (n - 2)
print("Sigma squared:", float(sigma_squared))

idx = int(np.argwhere(x == 0.47)[0])
y_fit = np.matmul(x_aug[idx, :], beta)
residual = float(y[idx] - y_fit)

print("y_hat(0.47):", y_fit)
print("Residual at x = 0.47:", residual)

# Q9
q9_np = np.genfromtxt("datafiles/fish.csv", delimiter=",")[1:, [4, 6]]
length = q9_np[:, 0]
ddt = q9_np[:, 1]
n = len(ddt)
ln_ddt = np.log(ddt)

utils_93.normal_probability_plot(ddt)
utils_93.normal_probability_plot(ln_ddt)
x = np.concatenate([np.ones((n, 1)), length.reshape((-1, 1))], axis=1)
beta = np.linalg.solve(
    np.matmul(np.transpose(x), x),
    np.matmul(np.transpose(x), ln_ddt.reshape((-1, 1))),
)
