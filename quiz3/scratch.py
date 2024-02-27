from math import sqrt
from scipy.stats import norm, t

# Q1
n1 = 1500 * 2
n2 = 1200 * 2
x1 = 387 * 2
x2 = 310 * 2

p1 = x1 / n1
p2 = x2 / n2

phat = (x1 + x2) / (n1 + n2)

se_coeff = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

z_alpha = norm.ppf(0.95)

print(
    p1, p2, phat, (p1 - p2 - z_alpha * se_coeff, p1 - p2 + z_alpha * se_coeff)
)

# Q2
n = 180
beta_0 = 40.3
beta_1 = 0.38
cd = 0.568
sigma_squared = 9
xbar = 66
sx = 6

x0 = 68
y_hat = beta_0 + beta_1 * x0

sxx = sx**2 * (n - 1)

se = sqrt(sigma_squared * (1 / n + (x0 - xbar) ** 2 / sxx))

z = (x0 - y_hat) / se

p = norm.sf(z)

print(y_hat, sxx, se, z, p)

# Q3

t0 = 6.3 * 10 ** -2 / sqrt(2.1 * 10 ** -2)
t_alpha = t.ppf(0.995, 10)
print(t0, t_alpha, 12 / 15)
