from scipy.stats import chi2


def calculate_p_value(sample_variance, hypothesized_variance, sample_size):
    # Calculate degrees of freedom
    df = sample_size - 1

    # Calculate the chi-squared test statistic
    chi_squared_stat = (df * sample_variance) / hypothesized_variance

    # Calculate the p-value for the two-tailed test
    # sf() gives the survival function (1-CDF)
    p_value = chi2.sf(chi_squared_stat, df) * 2

    return chi_squared_stat, p_value


# Given values
sample_variance = 23.04
hypothesized_variance = 18
sample_size = 10

# Calculate the chi-squared statistic and p-value
chi_squared_stat, p_value = calculate_p_value(
    sample_variance, hypothesized_variance, sample_size)

print(f"Chi-squared Statistic: {chi_squared_stat}")
print(f"P-value: {p_value}")
