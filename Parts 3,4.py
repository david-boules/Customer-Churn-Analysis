import pandas as pd
import numpy as np


# Load the dataset (make sure 'Churn_Modelling.csv' is in your working directory)
df = pd.read_csv('Churn_Modelling.csv')

# Rename columns for clarity:
# - 'Geography' represents the country.
# - 'Exited' indicates if a customer churned (1) or was retained (0).
df.rename(columns={'Geography': 'Country', 'Exited': 'Churned'}, inplace=True)

### -------------------------------
# Q3: International Churn Behavior
### -------------------------------

# Initialize a list to store churn statistics per country.
country_stats = []

# Group data by country and compute the sample proportion and its 95% confidence interval.
for country, group in df.groupby('Country'):
    n = len(group)
    churned = group['Churned'].sum()
    # Sample proportion of churned customers in this country.
    prop = churned / n
    # Standard error of the proportion.
    se = np.sqrt(prop * (1 - prop) / n)
    # 95% Confidence Interval
    ci_lower = prop - 1.96 * se
    ci_upper = prop + 1.96 * se

    country_stats.append({
        'Country': country,
        'n': n,
        'Churned': churned,
        'Proportion': prop,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper
    })

stats_df = pd.DataFrame(country_stats)
print("Churn statistics by country:")
print(stats_df)

# For illustrative purposes, compute the relative risk between France and Germany (if both are available)
if set(['France', 'Germany']).issubset(stats_df['Country'].unique()):
    p_france = stats_df.loc[stats_df['Country'] == 'France', 'Proportion'].values[0]
    p_germany = stats_df.loc[stats_df['Country'] == 'Germany', 'Proportion'].values[0]
    relative_risk = p_france / p_germany if p_germany != 0 else np.nan
    print("\nRelative Risk (France / Germany): {:.4f}".format(relative_risk))
else:
    print("\nData for both France and Germany are not available to compute relative risk.")

# We got proportions for each country and the relative risk between France and Germany.
# Germany had the highest proportion of customers by far.
# France is less risky that Germany for investment, as it has a lower churn rate

### -------------------------------
# Q4: Salary Variability for Churned vs. Non-Churned Customers
### -------------------------------

# Group by churn status (0: Retained, 1: Churned) and calculate salary variance along with additional summary stats.
salary_stats = df.groupby('Churned')['EstimatedSalary'].agg(['mean', 'std', 'var', 'count'])

# Rename indices for clarity.
salary_stats = salary_stats.rename(index={0: 'Retained', 1: 'Churned'})
print("\nSalary statistics by churn status:")
print(salary_stats)

# Compute the variance ratio: variance of salaries for churned customers divided by variance for retained customers.
if salary_stats.loc['Retained', 'var'] != 0:
    variance_ratio = salary_stats.loc['Churned', 'var'] / salary_stats.loc['Retained', 'var']
    print("\nVariance Ratio (Churned / Retained): {:.4f}".format(variance_ratio))
else:
    print("\nVariance for retained customers is zero, so variance ratio cannot be computed.")

# We got the variances for the churned and retained customers, and their variances, in particular
# We also got a ratio between variances (Churned/Retained) and the ratio was  very close to 1, which implies...
# There is no significant difference in the salaries of churned and unchurned customers
