import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Replace 'your_file_name.xlsx' with the actual path to your file.
file_path = '.\Problem Set 1\Problem_Set1_2025_AX.xlsx'

# Read the specific sheet 'raw_data' into a DataFrame.
# The header is in the first row, which pandas correctly assumes by default.
df = pd.read_excel(file_path, sheet_name='raw_data')
# Print the number of rows and columns
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Print the first 5 rows to verify the data was loaded correctly.
# print(df.head())
print(df.columns)

returns = df.iloc[:, 1:51]  # Select only the 50 stock columns
each_stock_variance = returns.var()
print(each_stock_variance)

portfolio_sizes = [5, 10, 25, 50]
means = []
stds = []
# portfolio_variance = variance + covariance
portfolio_variances = []
variances = []
covariance = []


for n in portfolio_sizes:
    # Form equal-weight portfolio: mean across the first n stocks for each row
    # In pandas, axis=0 means "down the rows" (column-wise), and axis=1 means "across the columns" (row-wise).
    portfolio_returns = returns.iloc[:, :n].mean(axis=1)
    print(portfolio_returns)
    means.append(portfolio_returns.mean())
    stds.append(portfolio_returns.std())
    portfolio_variances.append(portfolio_returns.var())
    variances.append(each_stock_variance.iloc[:n].sum() / (n ** 2))

# Print results
for n, mean, std, portfolio_variance, variance in zip(portfolio_sizes, means, stds, portfolio_variances, variances):
    print(f"Portfolio size: {n}, Mean return: {mean:.4f}, Std dev: {std:.4f}, portfolio_variance: {portfolio_variance:.4f}, variance: {variance:.4f}")


# hw1(a) Plot standard deviation vs. number of stocks
plt.figure(figsize=(8, 5))
plt.plot(portfolio_sizes, stds, marker='o')
plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Estimated Standard Deviation')
plt.title('Portfolio Standard Deviation vs. Number of Stocks')
plt.grid(True)
plt.show()

# hw1(b) plot variance as % of portfolio variance
ratios = [v / pv * 100 for v, pv in zip(variances, portfolio_variances)]
plt.figure(figsize=(8, 5))
plt.plot(portfolio_sizes, ratios, marker='o')
plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Variance / Portfolio Variance (%)')
plt.title('Variance as % of Portfolio Variance vs. Number of Stocks')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
plt.grid(True)
plt.show()

# hw1(c) Value weighted portfolio vs equal weighted portfolio
print("It depends. On one hand, large caps are more stable. On the other hand, the covariance among large caps might be higher than covariance among large AND small caps.")
print("As the number of stocks in the portfolio increases, the equal-weighted portfolio may have less variance than the value-weighted portfolio.")


# hw1(d) Hypothesis testing
# calculating the t-statistic for each portfolio size
t_stats = []
p_values = []
n_obs = returns.shape[0]

for mean, std in zip(means, stds):
    t = mean / (std / (n_obs ** 0.5))
    t_stats.append(t)
    # Two-sided p-value
    p = 2 * (1 - stats.t.cdf(abs(t), df=n_obs-1))
    p_values.append(p)

# Print t-statistics and p-values
for n, t, p in zip(portfolio_sizes, t_stats, p_values):
    print(f"Portfolio size: {n}, t-statistic: {t:.4f}, p-value: {p:.4f}")

print("reject the null hypothesis (H0: mean return = 0) at 5% significance level for all portfolio sizes since p-values are all less than 0.05.")


# hw1(e) Distribution check
from scipy.stats import skew, kurtosis, norm, chi2

def normality_test(series, name="Series"):
    n = len(series)
    s = skew(series)
    k = kurtosis(series, fisher=False)  # Pearson's definition (normal=3)
    # Studentized range: (max - min) / std
    studentized_range = (series.max() - series.min()) / series.std()
    print(f"\n{name}:")
    print(f"  Studentized range: {studentized_range:.4f}")
    print(f"  Skewness: {s:.4f} (expected: 0)")
    print(f"  Kurtosis: {k:.4f} (expected: 3)")
    # Chi-square test for normality (approximate)
    chi2_stat = n * (s**2 / 6 + ((k-3)**2) / 24)
    p_value = 1 - chi2.cdf(chi2_stat, df=2)
    print(f"  Chi-square stat: {chi2_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Reject normality at 5% significance level.")
    else:
        print("  Cannot reject normality at 5% significance level.")

# First stock
normality_test(returns.iloc[:, 0], name="First Stock")
# Equal-weighted portfolio of all 50 stocks
ew_portfolio = returns.mean(axis=1)
normality_test(ew_portfolio, name="Equal-weighted Portfolio (50 stocks)")

# Market index (market portfolio index of all NYSE, AMEX, and Nasdaq stocks not found)
market_df = pd.read_excel(file_path, sheet_name='mkt_return')
market_col = "Market (Value Weighted Index)"  # Adjust if your column name is different

# Get the market returns as a Series
market_returns = market_df[market_col]
normality_test(market_returns, name=market_col)

# hw1(f) Regression analysis
regression_data = df.loc[:, "TXN":"VMC"] 
print("type of regression_data:", type(regression_data))
print("shape of regression_data:", regression_data.shape) 
# Get the first 10 stock names (TXN through VMC)
stock_names = regression_data.columns[:10]
print("type of stock_names:", type(stock_names))
print("shape of stock_names:", stock_names.shape)

from scipy.stats import linregress
print("\nRegression results (first 10 stocks on market index):")
print(f"{'Stock':<8} {'Alpha (Intercept)':>15} {'Beta (Slope)':>15} {'R-squared':>15}")
for stock in stock_names:
    y = regression_data[stock]
    x = market_returns
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    print(f"{stock:<8} {intercept:15.6f} {slope:15.6f} {r_squared:15.6f}")

print("The slope coefficients (betas) indicate the sensitivity of each stock's returns to the market returns. A beta greater than 1 suggests the stock is more volatile than the market, while a beta less than 1 indicates it is less volatile.")
print("The intercepts (alphas) represent the expected return of the stock when the market return is zero. A positive alpha indicates the stock outperforms the market, while a negative alpha suggests underperformance.")
print("R-squared values indicate the proportion of variance in the stock returns explained by the market returns. Higher R-squared values suggest a better fit of the regression model to the data.")

# hw1 Part2 Market Model regression
# compare df.loc[:, "TXN"] and market_returns
print("\nComparing TXN returns and Market returns:")
print("Type of TXN returns:", type(df.loc[:, "TXN"]))
print("Shape of TXN returns:", df.loc[:, "TXN"].shape)
print("Type of Market returns:", type(market_returns))
print("Shape of Market returns:", market_returns.shape)

# merge date, TXN, and market returns
merged_df = pd.concat([df.loc[:, ["date", "TXN"]], market_returns], axis=1)
merged_df.columns = ["date", "TXN", market_col]
print("Merged DataFrame:")
print(merged_df.head())

import matplotlib.dates as mdates

# Ensure 'date' is datetime
print("Type of 'date' column before conversion:", type(merged_df['date'].iloc[0]))
merged_df['date'] = pd.to_datetime(merged_df['date'],format='%Y%m')
print("Type of 'date' column after to_datetime:", type(merged_df['date'].iloc[0]))
print(merged_df.head())

# Set date as index for easier rolling calculations
print("Shape of merged DataFrame before set_index:", merged_df.shape)
merged_df = merged_df.set_index('date')
print("Shape of merged DataFrame after set_index:", merged_df.shape)

# First method: expanding window (all past data up to each date)
merged_df['TXN_vol_expanding'] = merged_df['TXN'].expanding().std()
merged_df['Market_vol_expanding'] = merged_df[market_col].expanding().std()

# Second method: rolling window (1 year = 252 trading days)
merged_df['TXN_vol_rolling'] = merged_df['TXN'].rolling(window="365D", min_periods=12).std()
merged_df['Market_vol_rolling'] = merged_df[market_col].rolling(window="365D", min_periods=12).std()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index, merged_df['TXN_vol_expanding'], label='TXN Expanding Volatility', color='royalblue')
plt.plot(merged_df.index, merged_df['TXN_vol_rolling'], label='TXN 1-Year Rolling Volatility', linestyle='--', color='royalblue')
plt.plot(merged_df.index, merged_df['Market_vol_expanding'], label='Market Expanding Volatility', color='orange')
plt.plot(merged_df.index, merged_df['Market_vol_rolling'], label='Market 1-Year Rolling Volatility', linestyle='--', color='orange')
plt.xlabel('Date')
plt.ylabel('Volatility (Std Dev of Returns)')
plt.title('TXN and Market Volatility: Expanding vs. 1-Year Rolling Window')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Why does the estimate using only the most recent year of data move around more than the estimate using all data?  ")
print("The 1-year rolling window uses only the most recent data, making it more sensitive to short-term fluctuations and market events. In contrast, the expanding window incorporates all past data, smoothing out volatility and providing a more stable estimate over time.")

# hw1 Part2 OLS beta estimation with confidence intervals
import numpy as np
#Calculate OLS beta and its 95% confidence interval.
def calc_beta_and_ci(x, y, alpha=0.05):
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    x = np.array(x)
    y = np.array(y)
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    residuals = y - (beta * x + (y_mean - beta * x_mean))
    s2 = np.sum(residuals ** 2) / (n - 2)
    se_beta = np.sqrt(s2 / np.sum((x - x_mean) ** 2))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta
    return beta, ci_lower, ci_upper

print("Type of index after set_index:", type(merged_df.index))

# Expanding window
expanding_betas = []
expanding_ci_lower = []
expanding_ci_upper = []

for i in range(2, len(merged_df)):
    x = merged_df[market_col].iloc[:i]
    y = merged_df["TXN"].iloc[:i]
    beta, ci_l, ci_u = calc_beta_and_ci(x, y)
    expanding_betas.append(beta)
    expanding_ci_lower.append(ci_l)
    expanding_ci_upper.append(ci_u)

# Rolling window (1 year)
rolling_betas = []
rolling_ci_lower = []
rolling_ci_upper = []
window = "365D"

for end_date in merged_df.index[2:]:
    start_date = end_date - pd.Timedelta(days=365)
    window_df = merged_df.loc[start_date:end_date]
    x = window_df[market_col]
    y = window_df["TXN"]
    beta, ci_l, ci_u = calc_beta_and_ci(x, y)
    rolling_betas.append(beta)
    rolling_ci_lower.append(ci_l)
    rolling_ci_upper.append(ci_u)

# Align dates for plotting
plot_dates = merged_df.index[2:]

plt.figure(figsize=(12, 6))
plt.plot(plot_dates, expanding_betas, label='Expanding OLS Beta', color='royalblue')
plt.fill_between(plot_dates, expanding_ci_lower, expanding_ci_upper, color='royalblue', alpha=0.2, label='Expanding 95% CI')
plt.plot(plot_dates, rolling_betas, label='1-Year Rolling OLS Beta', color='orange')
plt.fill_between(plot_dates, rolling_ci_lower, rolling_ci_upper, color='orange', alpha=0.2, label='Rolling 95% CI')
plt.xlabel('Date')
plt.ylabel('OLS Beta')
plt.title('TXN OLS Beta: Expanding vs. 1-Year Rolling Window (with 95% CI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(-4, 6)
plt.show()
