import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Load adjusted closing prices and compute arithmetic daily log-returns (pct_change proxy).
# Dropping NaN values removes the first row introduced by the differencing operation.
data = pd.read_csv(os.path.join(DATA_DIR, 'sp500_sector_prices.csv'), index_col=0, parse_dates=True)
returns = data.pct_change().dropna()

# --- Price trend visualization ---
# Non-stationary price levels plotted to inspect structural breaks,
# momentum regimes, and cross-sectional dispersion over the sample period.
data.plot(figsize=(12, 6))
plt.title('Sector ETF Price Trends (2008–2025)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.tight_layout()
plt.show()

# --- Return distribution analysis ---
# Box plots expose the first four moments of the return distribution:
# median drift, interquartile spread, tail outliers (kurtosis), and directional skew.
returns.plot(kind='box', figsize=(12, 6))
plt.title('Sector ETF Daily Return Distributions')
plt.xlabel('Sector ETF')
plt.ylabel('Daily Return')
plt.tight_layout()
plt.show()

# --- Cross-sectional correlation heatmap ---
# Pearson correlation matrix of daily returns quantifies linear co-movement between sectors.
# High intra-market correlations reduce diversification benefits; low correlations
# indicate orthogonal exposure to distinct economic risk factors.
corr = returns.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Sector ETF Returns Correlation Heatmap')
plt.tight_layout()
plt.show()

print("Sector EDA complete.")
