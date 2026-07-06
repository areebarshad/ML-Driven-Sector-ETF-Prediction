import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Load price data and compute daily arithmetic returns via first-order differencing.
returns = pd.read_csv(
    os.path.join(DATA_DIR, 'sp500_sector_prices.csv'), index_col=0, parse_dates=True
).pct_change().dropna()

# Annualise returns and volatility assuming 252 trading days per year.
# Volatility is the annualised standard deviation of daily returns — a proxy for total risk
# under the assumption of i.i.d. normally distributed increments (which the EDA may refute).
mean_return = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

# Risk-free rate proxy: 3% p.a. approximating the long-run average 10-year US Treasury yield.
# The Sharpe ratio measures excess return per unit of total risk; values > 1 are conventionally
# considered acceptable, > 2 strong, and > 3 exceptional on a risk-adjusted basis.
risk_free_rate = 0.03
sharpe_ratio = (mean_return - risk_free_rate) / volatility

summary = pd.DataFrame({
    'Annual Return':     mean_return,
    'Annual Volatility': volatility,
    'Sharpe Ratio':      sharpe_ratio,
})
summary = summary.sort_values(by='Sharpe Ratio', ascending=False)

print(summary.to_string())
summary.to_csv(os.path.join(DATA_DIR, 'lgbm_risk_summary.csv'))

# Load model evaluation metrics to co-plot predictive performance alongside risk metrics.
results_df = pd.read_csv(os.path.join(DATA_DIR, 'sector_model_summary.csv'))

plt.style.use('ggplot')

# --- R² by Sector ---
# The coefficient of determination (R²) quantifies the proportion of variance in the
# target (forward return) explained by the LightGBM model's feature set.
# R² ∈ [0, 1]; values approaching 1 indicate high explanatory power, while
# negative values indicate the model underperforms a naive mean predictor.
results_sorted_r2 = results_df.sort_values(by='R2', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(results_sorted_r2['Sector'], results_sorted_r2['R2'], alpha=0.75, color='steelblue')
plt.title('Out-of-Sample R² by Sector', fontsize=16)
plt.xlabel('Sector ETF')
plt.ylabel('R² (Coefficient of Determination)')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Sharpe Ratio by Sector ---
# Annualised Sharpe ratio computed from historical return series.
# Provides a risk-adjusted ranking of sector ETFs independent of model predictions.
summary_sorted = summary.sort_values(by='Sharpe Ratio', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(summary_sorted.index, summary_sorted['Sharpe Ratio'], color='mediumseagreen', alpha=0.75)
plt.title('Annualised Sharpe Ratio by Sector (2008–2025)', fontsize=16)
plt.xlabel('Sector ETF')
plt.ylabel('Sharpe Ratio')
plt.axhline(y=1.0, color='black', linestyle='--', label='Sharpe = 1.0 threshold')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

print("Sector risk analysis complete.")
