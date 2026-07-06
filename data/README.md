# Data Directory

This directory holds the raw and generated data artifacts for the ML-Driven Sector
ETF Prediction pipeline. It is intentionally shipped empty (tracked via `.gitkeep`) —
all files below are regenerated deterministically by running the pipeline scripts in
`notebooks/`. The pipeline resolves this directory relative to the source files via
`LGBM_functions.get_data_dir()`, so no absolute paths are hardcoded.

## Generated Contents

| File | Produced by | Description |
|------|-------------|-------------|
| `sp500_sector_prices.csv` | `sector_data_collection` | Adjusted closing prices for the 9 sector ETFs + SPY, 2008–2025, from `yfinance`. |
| `sector_model_summary.csv` | `sector_predictive_analysis` / `LGBM_Prediction_Model` | Out-of-sample metrics per sector: R², RMSE, MAE, and the annualized directional Sharpe. |
| `lgbm_risk_summary.csv` | `sector_risk_analysis` | Annualized return, volatility, and Sharpe ratio per sector ETF. |
| `portfolio_backtest_returns.csv` | `portfolio_backtest` | Daily strategy return series for each allocation scheme (proportional, top-N long-short). |
| `portfolio_backtest_summary.csv` | `portfolio_backtest` | Un-leaked walk-forward trading Sharpe, annualized return/volatility, and max drawdown per strategy. |

## External Data Sources

Downloaded on demand during feature engineering (cached per process):

- **VIX** (`^VIX`, CBOE Implied Volatility Index) — 30-day forward-looking implied
  variance. Transformed into a **252-day rolling z-score** to extract local volatility
  *regime shocks* rather than non-stationary nominal levels, then lagged one day.
- **TNX** (`^TNX`, 10-Year US Treasury Yield) — the term-structure / duration risk
  factor. Also converted to a 252-day rolling z-score and lagged.

Both series feed a binary **volatility-regime indicator** (`market_vol_regime`,
triggered when the VIX z-score exceeds +1σ) and lagged (1–5 day) macro features.

## Notes

- All temporal features obey a strict look-ahead-free construction (`.shift(1)`); the
  prediction target is the **k-day forward** average return (`.shift(-k)`).
- Files are overwritten on each pipeline run. See `../notebooks/README.md` for the
  execution order and `../reports/Methodology_Enhancements/README.md` for the
  leakage-control derivation.
