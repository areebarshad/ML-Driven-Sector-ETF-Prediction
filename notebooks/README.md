# Notebooks Directory

This directory contains the Python source files for the ML-Driven Sector ETF Prediction project. All notebooks have been migrated from `.ipynb` (Jupyter/Colab) format to standalone `.py` scripts for IDE-based development and version control compatibility.

The modules are structured to follow a modular pipeline: raw data ingestion → exploratory analysis → feature engineering → model training → risk attribution.

---

## Module Breakdown

| Module | File | Description |
|--------|------|-------------|
| `sector_data_collection/` | `LGBM_sector_data_collection.py` | Downloads adjusted closing prices for all sector ETFs and SPY from Yahoo Finance via `yfinance`. Covers the full sample period (2008–2025) to ensure representation across multiple macroeconomic regimes (GFC, COVID-19, post-ZIRP tightening). Outputs `sp500_sector_prices.csv` to the `data/` directory. |
| `sector_eda/` | `LGBM_sector_EDA.py` | Performs exploratory data analysis on the sector return series. Generates: (1) non-stationary price trend plots for regime identification; (2) box plots of the daily return distribution exposing cross-sectional dispersion in mean, volatility, skewness, and excess kurtosis; (3) a Pearson correlation heatmap quantifying linear co-movement across sectors. |
| `sector_predictive_analysis/` | `LGBM_predictive_analysis.py` | Thin driver that imports the shared library and runs the leakage-free walk-forward regressor (`train_evaluate_model`) per sector, saving prediction and feature-importance plots and the OOS metrics summary. (The consolidated `LGBM_Prediction_Model` uses the hybrid *classifier* variant instead — see below.) |
| `sector_risk_analysis/` | `LGBM_sector_risk_analysis.py` | Computes annualised risk metrics: mean return (μ × 252), annualised volatility (σ × √252), and Sharpe ratio ((μ − r_f) / σ) with r_f = 3% as a long-run risk-free rate proxy. Generates R² and Sharpe ratio comparison charts across the sector universe. |
| `portfolio_backtest/` | `LGBM_portfolio_backtest.py` | Converts the walk-forward out-of-fold predictions into a tradeable cross-sectional long-short strategy and reports an **un-leaked walk-forward trading Sharpe**. Implements both a proportional scheme (`w = ŷ / Σ\|ŷ\|`) and a top-N long-short scheme, applying weights formed at `t` to next-day realized returns. Saves the cumulative equity curve and strategy return series. |
| `LGBM_model_functions/` | `LGBM_functions.py` | **Single source of truth** for the pipeline — every driver imports it. Exposes data ingestion (`download_prices`, `load_returns`, `download_macro_zscores`), stationarity screening (`adf_is_stationary`, `select_nonstationary_columns`, `apply_first_difference`), feature engineering (`generate_features`, `make_forward_target`), leakage-free training — both the regressor (`train_evaluate_model`) and the hybrid directional classifier (`train_evaluate_hybrid`) — the portfolio engine (`build_weights`, `backtest_portfolio`), and two-stage plot helpers (`plot_eda`, `plot_metric_bar`, `plot_actual_vs_predicted`, `plot_feature_importances`, `plot_equity_curve`, each with `save_path`/`close`). Both training functions run walk-forward `TimeSeriesSplit` CV (5 folds) with **per-fold `StandardScaler` and per-fold PCA** refitting, ADF-based first-differencing and pruning decided on the earliest training block — eliminating the global-scaler, global-PCA, target-construction, and static-split leakage of earlier revisions. |
| `LGBM_Prediction_Model/` | `LightGBM_prediction_model.py` | End-to-end consolidated orchestrator (canonical entry point). Runs data ingestion → EDA → **hybrid directional classifier** training with probability→return R² mapping → OOS evaluation (accuracy, AUC, mapped R², directional Sharpe) → risk attribution → cross-sectional portfolio backtest. Uses the two-stage non-blocking plotting architecture: Stage 1 saves every figure headlessly (`savefig`+`close`); Stage 2 re-renders all diagnostics under `plt.ion()` and ends with `plt.show(block=True)` for simultaneous display. |

---

## Execution Order

For a clean run from scratch, execute the modules in the following dependency order:

```
1. sector_data_collection/LGBM_sector_data_collection.py   # generates sp500_sector_prices.csv
2. sector_eda/LGBM_sector_EDA.py                           # requires sp500_sector_prices.csv
3. sector_predictive_analysis/LGBM_predictive_analysis.py  # generates sector_model_summary.csv
4. sector_risk_analysis/LGBM_sector_risk_analysis.py       # requires both CSVs
5. portfolio_backtest/LGBM_portfolio_backtest.py           # generates portfolio_backtest_*.csv
```

Alternatively, run `LGBM_Prediction_Model/LightGBM_prediction_model.py` directly — it consolidates the full pipeline (including the portfolio backtest) end-to-end.

Install dependencies first with `pip install -r ../requirements.txt`. The driver scripts import the shared library by adding `LGBM_model_functions/` to `sys.path`, so they can be run from any working directory.

---

## Design Notes

- **Look-ahead prevention**: All features are lagged by at least 1 trading day (`shift(1)`) before entering the model. No contemporaneous return information is used as a predictor.
- **Leakage controls**: `StandardScaler` normalisation (zero-mean, unit-variance) is fitted exclusively on training data within each cross-validation fold. Fitting the scaler on the full dataset before splitting would leak test-set statistics, inflating OOS metrics.
- **Multicollinearity pruning**: Features with pairwise Pearson |r| > 0.95 are identified from the upper triangular correlation matrix (computed on the earliest training block) and removed prior to model training, stabilising gradient descent in the boosting procedure.
- **Per-fold PCA dimensionality reduction**: Raw autoregressive lag features (lags 1–10) are projected onto 5 orthogonal principal components **fitted inside each CV fold on training rows only**, compressing collinear lag information while retaining the dominant axes of autocorrelation variance — without the global-PCA leakage of earlier revisions.
- **Stationarity screening**: An Augmented Dickey-Fuller (ADF) test flags non-stationary features (p > 0.05), which are first-differenced; the differencing decision is made on the earliest training block to stay leakage-free.
- **File paths**: All scripts resolve the `data/` and `plots/` directories relative to `__file__` via `LGBM_functions.get_data_dir()` / `get_plots_dir()`, making them portable across systems without hardcoded absolute paths.

---

## Related Directories

- `../data/` — Input price CSVs and output model evaluation summaries
- `../plots/` — All visualisations generated during EDA, model evaluation, and risk analysis
- `../reports/` — Model performance reports and risk assessment summaries
