# Notebooks Directory

This directory contains the Python source files for the ML-Driven Sector ETF Prediction project. All notebooks have been migrated from `.ipynb` (Jupyter/Colab) format to standalone `.py` scripts for IDE-based development and version control compatibility.

The modules are structured to follow a modular pipeline: raw data ingestion → exploratory analysis → feature engineering → model training → risk attribution.

---

## Module Breakdown

| Module | File | Description |
|--------|------|-------------|
| `sector_data_collection/` | `LGBM_sector_data_collection.py` | Downloads adjusted closing prices for all sector ETFs and SPY from Yahoo Finance via `yfinance`. Covers the full sample period (2008–2025) to ensure representation across multiple macroeconomic regimes (GFC, COVID-19, post-ZIRP tightening). Outputs `sp500_sector_prices.csv` to the `data/` directory. |
| `sector_eda/` | `LGBM_sector_EDA.py` | Performs exploratory data analysis on the sector return series. Generates: (1) non-stationary price trend plots for regime identification; (2) box plots of the daily return distribution exposing cross-sectional dispersion in mean, volatility, skewness, and excess kurtosis; (3) a Pearson correlation heatmap quantifying linear co-movement across sectors. |
| `sector_predictive_analysis/` | `LGBM_predictive_analysis.py` | Constructs the full feature matrix (rolling moments, PCA-compressed lags, macroeconomic regime indicators, cross-sectional signals, interaction terms) and trains a sector-specific LightGBM regressor on a fixed temporal split (2010–2020 in-sample, 2021–2025 out-of-time). Includes a feature importance pruning step (65th percentile threshold) to remove noise features. |
| `sector_risk_analysis/` | `LGBM_sector_risk_analysis.py` | Computes annualised risk metrics: mean return (μ × 252), annualised volatility (σ × √252), and Sharpe ratio ((μ − r_f) / σ) with r_f = 3% as a long-run risk-free rate proxy. Generates R² and Sharpe ratio comparison charts across the sector universe. |
| `LGBM_model_functions/` | `LGBM_functions.py` | Reusable function library exposing `generate_features()`, `train_evaluate_model()`, `plot_feature_importances()`, and `plot_actual_vs_predicted()`. The `train_evaluate_model()` function implements walk-forward `TimeSeriesSplit` cross-validation (5 folds) with per-fold `StandardScaler` refitting to eliminate global-scaler data leakage — a common source of inflated OOS metrics in financial ML pipelines. |
| `LGBM_Prediction_Model/` | `LightGBM_prediction_model.py` | End-to-end consolidated pipeline that executes the full workflow in a single script: data ingestion → EDA → feature engineering → walk-forward model training → OOS evaluation → risk attribution → visualisation. Serves as the canonical entry point for reproducing all reported results. |

---

## Execution Order

For a clean run from scratch, execute the modules in the following dependency order:

```
1. sector_data_collection/LGBM_sector_data_collection.py   # generates sp500_sector_prices.csv
2. sector_eda/LGBM_sector_EDA.py                           # requires sp500_sector_prices.csv
3. sector_predictive_analysis/LGBM_predictive_analysis.py  # generates sector_model_summary.csv
4. sector_risk_analysis/LGBM_sector_risk_analysis.py       # requires both CSVs
```

Alternatively, run `LGBM_Prediction_Model/LightGBM_prediction_model.py` directly — it consolidates the full pipeline end-to-end.

---

## Design Notes

- **Look-ahead prevention**: All features are lagged by at least 1 trading day (`shift(1)`) before entering the model. No contemporaneous return information is used as a predictor.
- **Leakage controls**: `StandardScaler` normalisation (zero-mean, unit-variance) is fitted exclusively on training data within each cross-validation fold. Fitting the scaler on the full dataset before splitting would leak test-set statistics, inflating OOS metrics.
- **Multicollinearity pruning**: Features with pairwise Pearson |r| > 0.95 are identified from the upper triangular correlation matrix and removed prior to model training, stabilising gradient descent in the boosting procedure.
- **PCA dimensionality reduction**: Raw autoregressive lag features (lags 1–10) are projected onto 5 orthogonal principal components, compressing collinear lag information while retaining the dominant axes of autocorrelation variance.
- **File paths**: All scripts resolve the `data/` directory relative to `__file__` using `os.path`, making them portable across systems without hardcoded absolute paths.

---

## Related Directories

- `../data/` — Input price CSVs and output model evaluation summaries
- `../plots/` — All visualisations generated during EDA, model evaluation, and risk analysis
- `../reports/` — Model performance reports and risk assessment summaries
