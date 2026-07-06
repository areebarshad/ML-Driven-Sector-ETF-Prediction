# ML-Driven Sector ETF Prediction

## Overview

This project constructs a machine learning pipeline for forecasting short-term returns of S&P 500 sector Exchange-Traded Funds (ETFs) using a gradient boosting framework. The core hypothesis is that a combination of cross-sectional sector dynamics, macroeconomic regime indicators, and sector-specific technical features contains exploitable predictive signal for 5-day forward returns — beyond what a naïve mean or random-walk baseline would produce.

The pipeline spans the full quantitative research workflow: raw price ingestion, feature engineering, model training with walk-forward cross-validation, out-of-sample evaluation, and risk-adjusted performance attribution.

---

## Project Goals

- **Cross-sectional sector analysis**: Evaluate nine S&P 500 GICS sector ETFs across return, volatility, and risk-adjusted performance dimensions, using SPY as a broad-market benchmark.
- **Predictive modelling**: Forecast 5-day forward rolling mean returns using a high-dimensional feature matrix derived from lagged returns, technical indicators, macroeconomic regime signals, and cross-sectional momentum factors.
- **Rigorous out-of-sample evaluation**: Assess model performance via walk-forward TimeSeriesSplit cross-validation to produce unbiased OOS metrics, avoiding the look-ahead and global-scaler leakage common in financial ML.
- **Risk attribution**: Quantify each sector's annualised Sharpe ratio, volatility, and maximum drawdown characteristics to contextualise predictive model utility within a portfolio management framework.

---

## Sector Universe

| Ticker | Sector | Key Characteristics |
|--------|--------|---------------------|
| XLK | Information Technology | High momentum, strong mean-reversion trends |
| XLF | Financials | Macro-driven, fat-tailed return distribution |
| XLV | Health Care | Defensive beta, low-volatility regime |
| XLP | Consumer Staples | Near-stationary returns, smooth autocorrelation |
| XLI | Industrials | Cyclical, noisy residuals, GDP-sensitive |
| XLB | Materials | Commodity-linked, exposure to global demand shocks |
| XLU | Utilities | Mean-reverting, duration-sensitive, minimal variance |
| XLRE | Real Estate | Rate-sensitive, exposure to term-premium shocks |
| XLY | Consumer Discretionary | Retail-cycle exposure, income-elasticity driven |
| SPY | S&P 500 (benchmark) | Broad-market factor, used for beta decomposition |

---

## Methodology

### Data Collection

- **Source**: Yahoo Finance via the `yfinance` Python library.
- **Sample period**: January 1, 2008 — December 31, 2025. This window encompasses multiple distinct macroeconomic regimes including the Global Financial Crisis (GFC), the European Sovereign Debt Crisis, the COVID-19 liquidity shock, and the post-zero-interest-rate-policy (ZIRP) monetary tightening cycle — providing sufficient regime diversity for model generalisation.
- **Price data**: Adjusted closing prices used to compute arithmetic daily returns via first-order differencing.
- **Macroeconomic indicators**: CBOE Volatility Index (`^VIX`) and 10-Year US Treasury Yield (`^TNX`) sourced separately and z-scored over a 60-day rolling window.

### Feature Engineering

The feature matrix (~80+ features per sector) is constructed with strict look-ahead prevention (all features lagged by ≥ 1 trading day). Feature groups include:

- **Rolling statistical moments**: Mean, standard deviation (volatility), skewness, and excess kurtosis over a configurable rolling window — characterising distributional shifts in the return process.
- **PCA-compressed lag structure**: 10 raw autoregressive lags are projected onto 5 orthogonal principal components via PCA, eliminating multicollinearity among the lag series while preserving the autocorrelation structure.
- **Momentum and momentum-to-volatility ratio**: Price continuation signals derived from the deviation of the lagged return from its rolling mean.
- **Technical indicators**: Rolling mean and standard deviation, coefficient of variation (volatility ratio), and exponentially weighted moments (EWM) for adaptive smoothing of the return signal.
- **Systematic risk proxies**: Rolling Pearson correlation and covariance with SPY, rolling beta (covariance over variance), and idiosyncratic market deviation.
- **Drawdown**: Rolling maximum drawdown within a 20-day lookback window — a non-linear downside risk measure.
- **Cross-sectional momentum**: Equal-weighted mean and volatility of all other sector returns (excluding the target), and their ratio as a sector-level Sharpe proxy.
- **Macroeconomic regime features**: VIX and TNX z-scores (60-day rolling standardisation), a binary volatility regime flag (VIX z-score > +1σ), and lagged versions (lags 1–5) to model delayed transmission of macroeconomic shocks.
- **Interaction terms**: Products of lagged momentum and rolling volatility — non-linear feature amplification.
- **Pairwise sector cross-correlations**: 15-day rolling Pearson correlations with all other sector ETFs, encoding contagion and sector-rotation dynamics.
- **Calendar effects**: Month, ISO week-of-year, day-of-week, and month-boundary flags for seasonality modelling.

### Model: LightGBM

LightGBM (Light Gradient Boosting Machine) is a histogram-based gradient boosting framework that constructs an additive ensemble of decision trees via functional gradient descent in the space of weak learners. Key algorithmic properties:

- **Leaf-wise tree growth** (vs. depth-wise): LightGBM splits the leaf with the largest loss reduction at each step, enabling deeper, more asymmetric trees and faster convergence relative to depth-first frameworks.
- **Histogram-based binning**: Continuous features are discretised into bins, reducing both memory footprint and computational complexity from O(#data × #features) to O(#bins × #features).
- **L1/L2 regularisation** (`reg_alpha`, `reg_lambda`): Penalise leaf weights to control overfitting on noisy financial return series.
- **Stochastic subsampling** (`subsample`, `colsample_bytree`): Row and column sampling per tree reduces variance and decorrelates individual estimators.
- **Sector-specific hyperparameters**: `max_depth` and `num_leaves` are tuned per sector to match each sector's return autocorrelation structure and signal-to-noise ratio.

### Training and Validation Protocol

- **Walk-forward cross-validation**: `TimeSeriesSplit` with 5 folds enforces temporal ordering — test folds always post-date training folds, preventing temporal data leakage.
- **Per-fold StandardScaler**: The `StandardScaler` (zero-mean, unit-variance normalisation) is refit exclusively on each training fold. Using a globally-fit scaler would leak test-set statistics into training — a common source of inflated OOS metrics in financial ML pipelines.
- **Multicollinearity pruning**: Features with pairwise Pearson |r| > 0.95 are removed from the upper triangular correlation matrix before training to stabilise gradient descent.
- **Sector dummy encoding**: A one-hot sector identifier is appended to the feature matrix, enabling a unified multi-sector model to learn sector-specific intercept adjustments.
- **Out-of-fold (OOF) aggregation**: Predictions across all test folds are concatenated to form a pseudo-OOS sequence spanning the full date range.

### Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| R² | Coefficient of determination | Proportion of variance in the target explained by the model; R² ∈ (−∞, 1], with 1 indicating perfect prediction |
| RMSE | Root Mean Squared Error | Penalises large prediction errors quadratically; sensitive to outliers |
| MAE | Mean Absolute Error | Robust to outlier prediction errors; measures average absolute deviation |
| Sharpe Ratio | (μ_strategy − r_f) / σ_strategy × √252 | Annualised risk-adjusted return of a sign-based directional strategy derived from model predictions |

---

## Results

The LightGBM models demonstrated strong predictive accuracy across the sector universe, with out-of-sample **R² scores ranging from 0.79 to 0.87**, indicating that the engineered feature space captures a substantial portion of the variance in 5-day forward returns. Sectors with more stable autocorrelation structures (XLU, XLP) exhibited higher R² scores, while macro-sensitive sectors (XLF, XLI) showed greater residual variance, consistent with their exposure to unpredictable macroeconomic shocks.

The annualised **Sharpe ratio of the directional strategy** exceeded 1.0 for the majority of sectors, suggesting that the model's directional accuracy is sufficient to generate risk-adjusted excess returns above the 3% risk-free rate proxy.

---

## Repository Structure

```
ML-Driven-Sector-ETF-Prediction/
├── data/
│   ├── sp500_sector_prices.csv          # Adjusted closing prices (2008–2025)
│   ├── sector_model_summary.csv         # OOS evaluation metrics per sector
│   └── lgbm_risk_summary.csv            # Annualised return, volatility, Sharpe per sector
├── notebooks/
│   ├── sector_data_collection/          # Price ingestion and preprocessing
│   ├── sector_eda/                      # Exploratory data analysis
│   ├── sector_predictive_analysis/      # Feature engineering and model training (fixed split)
│   ├── sector_risk_analysis/            # Risk attribution and performance visualisation
│   ├── LGBM_model_functions/            # Reusable pipeline functions (walk-forward CV)
│   └── LGBM_Prediction_Model/           # End-to-end consolidated pipeline
├── plots/
│   ├── Feature_Importances/             # Per-sector LightGBM feature importance charts
│   ├── Price_Trends_&_Return_Distributions/
│   ├── Returns_Correlation_Heatmap/
│   ├── R²_By_Sector/
│   ├── Sector_Predictions/              # Actual vs predicted return overlays
│   └── Sharpe_Ratio_By_Sector/
├── reports/
│   ├── Model_Reports/
│   └── Risk_Assessment_Summary/
└── README.md
```

---

## Dependencies

- `yfinance` — Yahoo Finance market data ingestion
- `pandas`, `numpy` — tabular data manipulation and numerical computing
- `lightgbm` — gradient boosting framework
- `scikit-learn` — preprocessing, model selection, evaluation metrics
- `matplotlib`, `seaborn` — statistical visualisation

---

## Author

**Areeb Arshad**  
Sophomore, Data Science & Economics  
Virginia Tech
