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
- **Macroeconomic regime features**: VIX and TNX **252-day rolling z-scores** (1-year standardisation window, extracting local regime shocks from non-stationary nominal levels), a binary volatility regime flag (VIX z-score > +1σ), and lagged versions (lags 1–5) to model delayed transmission of macroeconomic shocks.
- **Cross-sector spread dynamics**: Lagged return spreads `Spread = X_target − X_peer` against every peer sector — explicit relative-strength / lead-lag rotation signals.
- **Interaction terms**: Products of lagged momentum and rolling volatility — non-linear feature amplification.
- **Pairwise sector cross-correlations**: 15-day rolling Pearson correlations with all other sector ETFs, encoding contagion and sector-rotation dynamics.
- **Calendar effects**: Month, ISO week-of-year, day-of-week, and month-boundary flags for seasonality modelling.

All rolling features obey a strict look-ahead-free construction (`.shift(1)`), and non-stationary features are screened by an **Augmented Dickey-Fuller (ADF)** test and first-differenced where the unit-root null cannot be rejected (p > 0.05).

### Model: LightGBM — Hybrid Directional Classifier

The engine is an **`LGBMClassifier`** trained on the binary directional signal
`1[forward_return > 0]`, since sign is more learnable than magnitude on noisy
short-horizon returns. To retain a magnitude-aware regression diagnostic, each fold's
predicted up-probability is mapped back to the return scale via
`ŷ = (P(up) − 0.5) · 2 · vol_t`, where `vol_t` is the raw rolling volatility of the
target sector. The continuous out-of-sample R² is then the standard regression score of
this mapped prediction against the actual forward return. Classification-native metrics
(accuracy, ROC-AUC) and the directional Sharpe are reported alongside it.

LightGBM (Light Gradient Boosting Machine) is a histogram-based gradient boosting framework that constructs an additive ensemble of decision trees via functional gradient descent in the space of weak learners. Key algorithmic properties:

- **Leaf-wise tree growth** (vs. depth-wise): LightGBM splits the leaf with the largest loss reduction at each step, enabling deeper, more asymmetric trees and faster convergence relative to depth-first frameworks.
- **Histogram-based binning**: Continuous features are discretised into bins, reducing both memory footprint and computational complexity from O(#data × #features) to O(#bins × #features).
- **L1/L2 regularisation** (`reg_alpha`, `reg_lambda`): Penalise leaf weights to control overfitting on noisy financial return series.
- **Stochastic subsampling** (`subsample`, `colsample_bytree`): Row and column sampling per tree reduces variance and decorrelates individual estimators.
- **Sector-specific hyperparameters**: `max_depth` and `num_leaves` are tuned per sector to match each sector's return autocorrelation structure and signal-to-noise ratio.

### Training and Validation Protocol

- **Walk-forward cross-validation**: `TimeSeriesSplit` with 5 expanding folds enforces temporal ordering — test folds always post-date training folds, preventing temporal data leakage.
- **Per-fold StandardScaler**: The `StandardScaler` (zero-mean, unit-variance normalisation) is refit exclusively on each training fold. Using a globally-fit scaler would leak test-set statistics into training — a common source of inflated OOS metrics in financial ML pipelines.
- **Per-fold PCA**: The autoregressive lag block (lags 1–10) is compressed to 5 principal components fitted **inside each fold on training rows only**, eliminating the global-PCA leakage present in earlier revisions.
- **Leakage-free preprocessing decisions**: Multicollinearity pruning (pairwise |r| > 0.95) and ADF-based first-differencing decisions are made on the earliest training block, never on test data.
- **Sector dummy encoding**: A one-hot sector identifier is appended to the feature matrix, enabling a unified multi-sector model to learn sector-specific intercept adjustments.
- **Out-of-fold (OOF) aggregation**: Predictions across all test folds are concatenated to form a pseudo-OOS sequence spanning the full date range, which also feeds the portfolio backtest.

The forecasting target is the **k-day forward** average return (`returns.rolling(k).mean().shift(-k)`, k = 5) — a genuine future quantity, not a backward-looking summary of the past. See `reports/Methodology_Enhancements/` for the full leakage-audit derivation and knowledge-base references.

### Portfolio Backtest

The out-of-fold predictions are converted into a tradeable cross-sectional strategy under two schemes — proportional weighting (`w = ŷ / Σ|ŷ|`) and top-N long-short — with weights formed at `t` applied to next-day realized returns. This yields an **un-leaked walk-forward trading Sharpe**, annualised return/volatility, maximum drawdown, and a cumulative equity curve, isolating cross-sectional sector-selection skill from broad-market beta.

### Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Accuracy | Fraction of correctly classified directions | Primary classifier metric; 0.50 = coin-flip baseline |
| ROC-AUC | Area under the ROC curve | Ranking quality of the up-probability; 0.50 = random, > 0.50 = signal |
| Mapped R² | Regression R² of `(P−0.5)·2·vol` vs actual return | Magnitude-fit diagnostic; typically negative — a classifier is not a magnitude estimator |
| RMSE / MAE | Errors of the mapped continuous prediction | Reported for completeness alongside the mapped R² |
| Directional Sharpe | (μ_strategy − r_f) / σ_strategy × √252 | Annualised risk-adjusted return of the sign-based strategy (per-sector figure uses overlapping windows — indicative only) |

---

## Results

The pipeline now runs a **directional classifier** (`LGBMClassifier`) that predicts the
sign of the 5-day forward return, with a continuous out-of-sample R² retained by mapping
the predicted up-probability back to the return scale via `(P(up) − 0.5) · 2 · vol`. The
following are **genuine out-of-sample results** from a live 2008–2025 run (walk-forward,
leakage-free), not the optimistically biased figures of the pre-audit methodology.

| Sector | Accuracy | ROC-AUC | Directional Sharpe† | Mapped R²‡ |
|--------|:--------:|:-------:|:-------------------:|:----------:|
| XLK  | 0.578 | 0.511 | 2.10 | −2.33 |
| XLP  | 0.560 | 0.546 | 0.85 | −2.00 |
| XLF  | 0.554 | 0.542 | 1.37 | −2.12 |
| XLV  | 0.553 | 0.531 | 1.63 | −1.20 |
| XLB  | 0.538 | 0.512 | 0.87 | −1.93 |
| XLI  | 0.538 | 0.480 | 0.43 | −2.20 |
| XLY  | 0.537 | 0.506 | 0.55 | −1.94 |
| XLRE | 0.534 | 0.530 | 0.46 | −1.49 |
| XLU  | 0.523 | 0.513 | 0.40 | −1.33 |

**Mean directional accuracy ≈ 0.545, mean AUC ≈ 0.520** — a small but consistent edge
above the 0.50 coin-flip baseline, which is the *realistic* order of magnitude for
short-horizon daily sector prediction. XLK (57.8%) and XLP (56.0%) are the most
predictable; XLI's AUC (0.48) is below random, i.e. no reliable ranking signal.

**How to read the numbers honestly:**
- The **mapped R² is strongly negative** for every sector. This is expected and *not* a
  sign the model is broken: the heuristic probability→return mapping systematically
  over-states magnitude, so as a *point* estimator of return size it is worse than
  predicting the mean. A classifier's skill lives in **direction (accuracy/AUC)**, not
  magnitude — the negative R² simply quantifies that the mapping is a poor magnitude
  regressor, and it is reported for transparency rather than hidden.
- †The per-sector directional Sharpe is computed on **overlapping** 5-day windows, which
  inflates it through return autocorrelation; treat it as indicative, not tradeable.

**Portfolio backtest** (un-leaked, daily-rebalanced on next-day returns — the trustworthy
tradeable metric):

| Strategy | Sharpe | Annual Return | Annual Vol | Max Drawdown |
|----------|:------:|:-------------:|:----------:|:------------:|
| Proportional (`w = ŷ/Σ\|ŷ\|`) | **0.66** | 10.4% | 15.7% | −31.4% |
| Top-3 Long-Short | 0.28 | 3.2% | 11.4% | −36.4% |

The proportional cross-sectional strategy earns a positive risk-adjusted return
(Sharpe 0.66) but sits just **below passive SPY (Sharpe 0.69)** over the same window —
an honest outcome: the directional edge is real but modest, and does not by itself beat
buy-and-hold after accounting for its drawdown. See `reports/Model_Reports/` and
`reports/Methodology_Enhancements/` for full detail.

---

## Repository Structure

```
ML-Driven-Sector-ETF-Prediction/
├── data/                                # Empty on clone; regenerated by the pipeline
│   ├── sp500_sector_prices.csv          # Adjusted closing prices (2008–2025)
│   ├── sector_model_summary.csv         # OOS metrics per sector (R², RMSE, MAE, Sharpe)
│   ├── lgbm_risk_summary.csv            # Annualised return, volatility, Sharpe per sector
│   ├── portfolio_backtest_returns.csv   # Daily strategy return series per scheme
│   └── portfolio_backtest_summary.csv   # Trading Sharpe / drawdown per scheme
├── notebooks/
│   ├── sector_data_collection/          # Price ingestion
│   ├── sector_eda/                      # Exploratory data analysis
│   ├── sector_predictive_analysis/      # Per-sector walk-forward training driver
│   ├── sector_risk_analysis/            # Risk attribution and performance visualisation
│   ├── portfolio_backtest/              # Cross-sectional long-short backtest
│   ├── LGBM_model_functions/            # Shared library — single source of truth
│   └── LGBM_Prediction_Model/           # End-to-end consolidated orchestrator
├── plots/
│   ├── Feature_Importances/             # Per-sector LightGBM feature importance charts
│   ├── Price_Trends_&_Return_Distributions/
│   ├── Returns_Correlation_Heatmap/
│   ├── R²_By_Sector/
│   ├── Sector_Predictions/              # Actual vs predicted return overlays
│   ├── Sharpe_Ratio_By_Sector/
│   └── Portfolio_Backtest/              # Cross-sectional strategy equity curve
├── reports/
│   ├── Model_Reports/
│   ├── Risk_Assessment_Summary/
│   └── Methodology_Enhancements/        # Leakage audit + framework derivation
├── requirements.txt
└── README.md
```

---

## Dependencies

Install with `pip install -r requirements.txt`.

- `yfinance` — Yahoo Finance market data ingestion
- `pandas`, `numpy`, `scipy` — tabular data manipulation and numerical computing
- `lightgbm` — gradient boosting framework
- `scikit-learn` — preprocessing, model selection, evaluation metrics
- `statsmodels` — Augmented Dickey-Fuller stationarity screening
- `matplotlib`, `seaborn` — statistical visualisation

---

## Author

**Areeb Arshad**  
Sophomore, Data Science & Economics  
Virginia Tech
