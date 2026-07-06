# Model Report — ML-Driven Sector ETF Prediction

## Overview

**Objective**: Forecast the direction of the 5-day forward return for nine S&P 500 GICS sector ETFs using a gradient boosted decision tree classifier, with SPY serving as a broad-market benchmark for beta decomposition and systematic risk proxies.

**Approach**: A sector-specific **LightGBM Classifier** is trained on the binary directional signal `1[forward_return > 0]` over a high-dimensional feature matrix (~80+ engineered features per sector), using walk-forward time-series cross-validation to produce unbiased out-of-sample (OOS) estimates. A continuous R² is retained by mapping the predicted up-probability back to the return scale via `ŷ = (P(up) − 0.5) · 2 · vol`. The validation protocol enforces strict temporal ordering with per-fold feature normalisation and per-fold PCA to eliminate look-ahead, global-scaler, and global-PCA leakage.

**Sectors modelled**: XLK (Tech), XLF (Financials), XLV (Healthcare), XLI (Industrials), XLP (Consumer Staples), XLB (Materials), XLU (Utilities), XLRE (Real Estate), XLY (Consumer Discretionary).

---

## Data Summary

| Property | Value |
|----------|-------|
| Sample period | January 2008 — December 2025 |
| ETF price source | Yahoo Finance (`yfinance`), adjusted closing prices |
| Macro indicators | `^VIX` (CBOE Implied Volatility Index), `^TNX` (10-Year US Treasury Yield) |
| Return computation | Arithmetic daily returns via first-order percentage differencing |
| Regime diversity | GFC (2008–09), European Sovereign Debt Crisis (2011–12), COVID-19 shock (2020), post-ZIRP tightening cycle (2022–23) |

---

## Feature Engineering

All features are lagged by ≥ 1 trading day to enforce look-ahead-free construction.

| Feature Group | Description |
|---------------|-------------|
| Rolling statistical moments | Mean, standard deviation, skewness, excess kurtosis over a configurable window — characterises non-stationary distributional shifts |
| Momentum and momentum ratio | Deviation from rolling mean; ratio to rolling volatility as a signal-to-noise normaliser |
| Per-fold PCA-compressed lags | 10 raw autoregressive lags retained in the feature matrix, compressed into 5 orthogonal principal components **fitted per fold on training rows only** (no global-PCA leakage) |
| Technical indicators | Rolling mean/std, coefficient of variation (volatility ratio), exponentially weighted mean/std (EWM) |
| Systematic risk proxies | Rolling Pearson correlation and covariance with SPY; rolling beta (cov/var); idiosyncratic market deviation |
| Maximum drawdown | Rolling 20-day maximum drawdown — non-linear downside risk measure |
| Cross-sectional momentum | Equal-weighted mean and volatility of all other sector returns; their ratio as a sector Sharpe proxy |
| Cross-sector spread dynamics | Lagged return spreads `X_target − X_peer` against every peer — relative-strength / lead-lag rotation signals |
| Macroeconomic regime features | VIX and TNX **252-day rolling z-scores**; binary stress regime flag (VIX z-score > +1σ); lags 1–5 for delayed transmission |
| Stationarity screening | Augmented Dickey-Fuller test on each feature; non-stationary columns (p > 0.05) first-differenced, decided on the earliest training block |
| Interaction terms | Products of lagged momentum and rolling volatility — non-linear signal amplification |
| Pairwise sector correlations | 15-day rolling Pearson correlations with all other sectors — contagion and rotation dynamics |
| Calendar features | Month, ISO week-of-year, day-of-week, month-start/end flags — seasonality effects |

---

## Model Architecture

**Algorithm**: LightGBM Classifier — a histogram-based gradient boosted decision tree framework implementing leaf-wise (best-first) tree growth, which splits the highest-gain leaf at each step rather than growing all leaves at a fixed depth. The classifier optimises the binary directional signal; the predicted up-probability is mapped back to the return scale (`(P−0.5)·2·vol`) to retain a continuous R² diagnostic.

**Regularisation**: L1 penalty (`reg_alpha = 0.05`) and L2 penalty (`reg_lambda = 0.5`) applied to leaf weights. Stochastic subsampling (`subsample = 0.8`, `colsample_bytree = 0.8`) decorrelates individual estimators and reduces overfitting on noisy return series.

**Sector-specific hyperparameters** (`max_depth`, `num_leaves`) are calibrated to each sector's autocorrelation structure and signal-to-noise ratio:

| Sector | max_depth | num_leaves | Rationale |
|--------|-----------|------------|-----------|
| XLK | 9 | 64 | Trendy, momentum-driven |
| XLF | 14 | 128 | Macro-sensitive, complex non-linear dynamics |
| XLV | 10 | 72 | Stable beta, low-volatility regime |
| XLP | 7 | 48 | Near-stationary, smooth autocorrelation |
| XLI | 12 | 96 | Cyclical, noisy GDP-sensitive residuals |
| XLB | 11 | 80 | Commodity-linked, fat-tailed |
| XLU | 6 | 32 | Mean-reverting, minimal variance |
| XLRE | 10 | 64 | Duration-sensitive to yield curve shocks |
| XLY | 10 | 80 | Income-cycle exposure, retail-driven |

**Validation**: `TimeSeriesSplit` with 5 expanding folds. Within each fold the `StandardScaler` and the lag-block PCA are refit on training rows only — fitting either on the full dataset before splitting would leak test-period statistics into training and inflate OOS metrics.

**Multicollinearity pruning & stationarity**: Features with pairwise Pearson |r| > 0.95 are pruned, and non-stationary features are first-differenced; both decisions are made on the earliest training block so they never see test data.

---

## Performance Metrics (Out-of-Sample)

These are **genuine out-of-sample results** from a live 2008–2025 run of the
leakage-free walk-forward pipeline with the hybrid directional classifier. They
supersede the pre-audit figures (which reported an inflated ~0.84 average R² from a
leaky static-split regressor). Values regenerated from `data/sector_model_summary.csv`.

| Sector | Accuracy | ROC-AUC | Directional Sharpe† | Mapped R²‡ | RMSE | MAE |
|--------|:--------:|:-------:|:-------------------:|:----------:|------|-----|
| XLK  | 0.5783 | 0.5106 | 2.1012 | −2.3342 | 0.011724 | 0.008104 |
| XLF  | 0.5537 | 0.5419 | 1.3701 | −2.1193 | 0.010861 | 0.006947 |
| XLV  | 0.5532 | 0.5312 | 1.6336 | −1.2008 | 0.006796 | 0.004977 |
| XLP  | 0.5603 | 0.5461 | 0.8465 | −2.0031 | 0.006682 | 0.004440 |
| XLI  | 0.5376 | 0.4796 | 0.4324 | −2.2037 | 0.010177 | 0.006521 |
| XLB  | 0.5381 | 0.5124 | 0.8719 | −1.9314 | 0.010069 | 0.006775 |
| XLU  | 0.5234 | 0.5132 | 0.3981 | −1.3269 | 0.007955 | 0.005288 |
| XLRE | 0.5338 | 0.5299 | 0.4643 | −1.4871 | 0.009131 | 0.006215 |
| XLY  | 0.5366 | 0.5056 | 0.5507 | −1.9370 | 0.010893 | 0.007402 |

**Mean accuracy ≈ 0.545 · Mean AUC ≈ 0.520 · Mean mapped R² ≈ −1.83**

† Per-sector directional Sharpe uses **overlapping** 5-day windows and is therefore
optimistically biased by return autocorrelation — indicative only. The trustworthy
tradeable figure is the portfolio backtest below.
‡ The mapped R² is negative by construction: `(P−0.5)·2·vol` is a deliberately simple
probability→return mapping that over-states magnitude. A classifier's skill is
directional; R² is reported only as an honest magnitude-fit diagnostic.

## Portfolio Backtest (un-leaked, daily-rebalanced)

| Strategy | Sharpe | Annual Return | Annual Vol | Max Drawdown |
|----------|:------:|:-------------:|:----------:|:------------:|
| Proportional (`w = ŷ / Σ\|ŷ\|`) | 0.6625 | 10.37% | 15.65% | −31.41% |
| Top-3 Long-Short | 0.2777 | 3.16% | 11.39% | −36.42% |

Passive SPY over the same window returns a Sharpe of **0.687** — the proportional
strategy (0.66) is competitive but does **not** beat buy-and-hold, an honest reflection
of a small directional edge.

---

## Key Insights

- **Directional edge is real but modest.** Eight of nine sectors classify direction
  above the 0.50 coin-flip line (mean 0.545); XLK (57.8%) and XLP (56.0%) are most
  predictable. This is the realistic order of magnitude for daily sector prediction —
  the pre-audit ~0.84 R² was an artefact of look-ahead leakage.
- **AUC is thinner than accuracy.** Mean AUC ≈ 0.52, and XLI's 0.48 is below random,
  meaning its probability *ranking* carries essentially no signal even though raw
  accuracy is ~0.54 (a class-imbalance effect: markets drift up).
- **Magnitude is not learned.** The uniformly negative mapped R² confirms the model
  captures *sign*, not *size* — consistent with the choice of a classification engine.
- **Strategy value is limited.** The cross-sectional long-short books are positive but
  sub-SPY on a risk-adjusted basis; the edge would likely not survive realistic
  transaction costs. This is stated plainly rather than framed as outperformance.

---

## Limitations

- **Regime non-stationarity**: The model implicitly assumes that the statistical relationships between features and forward returns are stable across macroeconomic regimes. Structural breaks (e.g. zero-interest-rate regime transitions) can cause feature-return relationships to shift, degrading OOS performance.
- **Stationarity assumption**: Linear feature construction (rolling means, correlations) assumes weakly stationary return series. Heavy-tailed distributions and volatility clustering (ARCH/GARCH effects) in financial returns violate classical stationarity assumptions.
- **Transaction cost abstraction**: Reported Sharpe ratios are gross of bid-ask spreads, market impact, and short-selling costs. Given the daily-rebalanced cross-sectional strategy already sits below passive SPY (0.66 vs 0.69), the small edge would likely not survive realistic costs.
- **Heuristic magnitude mapping**: The continuous prediction is a fixed `(P−0.5)·2·vol` transform, not a calibrated regression, hence the negative mapped R². A stacked regressor on the classifier probability, or isotonic/Platt calibration, could recover magnitude information.
- **Naïve portfolio construction**: Weights are proportional or equal-weight top-N; a mean-variance / risk-parity optimiser over the joint prediction distribution (using the covariance-shrinkage machinery in the wiki) is not yet wired in.

---

## Future Directions

- **Alternative data integration**: Incorporate earnings surprise data, analyst sentiment, credit spreads, or options-derived implied volatility surfaces as additional predictive signals.
- **Sequence models**: Explore recurrent architectures (LSTM, Temporal Convolutional Networks) or transformer-based time-series models that capture long-range temporal dependencies without explicit feature engineering.
- **Bayesian hyperparameter optimisation**: Replace manual sector-specific tuning with automated optimisation via Optuna or Ray Tune with temporal cross-validation.
- **Ensemble stacking**: Combine LightGBM predictions with linear factor models (Fama-French 5-factor) via stacked generalisation to improve robustness to regime shifts.
- **Portfolio construction layer**: Apply predicted returns as inputs to a mean-variance or Black-Litterman optimiser to generate actionable sector allocation weights.

---

## Visual References

All diagnostic plots are stored in the `plots/` directory:

- `Sector_Predictions/` — Actual vs predicted return time-series overlays
- `R²_By_Sector/` — Out-of-sample R² comparison bar chart
- `Feature_Importances/` — Per-sector top-20 feature importance charts
- `Price_Trends_&_Return_Distributions/` — Historical price trends and return distribution box plots
- `Returns_Correlation_Heatmap/` — Cross-sectional Pearson correlation matrix
- `Sharpe_Ratio_By_Sector/` — Risk-adjusted return comparison
