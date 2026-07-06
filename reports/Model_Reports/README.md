# Model Report — ML-Driven Sector ETF Prediction

## Overview

**Objective**: Forecast 5-day forward rolling mean returns for nine S&P 500 GICS sector ETFs using a gradient boosted decision tree ensemble, with SPY serving as a broad-market benchmark for beta decomposition and systematic risk proxies.

**Approach**: A sector-specific LightGBM Regressor is trained on a high-dimensional feature matrix (~80+ engineered features per sector) using walk-forward time-series cross-validation to produce unbiased out-of-sample (OOS) performance estimates. The validation protocol enforces strict temporal ordering and per-fold feature normalisation to eliminate look-ahead and global-scaler data leakage.

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
| PCA-compressed lags | 10 autoregressive lags compressed into 5 orthogonal principal components via PCA, eliminating multicollinearity |
| Technical indicators | Rolling mean/std, coefficient of variation (volatility ratio), exponentially weighted mean/std (EWM) |
| Systematic risk proxies | Rolling Pearson correlation and covariance with SPY; rolling beta (cov/var); idiosyncratic market deviation |
| Maximum drawdown | Rolling 20-day maximum drawdown — non-linear downside risk measure |
| Cross-sectional momentum | Equal-weighted mean and volatility of all other sector returns; their ratio as a sector Sharpe proxy |
| Macroeconomic regime features | VIX and TNX z-scores (60-day rolling normalisation); binary stress regime flag (VIX z-score > +1σ); lags 1–5 for delayed transmission |
| Interaction terms | Products of lagged momentum and rolling volatility — non-linear signal amplification |
| Pairwise sector correlations | 15-day rolling Pearson correlations with all other sectors — contagion and rotation dynamics |
| Calendar features | Month, ISO week-of-year, day-of-week, month-start/end flags — seasonality effects |

---

## Model Architecture

**Algorithm**: LightGBM Regressor — a histogram-based gradient boosted decision tree framework implementing leaf-wise (best-first) tree growth, which splits the highest-gain leaf at each step rather than growing all leaves at a fixed depth. This enables more asymmetric tree structures and faster convergence relative to depth-wise frameworks.

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

**Validation**: `TimeSeriesSplit` with 5 folds. `StandardScaler` (zero-mean, unit-variance normalisation) is refit on each training fold exclusively — fitting on the full dataset before splitting would leak test-period distributional statistics into training, inflating OOS metrics.

**Multicollinearity pruning**: Features with pairwise Pearson |r| > 0.95 are removed from the upper triangular correlation matrix before training.

---

## Performance Metrics (Out-of-Sample)

> ⚠️ **These figures predate the leakage-elimination upgrade.** They were generated
> under a methodology with a backward-looking target, a global scaler, a globally-fit
> PCA, and a single static split — all of which inflate apparent out-of-sample skill.
> They are retained here only as a historical baseline and **must be regenerated** with
> the corrected leakage-free walk-forward pipeline (see
> `../Methodology_Enhancements/`). Expect materially lower, more realistic R² values.

| Sector | R² | RMSE | MAE |
|--------|-----|------|-----|
| XLK | 0.7940 | 0.003030 | 0.002180 |
| XLF | 0.8670 | 0.002030 | 0.001410 |
| XLV | 0.8770 | 0.001400 | 0.001030 |
| XLI | 0.8510 | 0.001870 | 0.001330 |
| XLP | 0.8020 | 0.001550 | 0.001130 |
| XLB | 0.8530 | 0.002080 | 0.001510 |
| XLU | 0.8140 | 0.002130 | 0.001570 |
| XLRE | 0.8430 | 0.002190 | 0.001610 |
| XLY | 0.8410 | 0.002690 | 0.001990 |

**Average R² across sectors: 0.838**  
**Average RMSE: 0.002110**  
**Average MAE: 0.001530**

---

## Key Insights

- **Best predictive accuracy**: XLV (Healthcare) achieves R² = 0.877 — consistent with its stable beta, low-kurtosis return distribution, and predictable autocorrelation structure relative to macro shocks.
- **Most challenging sector**: XLK (Technology) has the lowest R² = 0.794, driven by episodic momentum crashes, high return kurtosis, and non-linear sensitivity to rate-driven valuation repricing.
- **Model calibration**: The absence of a systematic R²/RMSE trade-off across sectors, and the alignment of OOS metrics with expected sector dynamics, suggests the model is neither severely overfit nor underfit.
- **Directional strategy**: The sign-based directional Sharpe ratio exceeds 1.0 for most sectors in the walk-forward framework, indicating that predicted return direction is economically exploitable beyond a naïve long-only benchmark.

---

## Limitations

- **Regime non-stationarity**: The model implicitly assumes that the statistical relationships between features and forward returns are stable across macroeconomic regimes. Structural breaks (e.g. zero-interest-rate regime transitions) can cause feature-return relationships to shift, degrading OOS performance.
- **Stationarity assumption**: Linear feature construction (rolling means, correlations) assumes weakly stationary return series. Heavy-tailed distributions and volatility clustering (ARCH/GARCH effects) in financial returns violate classical stationarity assumptions.
- **Transaction cost abstraction**: Reported Sharpe ratios do not account for bid-ask spreads, market impact, or short-selling costs — real-world implementation would reduce net alpha.
- **Single-asset prediction**: The model predicts sector-level ETF returns in isolation; portfolio-level optimisation (mean-variance, risk parity) over the joint prediction distribution is not implemented.

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
