# Methodology Enhancements — Leakage Elimination, Walk-Forward Validation, and Portfolio Backtesting

This report documents the rigorous methodological upgrade of the sector ETF
forecasting pipeline, transforming it from a *concurrent historical reconstruction*
(which silently leaked future information) into a valid *forward-looking predictive
system*. Each enhancement is grounded in the project's quantitative-finance knowledge
base; the governing reference is the synthesis note **"LightGBM Sector ETF Pipeline —
Leakage Audit and Theoretical Context."**

All logic described here is implemented once in
`notebooks/LGBM_model_functions/LGBM_functions.py` and imported by every driver
script, so the leakage-sensitive code has a single audited source.

---

## 1. Core Structural Enhancements

### 1.1 Mathematical Realignment of Features and Targets

Let `R_t` denote the vector of daily percentage returns across all ETFs on trading day `t`.

**Feature lag isolation (t − 1).** Every rolling feature is a function of information
available *strictly before* the decision point:

```
X_t = f(R_{t-1}, R_{t-2}, …, R_{t-n})
```

Seven rolling features in the original `generate_features` (`_rolling_mean_10`,
`_rolling_std_10`, SPY `_corr`/`_cov`, `_cumulative_return`, `_max_drawdown`, and the
pairwise sector `_corr`) used the day-`t` return inside a window ending at `t`, which
is unknowable at decision time. All are now lagged with `.shift(1)`. This lag
discipline is the standard convention in the realized-volatility literature, where
every estimator `RV_t` is an end-of-day quantity used to forecast `RV_{t+1}` (wiki:
*Realized Volatility*, *HAR Model*, *Volatility Forecasting Combinations…*).

**Target lead alignment (t + k).** The regression target is the **k-day forward**
average return (`k = 5`):

```
y_t = (1/k) · Σ_{i=1..k} R_{t+i}  =  returns.rolling(k).mean().shift(-k)
```

The original target `returns.rolling(5).mean()` (without `.shift(-5)`) was a
*backward-looking* mean of the past — the model was reconstructing history, not
forecasting. The forward-return convention (`features at t → return at t+1…t+k`) is
formalized in the wiki notes *Conditional Autoencoder* (`E_t[M_{t+1} r_{i,t+1}] = 0`)
and *Can a Machine Learn from Behavioral Biases*.

### 1.2 Prevention of Data-Leakage in Feature Scaling

A globally-fit `StandardScaler` computes `μ` and `σ` from the entire dataset,
including the test period, and then scales the training set — leaking test-period
moments into training. The fix fits the scaler on each fold's training rows only:

```
μ_train, σ_train  ←  from X_train only
X_train_scaled = (X_train − μ_train) / σ_train
X_test_scaled  = (X_test  − μ_train) / σ_train      # train statistics only
```

Grounding: *Walk-Forward Backtesting* (the optimizer sees only data available at each
retraining date) and *Markov Cross-Validation* Criterion 3 (train/test independence).

**Residual leakage also fixed:** PCA on the autoregressive lag block was previously
fit on the full series before any fold boundary. It is now fit on the *training* lag
block within each fold and applied to the held-out block (`_fold_pca_transform`).

---

## 2. Walk-Forward Validation Architecture

A single static 2010–2020 / 2021–2025 split yields one metric of unknown variance and
cannot separate genuine skill from a favorable test period. It is replaced by an
**expanding-window walk-forward** scheme (`sklearn.TimeSeriesSplit`, 5 folds):

```
for fold j = 1..M:
    Train = {1 … T_j},   Test = {T_j+1 … T_j+Δ}
    fit scaler + PCA on Train only;  fit LGBMRegressor;  predict Test block
combine out-of-fold predictions → global OOS arrays → R², RMSE, Sharpe
```

Grounding: *Markov Cross-Validation* proves why standard k-fold fails for
autocorrelated series (three criteria) and that the CV-estimator variance scales as
`Var(ê)/2m` — more folds tighten the estimate. `TimeSeriesSplit` is the sklearn
BCV-style blocked implementation, sufficient to eliminate leakage. `min(p/3…)`
partition rules and distance bounds are detailed in *Markov Cross-Validation for Time
Series Model Evaluations (Formulas)*.

---

## 3. Feature-Space Expansion

### 3.1 Stationarity and Macro-Regime Preprocessing

- **252-day rolling z-score** for macro indicators. Raw VIX/TNX *levels* are trending
  and non-stationary, distorting tree splits. Standardizing over a rolling 1-year
  baseline, `z_t = (x_t − mean_N(x)) / std_N(x)`, isolates local regime shocks. (Prior
  revisions used a 60-day window; the framework specifies `N = 252`.)
- **Augmented Dickey-Fuller (ADF) screening.** Every non-binary, non-calendar feature
  is tested for a unit root (H₀: non-stationary). Features failing at the 95% level
  (`p > 0.05`) are passed through a first-difference transform `Δz_t = z_t − z_{t-1}`
  before entering the model. The *decision* of which columns to difference is made on
  the earliest training block only (leakage-free); first-differencing itself is a
  strictly causal operation. Grounding: *GARCH Models* / financial-econometrics
  treatment of non-stationarity and unit roots. (Requires `statsmodels`; the filter
  degrades gracefully to a no-op if unavailable.)

### 3.2 Advanced Cross-Sector Interaction Signals

- **Sector spread dynamics:** `Spread_{t-1} = X_{target,t-1} − X_{peer,t-1}` for every
  peer sector — an explicit relative-strength / lead-lag rotation signal, richer than
  a raw pairwise correlation.
- **Rolling 20-day asset beta:** `β_{i,Market} = Cov(R_i, R_SPY) / Var(R_SPY)`,
  decomposing systematic vs idiosyncratic return, retained and lagged.

### 3.3 Simulated Portfolio Backtest Generation

The model's out-of-fold predictions `ŷ` are converted into a tradeable cross-sectional
strategy (`notebooks/portfolio_backtest/`):

```
Proportional:     w_{i,t} = ŷ_{i,t} / Σ_j |ŷ_{j,t}|        (dollar-neutral, gross 1)
Top-N Long-Short: +1/N on the top-N, −1/N on the bottom-N   (zero-cost book)
```

Weights formed at `t` are applied to the **next-day** realized return
(`returns.shift(-1)`), so the P&L is strictly out-of-sample and driven purely by
walk-forward allocations. Using the 1-day-ahead return as the tradeable proxy avoids
the overlapping-window autocorrelation that a raw 5-day forward P&L would introduce.
Output: an **un-leaked walk-forward trading Sharpe** (annualized ×√252), annualized
return/volatility, and maximum drawdown per scheme, plus a cumulative equity curve.
Grounding: *Sharpe Ratio* (risk-adjusted objective, √252 annualization),
*Mean-Variance Optimization* / *Tangency Portfolio* (long-short isolates
cross-sectional selection skill from market beta), *Maximum Drawdown* (tail risk the
Sharpe ratio ignores).

---

## 3bis. Hybrid Directional Classification with Continuous R² Mapping

The engine was migrated from `LGBMRegressor` to **`LGBMClassifier`**
(`train_evaluate_hybrid`). The optimisation target is the binary directional signal
`y_signal = 1[y_continuous > 0]`, because on noisy short-horizon returns the *sign* is
far more learnable than the *magnitude*. To preserve a magnitude-aware diagnostic, each
fold's predicted probability of an up-move is mapped back to the return scale:

```
ŷ_continuous = (P(up) − 0.5) · 2 · vol_t
```

where `vol_t` is the raw (unscaled) rolling volatility of the target sector — identical
to the `{ticker}_vol` feature — so a probability of 1 predicts +vol, 0 predicts −vol,
and 0.5 predicts 0. The continuous out-of-sample R² is the standard regression score of
`ŷ_continuous` against the realised forward return. All leakage controls (per-fold PCA,
per-fold scaler, ADF differencing, pruning) are inherited unchanged.

**Reported metrics and the honest R² story.** The pipeline reports accuracy and ROC-AUC
(the classifier's native metrics), the annualised directional Sharpe, and the mapped R².
On the live 2008–2025 run the mapped R² is **negative for every sector (mean ≈ −1.83)**.
This is expected and is reported rather than hidden: `(P−0.5)·2·vol` is a deliberately
simple, uncalibrated transform that over-states magnitude, so as a *point* estimator of
return size it is worse than the mean. A classifier's skill is directional — mean
accuracy ≈ 0.545 and mean AUC ≈ 0.520, a small but consistent edge over the 0.50
baseline. The negative R² quantifies the (known) weakness of the mapping, not a failure
of the model.

## 3ter. Two-Stage Non-Blocking Plotting Architecture

Rendering was decoupled into two stages so the pipeline never freezes on a blocking GUI
call:

- **Stage 1 — headless asset generation (throughout).** Every plot helper builds its
  figure, writes it to `plots/<category>/…` via `savefig`, and immediately `plt.close()`s
  it. This flushes memory and keeps the execution thread moving. No helper calls
  `plt.show()`.
- **Stage 2 — simultaneous batch display (at the very end).** After all computation,
  the orchestrator calls `plt.ion()`, re-renders every diagnostic with `close=False`, and
  finishes with `plt.show(block=True)` so that, under an interactive backend, all windows
  populate at once without blocking earlier work. Under a headless backend (Agg — used
  for the reproducible run) Stage 2 is a harmless no-op and the Stage-1 PNGs are the
  durable output.

---

## 4. Verification

The pipeline is validated by a synthetic end-to-end test harness (no network) that
asserts: forward-target alignment, look-ahead-free feature construction, weight-scheme
invariants (proportional gross = 1; long-short net = 0, gross = 2), backtest metric
sanity, ADF discrimination (white-noise stationary, random-walk non-stationary), and a
full walk-forward LightGBM run with per-fold PCA. The decisive check is the **leakage
guard**: a pure-noise future target must yield OOS R² ≈ 0 (observed ≈ −0.11) for the
regressor path, and for the hybrid classifier a shuffled/noise target must yield
**AUC ≈ 0.50** (observed ≈ 0.49) — no ability to rank a random signal. Because a leaking
pipeline would "predict" even random targets, these null results confirm the absence of
look-ahead leakage.

---

## Knowledge-Base References

| Wiki page | Role |
|-----------|------|
| LightGBM Sector ETF Pipeline — Leakage Audit and Theoretical Context | Governing synthesis; maps each failure to theory |
| Walk-Forward Backtesting | Structural validation principle |
| Markov Cross-Validation (+ Formulas) | Why standard CV fails for time series; estimator variance |
| Realized Volatility · HAR Model | Lag discipline for rolling features |
| GARCH Models | Non-stationarity, unit roots, first-differencing |
| Sharpe Ratio · Mean-Variance Optimization · Maximum Drawdown | Portfolio backtest metrics |
| LightGBM · Gradient Boosting Decision Trees | Model theory (leaf-wise growth, GOSS, EFB) |
| Jane Street Prediction Model based on LightGBM | Closest analogous application |
| Conditional Autoencoder · Can a Machine Learn from Behavioral Biases | Forward-return target convention |
