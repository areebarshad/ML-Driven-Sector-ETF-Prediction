"""
LGBM_functions.py — Core library for the ML-Driven Sector ETF Prediction pipeline.

This module is the single source of truth for feature engineering, leakage-free
walk-forward model training, and portfolio backtesting. All driver scripts import
from here rather than re-inlining the logic, which prevents the divergent-copy
bugs that reintroduced data leakage in earlier revisions.

Theoretical grounding (see the project's QuantFinance wiki):
  - Feature/target temporal alignment and the .shift(1) lag discipline are the
    standard convention in the realized-volatility and HAR-model literature: a
    feature at row t must be a function of R_{t-1}, R_{t-2}, ... only.
  - Per-fold scaling and per-fold PCA eliminate the two forms of look-ahead
    leakage (global-scaler and global-PCA) — the optimizer sees only information
    available at each retraining date (Walk-Forward Backtesting principle).
  - TimeSeriesSplit is the sklearn BCV-style blocked CV that satisfies the
    train/test-independence criteria of Markov Cross-Validation for time series.
  - ADF stationarity screening + first-differencing addresses the non-stationarity
    of raw macro level series (GARCH / financial-econometrics literature).
  - The Sharpe ratio (annualized by sqrt(252)) is the risk-adjusted objective; the
    cross-sectional long-short backtest converts raw predictions into a tradeable,
    un-leaked P&L series.
"""

import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# The Augmented Dickey-Fuller test requires statsmodels. It is an optional
# dependency: if unavailable, the stationarity filter degrades gracefully to a
# no-op with a warning, so the rest of the pipeline still runs.
try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


# ── Global Configuration ────────────────────────────────────────────────────────

TICKERS = ['XLK', 'XLF', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLY', 'SPY']

# Non-benchmark sectors (SPY is used only as the systematic-risk reference asset).
SECTORS = [t for t in TICKERS if t != 'SPY']

START_DATE = '2008-01-01'
END_DATE = '2025-12-31'

# Per-sector LightGBM tree-complexity hyperparameters, calibrated to each sector's
# return autocorrelation structure and signal-to-noise ratio. LightGBM grows trees
# leaf-wise, so num_leaves is the primary complexity knob and must be capped to
# prevent overfitting on noisy financial return series.
SECTOR_PARAMS = {
    "XLK":  {"max_depth": 9,  "num_leaves": 64},   # Tech: momentum-driven, trendy
    "XLF":  {"max_depth": 14, "num_leaves": 128},  # Financials: macro-sensitive, fat tails
    "XLV":  {"max_depth": 10, "num_leaves": 72},   # Healthcare: defensive, low-beta regime
    "XLP":  {"max_depth": 7,  "num_leaves": 48},   # Consumer Staples: near-stationary, smooth
    "XLI":  {"max_depth": 12, "num_leaves": 96},   # Industrials: cyclical, noisy residuals
    "XLB":  {"max_depth": 11, "num_leaves": 80},   # Materials: commodity-linked, fat tails
    "XLU":  {"max_depth": 6,  "num_leaves": 32},   # Utilities: mean-reverting, minimal variance
    "XLRE": {"max_depth": 10, "num_leaves": 64},   # Real Estate: duration-sensitive to rate shocks
    "XLY":  {"max_depth": 10, "num_leaves": 80},   # Consumer Discretionary: income-cycle exposure
    "SPY":  {"max_depth": 10, "num_leaves": 72},   # Broad-market benchmark
}

# Baseline LightGBM estimator parameters shared across sectors. L1/L2 regularization
# and stochastic row/column subsampling decorrelate the boosted trees and control
# variance on noisy targets.
LGBM_BASE_PARAMS = dict(
    n_estimators=1000, learning_rate=0.01, verbose=-1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.5, random_state=42,
)

FORWARD_HORIZON = 5   # k-day forward return prediction horizon
ROLLING_WINDOW = 10   # default rolling-statistic lookback
N_LAGS = 10           # number of autoregressive lags fed to per-fold PCA
N_PCA = 5             # principal components retained from the lag block
MACRO_Z_WINDOW = 252  # 1-year rolling window for macro z-score normalization
ADF_SIGNIF = 0.05     # ADF p-value threshold; p > 0.05 => difference the feature


# ── Path Helpers ────────────────────────────────────────────────────────────────

def get_data_dir():
    """Return the absolute path to the project-level data/ directory, creating it."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_plots_dir(subfolder=None):
    """Return an absolute path inside the project-level plots/ directory."""
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots'))
    if subfolder:
        plots_dir = os.path.join(plots_dir, subfolder)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


# ── Data Ingestion ──────────────────────────────────────────────────────────────

def download_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, save=True):
    """
    Download adjusted closing prices and optionally persist to data/sp500_sector_prices.csv.

    The 2008-2025 window spans multiple macro regimes (GFC, COVID-19 shock,
    post-ZIRP tightening), providing the regime diversity required for the
    walk-forward evaluation to be informative about generalization.
    """
    data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    if save:
        data.to_csv(os.path.join(get_data_dir(), 'sp500_sector_prices.csv'))
    return data


def load_returns(path=None):
    """Load prices and return daily arithmetic returns (first-order differencing)."""
    if path is None:
        path = os.path.join(get_data_dir(), 'sp500_sector_prices.csv')
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    return data.pct_change().dropna()


# Module-level cache so the (expensive, network-bound) macro download runs once
# per process rather than once per sector inside generate_features.
_MACRO_CACHE = {}


def download_macro_zscores(start=START_DATE, end=END_DATE, window=MACRO_Z_WINDOW):
    """
    Download VIX (implied volatility) and TNX (10-year Treasury yield) and convert
    them to feature-ready rolling z-scores.

    Raw macro *levels* are strongly trending / autocorrelated and non-stationary,
    which distorts decision-tree splits. Standardizing over a rolling 1-year
    (252-day) baseline extracts local regime *shocks* rather than nominal levels:

        z_t = (x_t - mean_{t-N+1..t}(x)) / std_{t-N+1..t}(x)

    The returned series are lagged by one day (.shift(1)) so that the feature at
    row t uses only information through t-1. A binary volatility-regime flag marks
    stress states where the VIX z-score exceeds +1 sigma.
    """
    key = (start, end, window)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key].copy()

    vix_raw = yf.download("^VIX", start=start, end=end, progress=False)['Close']
    tnx_raw = yf.download("^TNX", start=start, end=end, progress=False)['Close']

    def _zscore(x):
        return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-6)

    macro = pd.DataFrame(index=vix_raw.index)
    macro['vix_zscore'] = _zscore(vix_raw).shift(1)
    macro['tnx_zscore'] = _zscore(tnx_raw).shift(1)
    macro['market_vol_regime'] = (macro['vix_zscore'] > 1).astype(int)

    # Lagged macro features (lags 1-5) capture delayed transmission of macro shocks.
    for lag in range(1, 6):
        macro[f'vix_zscore_lag{lag}'] = macro['vix_zscore'].shift(lag)
        macro[f'tnx_zscore_lag{lag}'] = macro['tnx_zscore'].shift(lag)

    _MACRO_CACHE[key] = macro
    return macro.copy()


# ── Stationarity Screening (ADF) ────────────────────────────────────────────────

def adf_is_stationary(series, signif=ADF_SIGNIF):
    """
    Augmented Dickey-Fuller test. Returns True if the null of a unit root is
    rejected at the given significance level (i.e. the series is stationary).

    H0: series has a unit root (non-stationary). Reject when p-value <= signif.
    If statsmodels is unavailable, conservatively returns True (no differencing).
    """
    if not _HAS_STATSMODELS:
        return True
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    # ADF needs sufficient observations and non-constant input.
    if s.shape[0] < 20 or s.nunique() <= 1:
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalue = adfuller(s, autolag='AIC')[1]
        return bool(pvalue <= signif)
    except Exception:
        return True


def select_nonstationary_columns(features, signif=ADF_SIGNIF, exclude=None):
    """
    Screen every candidate feature column with the ADF test and return the list of
    columns that FAIL stationarity (p > signif) and should be first-differenced.

    Binary / calendar / dummy columns are excluded from screening — differencing
    them is meaningless.
    """
    exclude = set(exclude or [])
    # Never difference obviously discrete/seasonal columns.
    skip_tokens = ('month', 'week_of_year', 'day_of_week', 'is_month',
                   'market_vol_regime', 'sector_')
    to_difference = []
    for col in features.columns:
        if col in exclude or any(tok in col for tok in skip_tokens):
            continue
        if features[col].nunique() <= 2:  # binary indicator
            continue
        if not adf_is_stationary(features[col], signif=signif):
            to_difference.append(col)
    return to_difference


def apply_first_difference(features, columns):
    """
    Apply a first-difference transform (delta z_t = z_t - z_{t-1}) to the given
    columns in place on a copy. First-differencing is a strictly causal operation
    (uses t and t-1 only), so it introduces no look-ahead; only the *decision* of
    which columns to difference must be made on training data.
    """
    out = features.copy()
    for col in columns:
        out[col] = out[col].diff()
    return out


# ── Feature Engineering ─────────────────────────────────────────────────────────

def generate_features(returns, target_ticker, tickers=TICKERS,
                      macro_z=None, rolling_window=ROLLING_WINDOW, n_lags=N_LAGS):
    """
    Construct the look-ahead-free feature matrix for a single target sector ETF.

    Every feature is lagged by at least one trading day. Unlike earlier revisions,
    the raw autoregressive lag columns (lag_1..lag_n) are RETAINED here rather than
    PCA-transformed in place — PCA is fitted per-fold inside train_evaluate_model to
    avoid fitting the projection on future data.

    Feature groups:
      - Rolling statistical moments (mean, std, skew, kurtosis)
      - Momentum and momentum-to-volatility ratio
      - Raw autoregressive lags (lag_1..lag_n) for downstream per-fold PCA
      - Rolling SPY correlation / covariance, rolling beta, market deviation
      - Rolling cumulative return and maximum drawdown
      - Cross-sectional sector momentum / volatility / Sharpe
      - Exponentially weighted mean / std (adaptive smoothing)
      - Cross-sector return SPREAD dynamics: target vs each peer and vs SPY
      - 252-day macro z-scores (VIX, TNX) + lags + volatility-regime flag
      - Pairwise sector cross-correlations
      - Calendar / seasonality features
    """
    lagged = returns.shift(1)
    f = pd.DataFrame(index=returns.index)

    # Rolling statistical moments (on lagged returns => strictly historical)
    f[f'{target_ticker}_mean'] = lagged[target_ticker].rolling(rolling_window).mean()
    f[f'{target_ticker}_vol']  = lagged[target_ticker].rolling(rolling_window).std()
    f[f'{target_ticker}_skew'] = lagged[target_ticker].rolling(rolling_window).skew()
    f[f'{target_ticker}_kurt'] = lagged[target_ticker].rolling(rolling_window).kurt()
    f[f'{target_ticker}_momentum'] = (
        lagged[target_ticker] - lagged[target_ticker].rolling(rolling_window).mean()
    )
    f[f'{target_ticker}_momentum_ratio'] = (
        f[f'{target_ticker}_momentum'] / (f[f'{target_ticker}_vol'] + 1e-6)
    )

    # Raw autoregressive lags (kept for per-fold PCA compression downstream)
    for i in range(1, n_lags + 1):
        f[f'lag_{i}'] = returns[target_ticker].shift(i)

    # Technical indicators (all shifted by 1 — fixes the historical unlagged bug)
    f[f'{target_ticker}_rolling_mean_10'] = returns[target_ticker].rolling(10).mean().shift(1)
    f[f'{target_ticker}_rolling_std_10']  = returns[target_ticker].rolling(10).std().shift(1)
    f[f'{target_ticker}_volatility_ratio'] = (
        f[f'{target_ticker}_rolling_std_10'] / (f[f'{target_ticker}_rolling_mean_10'] + 1e-6)
    )

    # Systematic-risk proxies vs SPY (shifted)
    f[f'{target_ticker}_corr'] = returns[target_ticker].rolling(10).corr(returns['SPY']).shift(1)
    f[f'{target_ticker}_cov']  = returns[target_ticker].rolling(10).cov(returns['SPY']).shift(1)

    # Cumulative return and rolling maximum drawdown (shifted)
    f[f'{target_ticker}_cumulative_return'] = (
        returns[target_ticker].rolling(20).apply(np.sum).shift(1)
    )
    f[f'{target_ticker}_max_drawdown'] = (
        returns[target_ticker]
        .rolling(20)
        .apply(lambda x: np.min(x / (np.maximum.accumulate(x + 1e-6)) - 1))
        .shift(1)
    )

    # Cross-sectional sector momentum / volatility / Sharpe
    others = returns.drop(columns=[target_ticker])
    f['sector_momentum']   = others.mean(axis=1).shift(1)
    f['sector_volatility'] = others.std(axis=1).shift(1)
    f['sector_sharpe']     = f['sector_momentum'] / (f['sector_volatility'] + 1e-6)

    # Exponentially weighted moments (adaptive smoothing)
    f[f'{target_ticker}_ewm_mean'] = returns[target_ticker].ewm(span=10).mean().shift(1)
    f[f'{target_ticker}_ewm_std']  = returns[target_ticker].ewm(span=10).std().shift(1)

    # Rolling 20-day beta to SPY and idiosyncratic market deviation (shifted)
    f[f'{target_ticker}_rolling_beta'] = (
        returns[target_ticker].rolling(20).cov(returns['SPY']) /
        returns['SPY'].rolling(20).var()
    ).shift(1)
    f[f'{target_ticker}_market_dev'] = (
        (returns[target_ticker] - returns['SPY']).rolling(7).mean().shift(1)
    )

    # Momentum-volatility interaction terms (lags 1-5)
    for lag in range(1, 6):
        mom = returns[target_ticker].shift(lag) - returns[target_ticker].shift(lag + 5)
        f[f'{target_ticker}_lag{lag}_momentum'] = mom
        f[f'{target_ticker}_lag{lag}_mom_vol'] = mom * f[f'{target_ticker}_vol']

    # Cross-sector SPREAD dynamics: Spread_{t-1} = X_target,t-1 - X_peer,t-1.
    # Captures relative-strength / lead-lag rotation signals between sectors.
    for other in tickers:
        if other == target_ticker:
            continue
        f[f'{target_ticker}_{other}_spread'] = (
            returns[target_ticker].shift(1) - returns[other].shift(1)
        )
        # Pairwise 15-day rolling correlation (contagion / co-movement)
        f[f'{target_ticker}_{other}_corr'] = (
            returns[target_ticker].rolling(15).corr(returns[other]).shift(1)
        )

    f[f'{target_ticker}_returns_1d'] = returns[target_ticker].shift(1)

    # 252-day macro z-scores (VIX / TNX) joined on the date index
    if macro_z is None:
        macro_z = download_macro_zscores()
    f = f.join(macro_z)

    # Calendar / seasonality features
    f['month']          = f.index.month
    f['week_of_year']   = f.index.isocalendar().week.astype(int)
    f['day_of_week']    = f.index.dayofweek
    f['is_month_start'] = f.index.is_month_start.astype(int)
    f['is_month_end']   = f.index.is_month_end.astype(int)

    return f.dropna()


def make_forward_target(returns, target_ticker, horizon=FORWARD_HORIZON):
    """
    Construct the k-day FORWARD average-return target:

        y_t = (1/k) * sum_{i=1..k} R_{t+i}  =  returns.rolling(k).mean().shift(-k)

    The .shift(-k) is critical: it moves the k-day average forward so that the
    value at index t summarizes the FUTURE window [t+1, t+k]. Without it, the
    target is a backward-looking mean of the past and the model merely reconstructs
    history (the target-construction leakage fixed in this revision).
    """
    return returns[target_ticker].rolling(horizon).mean().shift(-horizon)


# ── Per-fold PCA (leakage-free lag compression) ─────────────────────────────────

def _fold_pca_transform(X_train, X_test, lag_cols, n_pca=N_PCA):
    """
    Fit PCA on the TRAINING fold's lag block only, then project both train and
    test lag blocks onto those components. This fixes the residual global-PCA
    leakage: previously PCA was fit on the entire lag matrix (including future
    rows) before any fold boundary existed.
    """
    present = [c for c in lag_cols if c in X_train.columns]
    if not present:
        return X_train.copy(), X_test.copy()

    k = min(n_pca, len(present))
    pca = PCA(n_components=k)
    train_comp = pca.fit_transform(X_train[present])
    test_comp = pca.transform(X_test[present])
    comp_cols = [f'lag_pca_{i}' for i in range(k)]

    Xtr = X_train.drop(columns=present).copy()
    Xte = X_test.drop(columns=present).copy()
    Xtr[comp_cols] = train_comp
    Xte[comp_cols] = test_comp
    return Xtr, Xte


# ── Walk-Forward Training & Evaluation ──────────────────────────────────────────

def train_evaluate_model(features, target, target_ticker, sector_params=None,
                        n_splits=5, n_pca=N_PCA, apply_adf=True, adf_signif=ADF_SIGNIF):
    """
    Train and evaluate a sector-specific LightGBM regressor with leakage-free
    walk-forward (expanding-window) cross-validation.

    Pipeline (all fold-dependent decisions use training data only):
      1. Align features/target on the common (dropna) index.
      2. Stationarity screening: decide which columns to first-difference using the
         EARLIEST training block (fold-0 train) so the decision never sees test
         data; apply the causal first-difference consistently across the series.
      3. Multicollinearity pruning (|Pearson r| > 0.95) decided on the fold-0 train
         block; raw PCA-lag columns are exempted (they are compressed per-fold).
      4. Append a one-hot sector dummy.
      5. For each of n_splits expanding folds:
           a. Fit PCA on the training lag block only; project train and test.
           b. Fit StandardScaler on training rows only; transform both.
           c. Fit LGBMRegressor; predict the held-out block.
           d. Accumulate out-of-fold (OOF) predictions.
      6. Report OOF metrics: R2, RMSE, MAE, and the annualized directional Sharpe
         of sign(pred) * actual.
      7. Refit on the full data for feature-importance reporting ONLY (never used
         for the reported OOS metrics).

    Returns a dict including OOF predictions/actuals indexed by date, enabling the
    cross-sectional portfolio backtest.
    """
    if sector_params is None:
        sector_params = SECTOR_PARAMS

    target = target.dropna()
    features = features.dropna()
    common = features.index.intersection(target.index)
    X = features.loc[common].copy()
    y = target.loc[common].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    first_train_idx = splits[0][0]
    X_first = X.iloc[first_train_idx]

    lag_cols = [f'lag_{i}' for i in range(1, N_LAGS + 1)]

    # (2) ADF stationarity screening decided on the earliest training block.
    if apply_adf and _HAS_STATSMODELS:
        diff_cols = select_nonstationary_columns(
            X_first, signif=adf_signif, exclude=set(lag_cols)
        )
        if diff_cols:
            X = apply_first_difference(X, diff_cols)
    elif apply_adf and not _HAS_STATSMODELS:
        warnings.warn("statsmodels not installed; skipping ADF stationarity filter.")

    # (3) Multicollinearity pruning decided on the earliest training block.
    X_first = X.iloc[first_train_idx].dropna()
    prune_candidates = [c for c in X_first.columns if c not in lag_cols]
    corr = X_first[prune_candidates].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=to_drop)

    # Re-align after differencing introduced leading NaNs.
    X = X.dropna()
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    # (4) Sector dummy.
    X['sector_dummy'] = 1.0

    oof_preds, oof_actuals, oof_index = [], [], []
    params = sector_params.get(target_ticker, {"max_depth": 10, "num_leaves": 64})

    for train_idx, test_idx in splits:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # (5a) Per-fold PCA on the lag block.
        X_tr, X_te = _fold_pca_transform(X_tr, X_te, lag_cols, n_pca=n_pca)

        # (5b) Per-fold scaling — train statistics only. Kept as DataFrames with
        # identical columns so LightGBM's fit/predict feature names stay consistent.
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

        # (5c) Fit / predict.
        model = LGBMRegressor(**LGBM_BASE_PARAMS, **params)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        oof_preds.extend(preds)
        oof_actuals.extend(y_te.values)
        oof_index.extend(y_te.index)

    # (6) Aggregated out-of-sample metrics.
    r2 = r2_score(oof_actuals, oof_preds)
    rmse = math.sqrt(mean_squared_error(oof_actuals, oof_preds))
    mae = mean_absolute_error(oof_actuals, oof_preds)
    strat = np.sign(oof_preds) * np.array(oof_actuals)
    sharpe = strat.mean() / (strat.std() + 1e-9) * math.sqrt(252)

    print(f"[{target_ticker}] R2 (OOS): {r2:.4f} | RMSE: {rmse:.6f} | "
          f"MAE: {mae:.6f} | Directional Sharpe: {sharpe:.4f}")

    # (7) Full-data refit for feature-importance reporting only.
    X_full_tr, _ = _fold_pca_transform(X, X, lag_cols, n_pca=n_pca)
    scaler_final = StandardScaler()
    X_full_s = pd.DataFrame(scaler_final.fit_transform(X_full_tr),
                            index=X_full_tr.index, columns=X_full_tr.columns)
    model_final = LGBMRegressor(**LGBM_BASE_PARAMS, **params)
    model_final.fit(X_full_s, y)
    importances_df = pd.DataFrame(
        {"Feature": X_full_tr.columns, "Importance": model_final.feature_importances_}
    )

    return {
        "model": model_final,
        "r2": r2, "rmse": rmse, "mae": mae, "sharpe": sharpe,
        "y_test": pd.Series(oof_actuals, index=oof_index),
        "y_pred": pd.Series(oof_preds, index=oof_index),
        "importances_df": importances_df,
    }


# ── Hybrid Classification Engine with Continuous R² Mapping ──────────────────────

def train_evaluate_hybrid(features, y_continuous, target_ticker, vol_series,
                          sector_params=None, n_splits=5, n_pca=N_PCA,
                          apply_adf=True, adf_signif=ADF_SIGNIF):
    """
    Train a directional LightGBM CLASSIFIER while retaining a continuous out-of-sample
    R² by mapping predicted probabilities back to the return scale.

    Rationale (see reports/Methodology_Enhancements): directional sign is easier to
    learn than magnitude on noisy short-horizon returns, so the optimization target is
    the binary signal  y_signal = 1[y_continuous > 0]. To keep a magnitude-aware
    regression diagnostic, each fold's predicted probability of a positive return is
    mapped to a volatility-scaled continuous prediction:

        y_pred_continuous = (P(up) - 0.5) * 2 * vol_t

    where vol_t is the RAW (unscaled) rolling volatility of the target sector at time t
    (same definition as the {ticker}_vol feature). A probability of 1 predicts +vol,
    0 predicts -vol, 0.5 predicts 0 — dimensionally consistent with realized returns.

    The continuous R² is then the standard regression score of the mapped predictions
    against the actual continuous forward returns:

        R²_OOS = 1 - SS_res / SS_tot

    All leakage controls from train_evaluate_model are preserved (per-fold PCA,
    per-fold scaler, ADF differencing and multicollinearity pruning decided on the
    earliest training block). Classification-native metrics (accuracy, ROC-AUC) and the
    annualized directional Sharpe are reported alongside R², because a heuristic
    probability→return mapping generally yields a low continuous R² even when
    directional skill (accuracy/AUC) is real.

    Returns a dict whose 'y_pred' is the continuous mapped prediction, so the
    cross-sectional portfolio backtest can consume it exactly like the regressor path.
    """
    if sector_params is None:
        sector_params = SECTOR_PARAMS

    y_continuous = y_continuous.dropna()
    features = features.dropna()
    common = features.index.intersection(y_continuous.index)
    X = features.loc[common].copy()
    y_cont = y_continuous.loc[common].copy()
    # Binary directional signal target used for classifier optimization.
    y_signal = (y_cont > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    first_train_idx = splits[0][0]
    X_first = X.iloc[first_train_idx]
    lag_cols = [f'lag_{i}' for i in range(1, N_LAGS + 1)]

    # (2) ADF stationarity screening decided on the earliest training block.
    if apply_adf and _HAS_STATSMODELS:
        diff_cols = select_nonstationary_columns(X_first, signif=adf_signif, exclude=set(lag_cols))
        if diff_cols:
            X = apply_first_difference(X, diff_cols)
    elif apply_adf and not _HAS_STATSMODELS:
        warnings.warn("statsmodels not installed; skipping ADF stationarity filter.")

    # (3) Multicollinearity pruning decided on the earliest training block.
    X_first = X.iloc[first_train_idx].dropna()
    prune_candidates = [c for c in X_first.columns if c not in lag_cols]
    corr = X_first[prune_candidates].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=to_drop)

    # Re-align after differencing introduced leading NaNs.
    X = X.dropna()
    common = X.index.intersection(y_cont.index)
    X, y_cont, y_signal = X.loc[common], y_cont.loc[common], y_signal.loc[common]
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    # (4) Sector dummy.
    X['sector_dummy'] = 1.0

    # Raw volatility aligned to the sample (return-scale mapping factor).
    vol_aligned = vol_series.reindex(X.index).ffill().bfill()

    oof_prob, oof_cont_pred, oof_cont_actual, oof_signal, oof_index = [], [], [], [], []
    params = sector_params.get(target_ticker, {"max_depth": 10, "num_leaves": 64})

    for train_idx, test_idx in splits:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_sig_tr = y_signal.iloc[train_idx]
        y_cont_te = y_cont.iloc[test_idx]
        y_sig_te = y_signal.iloc[test_idx]

        # (5a) Per-fold PCA on the lag block.
        X_tr, X_te = _fold_pca_transform(X_tr, X_te, lag_cols, n_pca=n_pca)

        # (5b) Per-fold scaling — train statistics only.
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

        # (5c) Fit classifier; extract probability of a positive return.
        model = LGBMClassifier(**LGBM_BASE_PARAMS, **params)
        model.fit(X_tr_s, y_sig_tr)
        prob_positive = model.predict_proba(X_te_s)[:, 1]

        # Map probability centered at 0 back to the volatility-scaled return scale.
        vol_te = vol_aligned.iloc[test_idx].values
        y_pred_cont = (prob_positive - 0.5) * 2.0 * vol_te

        oof_prob.extend(prob_positive)
        oof_cont_pred.extend(y_pred_cont)
        oof_cont_actual.extend(y_cont_te.values)
        oof_signal.extend(y_sig_te.values)
        oof_index.extend(y_cont_te.index)

    oof_prob = np.asarray(oof_prob)
    oof_cont_pred = np.asarray(oof_cont_pred)
    oof_cont_actual = np.asarray(oof_cont_actual)
    oof_signal = np.asarray(oof_signal)

    # (6) Hybrid out-of-sample metrics.
    r2 = r2_score(oof_cont_actual, oof_cont_pred)          # continuous R² from mapping
    rmse = math.sqrt(mean_squared_error(oof_cont_actual, oof_cont_pred))
    mae = mean_absolute_error(oof_cont_actual, oof_cont_pred)
    pred_class = (oof_prob > 0.5).astype(int)
    accuracy = accuracy_score(oof_signal, pred_class)
    try:
        auc = roc_auc_score(oof_signal, oof_prob)
    except ValueError:
        auc = float('nan')
    strat = np.sign(oof_prob - 0.5) * oof_cont_actual
    sharpe = strat.mean() / (strat.std() + 1e-9) * math.sqrt(252)

    print(f"[{target_ticker}] R2 (OOS): {r2:.4f} | Accuracy: {accuracy:.4f} | "
          f"AUC: {auc:.4f} | Directional Sharpe: {sharpe:.4f}")

    # (7) Full-data refit for feature-importance reporting only.
    X_full_tr, _ = _fold_pca_transform(X, X, lag_cols, n_pca=n_pca)
    scaler_final = StandardScaler()
    X_full_s = pd.DataFrame(scaler_final.fit_transform(X_full_tr),
                            index=X_full_tr.index, columns=X_full_tr.columns)
    model_final = LGBMClassifier(**LGBM_BASE_PARAMS, **params)
    model_final.fit(X_full_s, y_signal)
    importances_df = pd.DataFrame(
        {"Feature": X_full_tr.columns, "Importance": model_final.feature_importances_}
    )

    return {
        "model": model_final,
        "r2": r2, "rmse": rmse, "mae": mae,
        "accuracy": accuracy, "auc": auc, "sharpe": sharpe,
        "y_test": pd.Series(oof_cont_actual, index=oof_index),
        "y_pred": pd.Series(oof_cont_pred, index=oof_index),
        "prob": pd.Series(oof_prob, index=oof_index),
        "importances_df": importances_df,
    }


# ── Portfolio Backtest (cross-sectional, un-leaked) ─────────────────────────────

def build_weights(pred_df, scheme='proportional', top_n=3):
    """
    Convert a cross-sectional matrix of OOF predictions (index = date, columns =
    sectors) into portfolio weights.

      - 'proportional': w_{i,t} = yhat_{i,t} / sum_j |yhat_{j,t}|  (dollar-neutral
        long-short with unit gross exposure).
      - 'long_short': equal-weight long the top_n predicted sectors and short the
        bottom_n, weights +/- 1/top_n (zero-cost long-short).
    """
    if scheme == 'proportional':
        denom = pred_df.abs().sum(axis=1).replace(0, np.nan)
        w = pred_df.div(denom, axis=0).fillna(0.0)
        return w

    if scheme == 'long_short':
        n_cols = pred_df.shape[1]
        top_n = min(top_n, n_cols // 2) if n_cols >= 2 else 1
        ranks = pred_df.rank(axis=1, ascending=False, method='first')
        longs = (ranks <= top_n).astype(float)
        shorts = (ranks > n_cols - top_n).astype(float)
        return (longs - shorts) / top_n

    raise ValueError(f"Unknown scheme: {scheme!r}")


def backtest_portfolio(weights, realized, rf=0.0, periods_per_year=252):
    """
    Backtest a daily-rebalanced cross-sectional strategy.

    weights_{i,t} are applied to the NEXT-day realized return realized_{i,t} (which
    must already be forward-aligned, e.g. returns.shift(-1)), so the P&L is strictly
    out-of-sample: the position formed from information available at t earns the
    return realized over (t, t+1]. Long-short books are financed internally, so the
    default risk-free rate is 0.

    Returns strategy return series, cumulative equity curve, annualized Sharpe,
    annualized return/volatility, and maximum drawdown.
    """
    common = weights.index.intersection(realized.index)
    w = weights.loc[common]
    r = realized.loc[common, w.columns]

    strat = (w * r).sum(axis=1)
    excess = strat - rf / periods_per_year
    sharpe = excess.mean() / (strat.std() + 1e-12) * math.sqrt(periods_per_year)
    cum = (1 + strat).cumprod()
    ann_ret = strat.mean() * periods_per_year
    ann_vol = strat.std() * math.sqrt(periods_per_year)
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "returns": strat, "cumulative": cum, "sharpe": sharpe,
        "ann_return": ann_ret, "ann_vol": ann_vol, "max_drawdown": max_dd,
    }


# ── Plotting (two-stage: Stage 1 saves headless, Stage 2 displays batched) ──────
#
# Every plot helper builds a figure, optionally writes it to disk (save_path), and
# either closes it immediately (close=True — Stage 1 headless asset generation, keeps
# the pipeline non-blocking) or leaves it open (close=False — Stage 2 simultaneous
# batch display via plt.ion()/plt.show(block=True) at the end of the main script).
# No helper calls plt.show() itself; display is centralized in the orchestrator.

def plot_feature_importances(importances_df, target_ticker, save_path=None, close=True):
    fig = plt.figure(figsize=(10, 6))
    top = importances_df.sort_values(by='Importance', ascending=False).head(20)
    plt.barh(top['Feature'], top['Importance'], color='teal')
    plt.title(f'Top 20 Feature Importances - {target_ticker}')
    plt.xlabel('Importance (split gain)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    if close:
        plt.close(fig)
    return fig


def plot_actual_vs_predicted(y_test, y_pred, target_ticker, title_note='',
                             save_path=None, close=True):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='crimson', alpha=0.7, linewidth=2)
    plt.plot(y_test.index, y_pred.values, label='Predicted (mapped)', color='royalblue',
             alpha=0.7, linewidth=2)
    plt.title(f'{target_ticker} - Hybrid Classifier: Mapped vs Actual Returns (OOS){title_note}',
              fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('5-Day Forward Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    if close:
        plt.close(fig)
    return fig


def plot_equity_curve(results_by_scheme, save_path=None, close=True):
    """Overlay cumulative equity curves for one or more backtest result dicts."""
    fig = plt.figure(figsize=(12, 6))
    for label, res in results_by_scheme.items():
        plt.plot(res['cumulative'].index, res['cumulative'].values,
                 label=f"{label} (Sharpe {res['sharpe']:.2f})", linewidth=2)
    plt.title('Walk-Forward Cross-Sectional Strategy - Cumulative Equity Curve', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Growth of $1 (gross of costs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    if close:
        plt.close(fig)
    return fig


def plot_metric_bar(labels, values, title, ylabel, color='steelblue',
                    hline=None, hline_label=None, save_path=None, close=True):
    """Generic labeled bar chart for a per-sector metric (R², accuracy, Sharpe, …)."""
    fig = plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=color, alpha=0.8)
    if hline is not None:
        plt.axhline(y=hline, color='black', linestyle='--', label=hline_label)
        plt.legend()
    plt.title(title, fontsize=15)
    plt.ylabel(ylabel)
    plt.xlabel('Sector ETF')
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    if close:
        plt.close(fig)
    return fig


def plot_eda(data, returns, plots_dir, close=True):
    """Generate and save the three EDA figures (price trends, return box, correlation)."""
    import seaborn as sns
    paths = {}

    fig1 = plt.figure(figsize=(12, 6))
    data.plot(ax=plt.gca())
    plt.title('Sector ETF Adjusted Closing Price Trends (2008-2025)')
    plt.xlabel('Date'); plt.ylabel('Price (USD)')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, 'Price_Trends_&_Return_Distributions', 'price_trends.png')
    os.makedirs(os.path.dirname(p1), exist_ok=True)
    plt.savefig(p1, dpi=120); paths['price_trends'] = p1
    if close:
        plt.close(fig1)

    fig2 = plt.figure(figsize=(12, 6))
    returns.plot(kind='box', ax=plt.gca())
    plt.title('Sector ETF Daily Return Distributions')
    plt.tight_layout()
    p2 = os.path.join(plots_dir, 'Price_Trends_&_Return_Distributions', 'return_distributions.png')
    plt.savefig(p2, dpi=120); paths['return_distributions'] = p2
    if close:
        plt.close(fig2)

    fig3 = plt.figure(figsize=(9, 7))
    sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=plt.gca())
    plt.title('Sector ETF Returns Correlation Heatmap')
    plt.tight_layout()
    p3 = os.path.join(plots_dir, 'Returns_Correlation_Heatmap', 'correlation_heatmap.png')
    os.makedirs(os.path.dirname(p3), exist_ok=True)
    plt.savefig(p3, dpi=120); paths['correlation_heatmap'] = p3
    if close:
        plt.close(fig3)

    return paths
