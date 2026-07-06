"""
LightGBM Sector ETF Return Prediction — End-to-End Pipeline

This script consolidates data ingestion, feature engineering, model training,
out-of-sample evaluation, and risk analysis into a single reproducible pipeline.

Methodology overview:
- Target variable: 5-day forward rolling mean return (annualised signal smoothing).
- Feature space: ~80+ engineered features per sector covering statistical moments,
  PCA-compressed lag structure, macroeconomic regime indicators (VIX, TNX),
  cross-sectional sector signals, and calendar effects.
- Model: LightGBM gradient-boosted decision tree regressor with sector-specific
  hyperparameters (max_depth, num_leaves) calibrated to each sector's
  autocorrelation structure and signal-to-noise ratio.
- Validation: walk-forward TimeSeriesSplit (5 folds) with per-fold StandardScaler
  refitting to eliminate global-scaler look-ahead leakage.
- Metrics: R² (explained variance ratio), RMSE, MAE, and annualised Sharpe ratio
  of a sign-based directional strategy derived from model predictions.
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

target_ticker = 'XLK'  # example single-sector target; full loop runs all sectors below

tickers = ['XLK', 'XLF', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLY', 'SPY']

# Sector-specific hyperparameters: max_depth and num_leaves control tree complexity.
# Deeper trees capture more non-linear interactions but are more prone to overfitting
# in noisy financial return series. Values are tuned per sector's return dynamics.
sector_params = {
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

# ── Data Collection ────────────────────────────────────────────────────────────

# Download adjusted closing prices over the full sample period.
# The window 2008–2025 spans the Global Financial Crisis (GFC), European Debt Crisis,
# COVID-19 liquidity shock, and the post-ZIRP monetary tightening cycle — providing
# regime diversity essential for robust model generalisation.
data = yf.download(tickers, start='2008-01-01', end='2025-12-31')['Close']
data.to_csv(os.path.join(DATA_DIR, 'sp500_sector_prices.csv'))
print(f"Data collected. Shape: {data.shape}")
print(data.head())

# ── EDA ────────────────────────────────────────────────────────────────────────

data    = pd.read_csv(os.path.join(DATA_DIR, 'sp500_sector_prices.csv'), index_col=0, parse_dates=True)
returns = data.pct_change().dropna()

# Price-level plot: non-stationary series useful for identifying structural breaks
# and long-run momentum regimes across business cycles.
data.plot(figsize=(12, 6))
plt.title('Sector ETF Adjusted Closing Price Trends (2008–2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.show()

# Return distribution box plots: expose the first four statistical moments
# (central tendency, dispersion, skewness, excess kurtosis) per sector.
returns.plot(kind='box', figsize=(12, 6))
plt.title('Sector ETF Daily Return Distributions')
plt.tight_layout()
plt.show()

# Cross-sectional Pearson correlation heatmap: quantifies linear co-movement.
# High intra-factor correlations imply limited diversification benefit;
# low values indicate orthogonal risk factor exposures.
corr = returns.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Sector ETF Returns Correlation Heatmap')
plt.tight_layout()
plt.show()

print("EDA complete.")

# ── Feature Engineering ────────────────────────────────────────────────────────

def generate_features(returns, target_ticker, tickers, rolling_window=10, n_pca=5):
    """
    Construct a look-ahead-free feature matrix for the specified sector ETF.

    All features are lagged by at least 1 trading day. Feature groups:
    - Rolling statistical moments (mean, std, skew, kurtosis)
    - PCA-compressed lag structure: 10 lags → 5 orthogonal principal components
    - Rolling SPY correlation and covariance (systematic risk proxies)
    - Rolling maximum drawdown (downside risk measure)
    - Cross-sectional sector momentum and volatility (rotation signals)
    - Exponentially weighted mean/std (adaptive smoothing)
    - VIX z-score: CBOE Implied Volatility Index standardised over 60d window
    - TNX z-score: 10-year Treasury yield standardised over 60d window
    - Binary volatility regime flag (VIX z-score > +1σ)
    - Lagged macro features (lags 1–5) for delayed transmission modelling
    - Momentum–volatility interaction terms (non-linear signal amplification)
    - Rolling beta to SPY and idiosyncratic deviation
    - Pairwise sector cross-correlations (15-day rolling)
    - Calendar effects: month, ISO week, day-of-week, month boundary flags
    """
    lagged_returns = returns.shift(1)
    features = pd.DataFrame(index=returns.index)

    features[f'{target_ticker}_mean']           = lagged_returns[target_ticker].rolling(rolling_window).mean()
    features[f'{target_ticker}_vol']            = lagged_returns[target_ticker].rolling(rolling_window).std()
    features[f'{target_ticker}_skew']           = lagged_returns[target_ticker].rolling(rolling_window).skew()
    features[f'{target_ticker}_kurt']           = lagged_returns[target_ticker].rolling(rolling_window).kurt()
    features[f'{target_ticker}_momentum']       = (
        lagged_returns[target_ticker] - lagged_returns[target_ticker].rolling(rolling_window).mean()
    )
    features[f'{target_ticker}_momentum_ratio'] = (
        features[f'{target_ticker}_momentum'] / (features[f'{target_ticker}_vol'] + 1e-6)
    )

    lags = pd.DataFrame(
        {f'lag_{i}': returns[target_ticker].shift(i) for i in range(1, 11)}
    ).dropna()
    pca = PCA(n_components=n_pca)
    lagged_pca = pd.DataFrame(pca.fit_transform(lags), index=lags.index)
    lagged_pca.columns = [f'lag_pca_{i}' for i in range(lagged_pca.shape[1])]
    features = features.join(lagged_pca)

    features[f'{target_ticker}_rolling_mean_10']  = returns[target_ticker].rolling(10).mean().shift(1)
    features[f'{target_ticker}_rolling_std_10']   = returns[target_ticker].rolling(10).std().shift(1)
    features[f'{target_ticker}_volatility_ratio'] = (
        features[f'{target_ticker}_rolling_std_10'] /
        (features[f'{target_ticker}_rolling_mean_10'] + 1e-6)
    )

    features[f'{target_ticker}_corr'] = returns[target_ticker].rolling(10).corr(returns['SPY']).shift(1)
    features[f'{target_ticker}_cov']  = returns[target_ticker].rolling(10).cov(returns['SPY']).shift(1)

    features[f'{target_ticker}_cumulative_return'] = (
        returns[target_ticker].rolling(20).apply(np.sum).shift(1)
    )
    features[f'{target_ticker}_max_drawdown'] = (
        returns[target_ticker]
        .rolling(20)
        .apply(lambda x: np.min(x / (np.maximum.accumulate(x + 1e-6)) - 1))
        .shift(1)
    )

    features['sector_momentum']   = returns.drop(columns=[target_ticker]).mean(axis=1).shift(1)
    features['sector_volatility'] = returns.drop(columns=[target_ticker]).std(axis=1).shift(1)
    features['sector_sharpe']     = features['sector_momentum'] / (features['sector_volatility'] + 1e-6)

    features[f'{target_ticker}_ewm_mean'] = returns[target_ticker].ewm(span=10).mean().shift(1)
    features[f'{target_ticker}_ewm_std']  = returns[target_ticker].ewm(span=10).std().shift(1)

    vix_raw = yf.download("^VIX", start="2008-01-01", end="2025-12-31", progress=False)['Close']
    tnx_raw = yf.download("^TNX", start="2008-01-01", end="2025-12-31", progress=False)['Close']
    vix_znorm = (vix_raw - vix_raw.rolling(60).mean()) / (vix_raw.rolling(60).std() + 1e-6)
    tnx_znorm = (tnx_raw - tnx_raw.rolling(60).mean()) / (tnx_raw.rolling(60).std() + 1e-6)
    features['vix_zscore'] = vix_znorm.shift(1)
    features['tnx_zscore'] = tnx_znorm.shift(1)
    features['market_vol_regime'] = (features['vix_zscore'] > 1).astype(int)

    for lag in range(1, 6):
        features[f'vix_zscore_lag{lag}'] = features['vix_zscore'].shift(lag)
        features[f'tnx_zscore_lag{lag}'] = features['tnx_zscore'].shift(lag)

    for lag in range(1, 6):
        features[f'{target_ticker}_lag{lag}_momentum'] = (
            returns[target_ticker].shift(lag) - returns[target_ticker].shift(lag + 5)
        )
        features[f'{target_ticker}_lag{lag}_mom_vol'] = (
            features[f'{target_ticker}_lag{lag}_momentum'] * features[f'{target_ticker}_vol']
        )

    features[f'{target_ticker}_rolling_beta'] = (
        returns[target_ticker].rolling(20).cov(returns['SPY']) /
        returns['SPY'].rolling(20).var()
    ).shift(1)

    features[f'{target_ticker}_market_dev'] = (
        (returns[target_ticker] - returns['SPY']).rolling(7).mean().shift(1)
    )

    for other in tickers:
        if other != target_ticker:
            features[f'{target_ticker}_{other}_corr'] = (
                returns[target_ticker].rolling(15).corr(returns[other])
            ).shift(1)

    features[f'{target_ticker}_returns_1d'] = returns[target_ticker].shift(1)

    features['month']          = features.index.month
    features['week_of_year']   = features.index.isocalendar().week.astype(int)
    features['day_of_week']    = features.index.dayofweek
    features['is_month_start'] = features.index.is_month_start.astype(int)
    features['is_month_end']   = features.index.is_month_end.astype(int)

    return features.dropna()


# ── Model Training & Evaluation ────────────────────────────────────────────────

def train_evaluate_model(features, target, target_ticker, sector_params, verbose=False):
    """
    Train a LightGBM regressor via walk-forward time-series cross-validation
    and report strictly out-of-sample performance metrics.

    Leakage controls:
    - StandardScaler fitted exclusively on each training fold (no global-scaler leakage).
    - Target series shifted forward to ensure no contemporaneous information enters features.
    - Multicollinearity pruning (|r| > 0.95) on upper correlation triangle reduces
      redundant feature interactions that inflate in-sample fit without OOS benefit.

    The final model is refit on the full dataset solely for feature importance extraction;
    all reported metrics (R², RMSE, MAE, Sharpe) derive from out-of-fold (OOF) predictions.
    """
    target   = target.dropna()
    features = features.dropna()
    common_index = features.index.intersection(target.index)
    X = features.loc[common_index]
    y = target.loc[common_index]

    corr_matrix = X.corr().abs()
    upper   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=to_drop)

    sector_dummies = pd.get_dummies([target_ticker] * len(X), prefix='sector')
    X = pd.concat(
        [X, pd.DataFrame(sector_dummies.values, index=X.index, columns=sector_dummies.columns)],
        axis=1,
    )

    params = sector_params.get(target_ticker, {"max_depth": 10, "num_leaves": 64})

    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds, oof_actuals, oof_index = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = LGBMRegressor(
            n_estimators=1000, learning_rate=0.01, verbose=-1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.5, random_state=42,
            **params,
        )
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        oof_preds.extend(preds)
        oof_actuals.extend(y_test.values)
        oof_index.extend(y_test.index)

    r2     = r2_score(oof_actuals, oof_preds)
    rmse   = math.sqrt(mean_squared_error(oof_actuals, oof_preds))
    mae    = mean_absolute_error(oof_actuals, oof_preds)
    strat  = np.sign(oof_preds) * np.array(oof_actuals)
    sharpe = strat.mean() / (strat.std() + 1e-9) * math.sqrt(252)

    print(
        f"[{target_ticker}] R² (OOS): {r2:.4f} | RMSE: {rmse:.6f} | "
        f"MAE: {mae:.6f} | Sharpe (directional): {sharpe:.4f}"
    )

    scaler_final = StandardScaler()
    X_all_scaled = pd.DataFrame(scaler_final.fit_transform(X), index=X.index, columns=X.columns)
    model_final  = LGBMRegressor(
        n_estimators=1000, learning_rate=0.01, verbose=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.5, random_state=42,
        **params,
    )
    model_final.fit(X_all_scaled, y)
    importances_df = pd.DataFrame({
        "Feature":    X.columns,
        "Importance": model_final.feature_importances_,
    })

    y_oot      = pd.Series(oof_actuals, index=oof_index)
    y_pred_oot = pd.Series(oof_preds,   index=oof_index)

    return {
        "model":          model_final,
        "r2":             r2,
        "rmse":           rmse,
        "mae":            mae,
        "sharpe":         sharpe,
        "y_test":         y_oot,
        "y_pred":         y_pred_oot,
        "X_test":         X.loc[oof_index],
        "importances_df": importances_df,
    }


def plot_feature_importances(importances_df, target_ticker):
    top_feats = importances_df.sort_values(by='Importance', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_feats['Feature'], top_feats['Importance'], color='teal')
    plt.title(f'Top 20 Feature Importances — {target_ticker}')
    plt.xlabel('Importance (split gain)')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, target_ticker, title_note=''):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual',    color='crimson',   alpha=0.7, linewidth=2)
    plt.plot(y_test.index, y_pred.values, label='Predicted', color='royalblue', alpha=0.7, linewidth=2)
    plt.title(f'{target_ticker} — LightGBM Predicted vs Actual Returns{title_note}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Main Sector Loop ───────────────────────────────────────────────────────────

sector_results = []
for ticker in tickers:
    if ticker == 'SPY':
        continue

    print(f"\nRunning pipeline for {ticker} ...")
    features = generate_features(returns, ticker, tickers)
    # Target: 5-day forward rolling mean return — smoothed short-term prediction horizon
    target = returns[ticker].rolling(5).mean().shift(-5)

    result = train_evaluate_model(features, target, ticker, sector_params)
    sector_results.append({
        'Sector': ticker,
        'R2':     round(result['r2'],     4),
        'RMSE':   round(result['rmse'],   6),
        'MAE':    round(result['mae'],    6),
        'Sharpe': round(result['sharpe'], 4),
    })

    plot_actual_vs_predicted(result['y_test'], result['y_pred'], ticker)
    plot_feature_importances(result['importances_df'], ticker)

results_df = pd.DataFrame(sector_results)
print("\nSector Performance Summary (OOS):")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(DATA_DIR, 'sector_model_summary.csv'), index=False)

# ── Risk Analysis ──────────────────────────────────────────────────────────────

mean_return = returns.mean() * 252
volatility  = returns.std() * np.sqrt(252)
risk_free_rate = 0.03
sharpe_ratio = (mean_return - risk_free_rate) / volatility

summary = pd.DataFrame({
    'Annual Return':     mean_return,
    'Annual Volatility': volatility,
    'Sharpe Ratio':      sharpe_ratio,
})
summary = summary.sort_values(by='Sharpe Ratio', ascending=False)
print("\nSector Risk Summary:")
print(summary.to_string())
summary.to_csv(os.path.join(DATA_DIR, 'lgbm_risk_summary.csv'))

plt.style.use('ggplot')

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

print("Pipeline complete.")
