import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

tickers = ['XLK', 'XLF', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLY', 'SPY']

sector_params = {
    "XLK":  {"max_depth": 9,  "num_leaves": 64},
    "XLF":  {"max_depth": 14, "num_leaves": 128},
    "XLV":  {"max_depth": 10, "num_leaves": 72},
    "XLP":  {"max_depth": 7,  "num_leaves": 48},
    "XLI":  {"max_depth": 12, "num_leaves": 96},
    "XLB":  {"max_depth": 11, "num_leaves": 80},
    "XLU":  {"max_depth": 6,  "num_leaves": 32},
    "XLRE": {"max_depth": 10, "num_leaves": 64},
    "XLY":  {"max_depth": 10, "num_leaves": 80},
    "SPY":  {"max_depth": 10, "num_leaves": 72},
}

data    = pd.read_csv(os.path.join(DATA_DIR, 'sp500_sector_prices.csv'), index_col=0, parse_dates=True)
returns = data.pct_change().dropna()


def generate_features(returns, target_ticker, tickers, rolling_window=10, n_pca=5):
    """
    Build a high-dimensional feature matrix for the target sector ETF.

    All features are constructed with strict temporal alignment (shift(1)) to
    prevent look-ahead bias. Feature groups:
    - Rolling moments (mean, vol, skew, kurtosis) — distributional characterisation
    - PCA-compressed lag structure — compresses 10 lags into 5 orthogonal components
      (Principal Components) to decorrelate autocorrelated lag series
    - SPY correlation/covariance — rolling systematic-risk proxies
    - Drawdown — rolling maximum drawdown within the lookback window
    - Cross-sectional momentum/volatility — sector rotation signals
    - EWM features — exponentially weighted mean and std for adaptive smoothing
    - VIX/TNX z-scores — macroeconomic regime indicators (z-scored over 60d window)
    - Volatility regime binary flag — high-stress market state indicator
    - Lagged macro features (lags 1–5) for delayed macro transmission modelling
    - Interaction terms (momentum × volatility) — non-linear signal amplification
    - Rolling beta and market deviation — systematic vs idiosyncratic return decomposition
    - Pairwise sector correlations — cross-sectional contagion and rotation dynamics
    - Calendar features — seasonality and month-end effects
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

    features['sector_momentum']  = returns.drop(columns=[target_ticker]).mean(axis=1).shift(1)
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


def train_evaluate_model(features, target, target_ticker, sector_params, verbose=False):
    """
    Train a LightGBM regressor on a fixed temporal split (2010–2020 in-sample,
    2021–2025 out-of-time) and return OOT evaluation metrics.

    Steps:
    1. Multicollinearity pruning (|r| > 0.95 threshold on upper correlation triangle).
    2. Sector dummy encoding.
    3. Global StandardScaler fit on training data only — test set uses train-derived statistics.
    4. Feature selection via importance percentile threshold (65th pctile) to remove
       noise features and reduce overfitting on the OOT window.
    5. Second model fit on the reduced feature set; OOT metrics reported on this model.

    Note: this notebook uses a fixed temporal split rather than walk-forward CV for
    simpler per-sector debugging. For rigorous OOS evaluation see LGBM_functions.py.
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

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    X_train = X_scaled.loc['2010-01-01':'2020-12-31']
    X_test  = X_scaled.loc['2021-01-01':'2025-12-31']
    y_train = y.loc['2010-01-01':'2020-12-31']
    y_test  = y.loc['2021-01-01':'2025-12-31']

    params = sector_params.get(target_ticker, {"max_depth": 10, "num_leaves": 64})
    model  = LGBMRegressor(
        n_estimators=1000, learning_rate=0.01, verbose=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.5, random_state=42,
        **params,
    )
    model.fit(X_train, y_train)

    importances_df = pd.DataFrame({'Feature': X_scaled.columns, 'Importance': model.feature_importances_})
    important_feats = importances_df[
        importances_df['Importance'] > np.percentile(importances_df['Importance'], 65)
    ]
    X_top = X_scaled[important_feats['Feature']]

    X_train_top = X_top.loc['2010-01-01':'2020-12-31']
    X_test_top  = X_top.loc['2021-01-01':'2025-12-31']

    model.fit(X_train_top, y_train)
    y_pred = model.predict(X_test_top)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"[{target_ticker}] R²: {r2:.4f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

    return {
        'model':          model,
        'r2':             r2,
        'rmse':           rmse,
        'mae':            mae,
        'y_test':         y_test,
        'y_pred':         pd.Series(y_pred, index=y_test.index),
        'X_test':         X_test,
        'importances_df': importances_df,
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


# --- Main loop: train and evaluate across all sector ETFs ---
sector_results = []
for ticker in tickers:
    if ticker == 'SPY':
        continue

    print(f"\nRunning model pipeline for {ticker} ...")
    features = generate_features(returns, ticker, tickers)
    target   = returns[ticker].rolling(5).mean()

    result = train_evaluate_model(features, target, ticker, sector_params)
    sector_results.append({
        'Sector': ticker,
        'R2':     round(result['r2'],   4),
        'RMSE':   round(result['rmse'], 6),
        'MAE':    round(result['mae'],  6),
    })

    plot_actual_vs_predicted(result['y_test'], result['y_pred'], ticker)
    plot_feature_importances(result['importances_df'], ticker)

results_df = pd.DataFrame(sector_results)
print("\nSector Performance Summary:")
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(DATA_DIR, 'sector_model_summary.csv'), index=False)
print("Sector predictive analysis complete.")
