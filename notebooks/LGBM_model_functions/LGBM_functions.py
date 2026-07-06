import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generate_features(returns, target_ticker, tickers, rolling_window=10, n_pca=5):
    """
    Construct the full feature matrix for a single target sector ETF.

    Feature groups:
    - Rolling statistical moments (mean, std, skew, kurtosis) over a configurable window
      to capture non-stationary distributional shifts in the return series.
    - Momentum and momentum-to-volatility ratio as proxies for price continuation signals.
    - PCA-compressed lag structure: 10 raw lags are projected onto 5 orthogonal components
      to eliminate multicollinearity while preserving autocorrelation information.
    - SPY rolling correlation and covariance: measure of systematic (market-beta) exposure.
    - Rolling max drawdown: captures downside deviation within the lookback window.
    - Sector-wide cross-sectional momentum and volatility: encodes relative-strength signals
      across the sector universe.
    - Exponentially weighted mean and std (EWM): down-weights stale observations,
      improving responsiveness to recent volatility regime shifts.
    - VIX z-score: standardised CBOE Volatility Index capturing implied variance expectations;
      z-scored over a 60-day rolling window to remove secular trend.
    - TNX z-score: standardised 10-year Treasury yield; encodes duration/rate sensitivity.
    - Volatility regime indicator: binary flag (VIX z-score > 1σ) for high-stress regimes.
    - Lagged macro features (lags 1–5) for VIX and TNX to model delayed macro transmission.
    - Momentum–volatility interaction terms: cross-products of lagged momentum and rolling vol.
    - Rolling beta to SPY: systematic risk exposure estimated via rolling covariance/variance.
    - Market deviation: sector alpha relative to SPY over a 7-day window.
    - Pairwise sector correlations: captures cross-sector contagion and rotation dynamics.
    - Calendar features: month, week-of-year, day-of-week, month-start/end flags.

    All features are lagged by at least 1 day to enforce strict look-ahead-free construction.
    """
    lagged_returns = returns.shift(1)
    features = pd.DataFrame(index=returns.index)

    # Rolling statistical moments
    features[f'{target_ticker}_mean'] = lagged_returns[target_ticker].rolling(rolling_window).mean()
    features[f'{target_ticker}_vol']  = lagged_returns[target_ticker].rolling(rolling_window).std()
    features[f'{target_ticker}_skew'] = lagged_returns[target_ticker].rolling(rolling_window).skew()
    features[f'{target_ticker}_kurt'] = lagged_returns[target_ticker].rolling(rolling_window).kurt()
    features[f'{target_ticker}_momentum'] = (
        lagged_returns[target_ticker] - lagged_returns[target_ticker].rolling(rolling_window).mean()
    )
    features[f'{target_ticker}_momentum_ratio'] = (
        features[f'{target_ticker}_momentum'] / (features[f'{target_ticker}_vol'] + 1e-6)
    )

    # PCA-compressed lag structure (eliminates multicollinearity among raw lags)
    lags = pd.DataFrame(
        {f'lag_{i}': returns[target_ticker].shift(i) for i in range(1, 11)}
    ).dropna()
    pca = PCA(n_components=n_pca)
    lagged_pca = pd.DataFrame(pca.fit_transform(lags), index=lags.index)
    lagged_pca.columns = [f'lag_pca_{i}' for i in range(lagged_pca.shape[1])]
    features = features.join(lagged_pca)

    # Technical indicators (all shifted by 1 to avoid look-ahead)
    features[f'{target_ticker}_rolling_mean_10'] = returns[target_ticker].rolling(10).mean().shift(1)
    features[f'{target_ticker}_rolling_std_10']  = returns[target_ticker].rolling(10).std().shift(1)
    features[f'{target_ticker}_volatility_ratio'] = (
        features[f'{target_ticker}_rolling_std_10'] /
        (features[f'{target_ticker}_rolling_mean_10'] + 1e-6)
    )

    # Systematic risk proxies vs SPY
    features[f'{target_ticker}_corr'] = returns[target_ticker].rolling(10).corr(returns['SPY']).shift(1)
    features[f'{target_ticker}_cov']  = returns[target_ticker].rolling(10).cov(returns['SPY']).shift(1)

    # Cumulative return and maximum drawdown within rolling window
    features[f'{target_ticker}_cumulative_return'] = (
        returns[target_ticker].rolling(20).apply(np.sum).shift(1)
    )
    features[f'{target_ticker}_max_drawdown'] = (
        returns[target_ticker]
        .rolling(20)
        .apply(lambda x: np.min(x / (np.maximum.accumulate(x + 1e-6)) - 1))
        .shift(1)
    )

    # Cross-sectional sector momentum and volatility
    features['sector_momentum']  = returns.drop(columns=[target_ticker]).mean(axis=1).shift(1)
    features['sector_volatility'] = returns.drop(columns=[target_ticker]).std(axis=1).shift(1)
    features['sector_sharpe']     = features['sector_momentum'] / (features['sector_volatility'] + 1e-6)

    # Exponentially weighted moments
    features[f'{target_ticker}_ewm_mean'] = returns[target_ticker].ewm(span=10).mean().shift(1)
    features[f'{target_ticker}_ewm_std']  = returns[target_ticker].ewm(span=10).std().shift(1)

    # Macroeconomic regime indicators: VIX (implied vol) and TNX (10-yr yield)
    vix_raw = yf.download("^VIX", start="2008-01-01", end="2025-12-31", progress=False)['Close']
    tnx_raw = yf.download("^TNX", start="2008-01-01", end="2025-12-31", progress=False)['Close']

    vix_znorm = (vix_raw - vix_raw.rolling(60).mean()) / (vix_raw.rolling(60).std() + 1e-6)
    tnx_znorm = (tnx_raw - tnx_raw.rolling(60).mean()) / (tnx_raw.rolling(60).std() + 1e-6)
    features['vix_zscore'] = vix_znorm.shift(1)
    features['tnx_zscore'] = tnx_znorm.shift(1)

    # Binary volatility regime flag: 1 when VIX z-score exceeds +1σ (stress regime)
    features['market_vol_regime'] = (features['vix_zscore'] > 1).astype(int)

    # Lagged macro features (lags 1–5)
    for lag in range(1, 6):
        features[f'vix_zscore_lag{lag}'] = features['vix_zscore'].shift(lag)
        features[f'tnx_zscore_lag{lag}'] = features['tnx_zscore'].shift(lag)

    # Momentum–volatility interaction terms
    for lag in range(1, 6):
        features[f'{target_ticker}_lag{lag}_momentum'] = (
            returns[target_ticker].shift(lag) - returns[target_ticker].shift(lag + 5)
        )
        features[f'{target_ticker}_lag{lag}_mom_vol'] = (
            features[f'{target_ticker}_lag{lag}_momentum'] * features[f'{target_ticker}_vol']
        )

    # Rolling beta and idiosyncratic deviation from SPY
    features[f'{target_ticker}_rolling_beta'] = (
        returns[target_ticker].rolling(20).cov(returns['SPY']) /
        returns['SPY'].rolling(20).var()
    ).shift(1)

    features[f'{target_ticker}_market_dev'] = (
        (returns[target_ticker] - returns['SPY']).rolling(7).mean().shift(1)
    )

    # Pairwise sector cross-correlations (15-day rolling)
    for other in tickers:
        if other != target_ticker:
            features[f'{target_ticker}_{other}_corr'] = (
                returns[target_ticker].rolling(15).corr(returns[other])
            ).shift(1)

    features[f'{target_ticker}_returns_1d'] = returns[target_ticker].shift(1)

    # Calendar features for seasonality effects
    features['month']         = features.index.month
    features['week_of_year']  = features.index.isocalendar().week.astype(int)
    features['day_of_week']   = features.index.dayofweek
    features['is_month_start'] = features.index.is_month_start.astype(int)
    features['is_month_end']   = features.index.is_month_end.astype(int)

    return features.dropna()


def train_evaluate_model(features, target, target_ticker, sector_params, verbose=False):
    """
    Train and evaluate a sector-specific LightGBM regressor using walk-forward
    (time-series) cross-validation to produce strictly out-of-sample (OOS) metrics.

    Pipeline:
    1. Multicollinearity pruning: features with pairwise Pearson |r| > 0.95 are removed
       to stabilise gradient descent and reduce effective model complexity.
    2. Sector dummy encoding: one-hot vector appended to allow a unified multi-sector
       model to learn sector-specific intercept adjustments.
    3. Walk-forward CV (TimeSeriesSplit, n_splits=5): the scaler is re-fitted exclusively
       on each training fold to eliminate global-scaler data leakage — a common source
       of inflated OOS metrics in financial ML.
    4. OOS aggregation: out-of-fold (OOF) predictions are concatenated to form a
       pseudo-OOS sequence covering the full date range.
    5. Evaluation metrics: R² (explained variance), RMSE, MAE, and annualised Sharpe ratio
       of the sign-based strategy (long when predicted return > 0, short otherwise).
    6. Final refit on full dataset for feature importance extraction only — this model
       is not used for any reported OOS statistics.

    Returns a results dict containing the fitted model, OOS metrics, OOF predictions,
    and feature importances.
    """
    target   = target.dropna()
    features = features.dropna()
    common_index = features.index.intersection(target.index)
    X = features.loc[common_index]
    y = target.loc[common_index]

    # Prune highly collinear features (upper triangle of correlation matrix)
    corr_matrix = X.corr().abs()
    upper   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=to_drop)

    # Sector dummy variable
    sector_dummies = pd.get_dummies([target_ticker] * len(X), prefix='sector')
    X = pd.concat(
        [X, pd.DataFrame(sector_dummies.values, index=X.index, columns=sector_dummies.columns)],
        axis=1,
    )

    params = sector_params.get(target_ticker, {"max_depth": 10, "num_leaves": 64})

    # Walk-forward cross-validation: scaler fitted only on train fold each iteration
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

    # Aggregated OOS evaluation metrics
    r2     = r2_score(oof_actuals, oof_preds)
    rmse   = math.sqrt(mean_squared_error(oof_actuals, oof_preds))
    mae    = mean_absolute_error(oof_actuals, oof_preds)
    strat  = np.sign(oof_preds) * np.array(oof_actuals)
    sharpe = strat.mean() / (strat.std() + 1e-9) * math.sqrt(252)

    print(
        f"[{target_ticker}] R² (OOS): {r2:.4f} | RMSE: {rmse:.6f} | "
        f"MAE: {mae:.6f} | Sharpe: {sharpe:.4f}"
    )

    # Full-data refit for feature importance only (not used for OOS reporting)
    scaler_final  = StandardScaler()
    X_all_scaled  = pd.DataFrame(scaler_final.fit_transform(X), index=X.index, columns=X.columns)
    model_final   = LGBMRegressor(
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
    """Bar chart of top-20 LightGBM split-gain feature importances."""
    top_feats = importances_df.sort_values(by='Importance', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_feats['Feature'], top_feats['Importance'], color='teal')
    plt.title(f'Top 20 Feature Importances — {target_ticker}')
    plt.xlabel('Importance (split gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, target_ticker, title_note=''):
    """Time-series overlay of realised vs model-predicted forward returns."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values,  label='Actual',    color='crimson',   alpha=0.7, linewidth=2)
    plt.plot(y_test.index, y_pred.values,  label='Predicted', color='royalblue', alpha=0.7, linewidth=2)
    plt.title(f'{target_ticker} — LightGBM Predicted vs Actual Returns{title_note}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
