"""
LightGBM_prediction_model.py — Canonical end-to-end pipeline orchestrator.

Runs the full workflow in dependency order using the shared LGBM_functions library:

    1. Data ingestion        (Yahoo Finance -> data/sp500_sector_prices.csv)
    2. Exploratory analysis  (price trends, return distributions, correlation)
    3. Feature engineering + leakage-free walk-forward model training (per sector)
    4. Out-of-sample evaluation (R2 / RMSE / MAE / directional Sharpe)
    5. Risk attribution      (annualized return, volatility, Sharpe by sector)
    6. Cross-sectional portfolio backtest (un-leaked walk-forward trading Sharpe)

All leakage-sensitive logic (feature/target alignment, per-fold PCA, per-fold
scaling, ADF stationarity screening, walk-forward CV) lives in LGBM_functions.py so
that this orchestrator and the per-stage driver scripts share a single, audited
implementation. See the QuantFinance wiki synthesis note "LightGBM Sector ETF
Pipeline — Leakage Audit and Theoretical Context" for the full derivation.
"""

import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_FUNCTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LGBM_model_functions')
)
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import LGBM_functions as lgf  # noqa: E402


def run_eda(data, returns):
    """Exploratory data analysis: price trends, return distributions, correlations."""
    data.plot(figsize=(12, 6))
    plt.title('Sector ETF Adjusted Closing Price Trends (2008-2025)')
    plt.xlabel('Date'); plt.ylabel('Price (USD)')
    plt.tight_layout(); plt.show()

    returns.plot(kind='box', figsize=(12, 6))
    plt.title('Sector ETF Daily Return Distributions')
    plt.tight_layout(); plt.show()

    sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Sector ETF Returns Correlation Heatmap')
    plt.tight_layout(); plt.show()


def main():
    # 1. Data ingestion.
    data_path = os.path.join(lgf.get_data_dir(), 'sp500_sector_prices.csv')
    if not os.path.exists(data_path):
        print("Downloading price data from Yahoo Finance ...")
        lgf.download_prices()
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = data.pct_change().dropna()
    print(f"Loaded prices. Shape: {data.shape}")

    # 2. EDA.
    run_eda(data, returns)

    # 3-4. Feature engineering + walk-forward training + evaluation per sector.
    macro_z = lgf.download_macro_zscores()
    oof_pred, sector_results = {}, []
    for ticker in lgf.SECTORS:
        print(f"\nRunning pipeline for {ticker} ...")
        features = lgf.generate_features(returns, ticker, lgf.TICKERS, macro_z=macro_z)
        target = lgf.make_forward_target(returns, ticker, horizon=lgf.FORWARD_HORIZON)
        result = lgf.train_evaluate_model(features, target, ticker)
        oof_pred[ticker] = result['y_pred']
        sector_results.append({
            'Sector': ticker,
            'R2': round(result['r2'], 4),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6),
            'Directional_Sharpe': round(result['sharpe'], 4),
        })
        lgf.plot_actual_vs_predicted(result['y_test'], result['y_pred'], ticker)
        lgf.plot_feature_importances(result['importances_df'], ticker)

    results_df = pd.DataFrame(sector_results)
    print("\nSector Performance Summary (OOS):")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(lgf.get_data_dir(), 'sector_model_summary.csv'), index=False)

    # 5. Risk attribution.
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    risk_free_rate = 0.03
    sharpe_ratio = (mean_return - risk_free_rate) / volatility
    summary = pd.DataFrame({
        'Annual Return': mean_return, 'Annual Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
    }).sort_values(by='Sharpe Ratio', ascending=False)
    print("\nSector Risk Summary:")
    print(summary.to_string())
    summary.to_csv(os.path.join(lgf.get_data_dir(), 'lgbm_risk_summary.csv'))

    plt.style.use('ggplot')
    r2_sorted = results_df.sort_values(by='R2', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(r2_sorted['Sector'], r2_sorted['R2'], alpha=0.75, color='steelblue')
    plt.title('Out-of-Sample R2 by Sector', fontsize=16)
    plt.xlabel('Sector ETF'); plt.ylabel('R2'); plt.ylim(bottom=min(0, r2_sorted['R2'].min()))
    plt.grid(True, axis='y'); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(summary.index, summary['Sharpe Ratio'], color='mediumseagreen', alpha=0.75)
    plt.title('Annualized Sharpe Ratio by Sector (2008-2025)', fontsize=16)
    plt.xlabel('Sector ETF'); plt.ylabel('Sharpe Ratio')
    plt.axhline(y=1.0, color='black', linestyle='--', label='Sharpe = 1.0')
    plt.legend(); plt.grid(True, axis='y'); plt.tight_layout(); plt.show()

    # 6. Cross-sectional portfolio backtest (un-leaked walk-forward trading Sharpe).
    pred_matrix = pd.DataFrame(oof_pred).dropna()
    realized_next = returns[lgf.SECTORS].shift(-1)
    backtests = {}
    for label, scheme in [('Proportional', 'proportional'), ('Top-3 Long-Short', 'long_short')]:
        weights = lgf.build_weights(pred_matrix, scheme=scheme, top_n=3)
        res = lgf.backtest_portfolio(weights, realized_next)
        backtests[label] = res
        print(f"[{label}] Trading Sharpe: {res['sharpe']:.4f} | "
              f"Ann. return: {res['ann_return']:.4%} | Max DD: {res['max_drawdown']:.4%}")
    lgf.plot_equity_curve(backtests)

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
