"""
LightGBM_prediction_model.py — Canonical end-to-end pipeline (hybrid classifier).

Signal-classification variant: the engine is an LGBMClassifier that predicts the
binary directional signal 1[forward_return > 0], while a continuous out-of-sample R²
is retained by mapping each fold's predicted probability of an up-move back to the
return scale ( (P(up) - 0.5) * 2 * vol ). See LGBM_functions.train_evaluate_hybrid and
reports/Methodology_Enhancements for the derivation.

Plotting follows a two-stage, non-blocking architecture:
  Stage 1 (throughout): every figure is written to plots/<category>/... and closed
          immediately (plt.close), so asset generation never freezes the pipeline.
  Stage 2 (at the very end): plt.ion() + a batched re-render of all diagnostics +
          plt.show(block=True) so, under an interactive backend, every window opens
          simultaneously without blocking earlier computation. Under a headless
          backend (Agg) Stage 2 is a harmless no-op and the saved PNGs are the output.

Run:  python LightGBM_prediction_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_FUNCTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LGBM_model_functions')
)
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import LGBM_functions as lgf  # noqa: E402


def main():
    data_dir = lgf.get_data_dir()
    plots_dir = lgf.get_plots_dir()

    # ── 1. Data ingestion ───────────────────────────────────────────────────────
    data_path = os.path.join(data_dir, 'sp500_sector_prices.csv')
    if not os.path.exists(data_path):
        print("Downloading price data from Yahoo Finance ...")
        lgf.download_prices()
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns = data.pct_change().dropna()
    print(f"Loaded prices. Shape: {data.shape}")

    # ── 2. EDA (Stage 1: headless save) ─────────────────────────────────────────
    lgf.plot_eda(data, returns, plots_dir, close=True)

    # ── 3-4. Hybrid classifier training + OOS evaluation per sector ─────────────
    macro_z = lgf.download_macro_zscores()
    results, oof_pred, sector_rows = {}, {}, []
    for ticker in lgf.SECTORS:
        print(f"\nRunning hybrid pipeline for {ticker} ...")
        features = lgf.generate_features(returns, ticker, lgf.TICKERS, macro_z=macro_z)
        y_continuous = lgf.make_forward_target(returns, ticker, horizon=lgf.FORWARD_HORIZON)
        # Raw (unscaled) rolling volatility — identical definition to the {ticker}_vol
        # feature — used as the return-scale factor in the probability mapping.
        vol_series = returns[ticker].shift(1).rolling(lgf.ROLLING_WINDOW).std()

        result = lgf.train_evaluate_hybrid(features, y_continuous, ticker, vol_series)
        results[ticker] = result
        oof_pred[ticker] = result['y_pred']
        sector_rows.append({
            'Sector': ticker,
            'R2': round(result['r2'], 4),
            'Accuracy': round(result['accuracy'], 4),
            'AUC': round(result['auc'], 4),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6),
            'Directional_Sharpe': round(result['sharpe'], 4),
        })

        # Stage 1: save per-sector prediction and feature-importance figures.
        pred_dir = lgf.get_plots_dir(os.path.join('Sector_Predictions', ticker))
        imp_dir = lgf.get_plots_dir(os.path.join('Feature_Importances', ticker))
        lgf.plot_actual_vs_predicted(result['y_test'], result['y_pred'], ticker,
                                     save_path=os.path.join(pred_dir, 'prediction.png'), close=True)
        lgf.plot_feature_importances(result['importances_df'], ticker,
                                     save_path=os.path.join(imp_dir, 'feature_importance.png'), close=True)

    results_df = pd.DataFrame(sector_rows)
    print("\nSector Performance Summary (OOS, hybrid classifier):")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(data_dir, 'sector_model_summary.csv'), index=False)

    # ── 5. Risk attribution ─────────────────────────────────────────────────────
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
    summary.to_csv(os.path.join(data_dir, 'lgbm_risk_summary.csv'))

    # Stage 1: summary bar charts.
    r2_sorted = results_df.sort_values(by='R2', ascending=False)
    lgf.plot_metric_bar(
        r2_sorted['Sector'].tolist(), r2_sorted['R2'].tolist(),
        'Out-of-Sample Continuous R2 by Sector (Mapped from Classifier)',
        'R2 (mapped probability -> return)', color='steelblue',
        save_path=os.path.join(lgf.get_plots_dir('R²_By_Sector'), 'r2_by_sector.png'), close=True)

    acc_sorted = results_df.sort_values(by='Accuracy', ascending=False)
    lgf.plot_metric_bar(
        acc_sorted['Sector'].tolist(), acc_sorted['Accuracy'].tolist(),
        'Directional Classification Accuracy by Sector', 'Accuracy',
        color='mediumpurple', hline=0.5, hline_label='Coin-flip (0.50)',
        save_path=os.path.join(lgf.get_plots_dir('Classification_Metrics'), 'accuracy_by_sector.png'),
        close=True)
    auc_sorted = results_df.sort_values(by='AUC', ascending=False)
    lgf.plot_metric_bar(
        auc_sorted['Sector'].tolist(), auc_sorted['AUC'].tolist(),
        'Directional ROC-AUC by Sector', 'ROC-AUC',
        color='indianred', hline=0.5, hline_label='Random (0.50)',
        save_path=os.path.join(lgf.get_plots_dir('Classification_Metrics'), 'auc_by_sector.png'),
        close=True)

    lgf.plot_metric_bar(
        summary.index.tolist(), summary['Sharpe Ratio'].tolist(),
        'Annualized Sharpe Ratio by Sector (2008-2025)', 'Sharpe Ratio',
        color='mediumseagreen', hline=1.0, hline_label='Sharpe = 1.0',
        save_path=os.path.join(lgf.get_plots_dir('Sharpe_Ratio_By_Sector'), 'sharpe_by_sector.png'),
        close=True)

    # ── 6. Cross-sectional portfolio backtest ───────────────────────────────────
    pred_matrix = pd.DataFrame(oof_pred).dropna()
    realized_next = returns[lgf.SECTORS].shift(-1)
    backtests = {}
    for label, scheme in [('Proportional', 'proportional'), ('Top-3 Long-Short', 'long_short')]:
        weights = lgf.build_weights(pred_matrix, scheme=scheme, top_n=3)
        res = lgf.backtest_portfolio(weights, realized_next)
        backtests[label] = res
        print(f"[{label}] Trading Sharpe: {res['sharpe']:.4f} | "
              f"Ann. return: {res['ann_return']:.4%} | Max DD: {res['max_drawdown']:.4%}")
    lgf.plot_equity_curve(
        backtests, save_path=os.path.join(lgf.get_plots_dir('Portfolio_Backtest'), 'equity_curve.png'),
        close=True)

    bt_summary = pd.DataFrame([
        {'Strategy': label, 'Sharpe': round(r['sharpe'], 4),
         'Annual_Return': round(r['ann_return'], 4), 'Annual_Vol': round(r['ann_vol'], 4),
         'Max_Drawdown': round(r['max_drawdown'], 4)}
        for label, r in backtests.items()
    ])
    bt_summary.to_csv(os.path.join(data_dir, 'portfolio_backtest_summary.csv'), index=False)
    pd.DataFrame({k: v['returns'] for k, v in backtests.items()}).to_csv(
        os.path.join(data_dir, 'portfolio_backtest_returns.csv'))

    # ── Stage 2: simultaneous non-blocking batch display ────────────────────────
    print("\nAll calculations complete. Launching all evaluation plots concurrently ...")
    plt.ion()  # interactive mode: figures render without blocking the thread
    for ticker in lgf.SECTORS:
        lgf.plot_actual_vs_predicted(results[ticker]['y_test'], results[ticker]['y_pred'],
                                     ticker, close=False)
        lgf.plot_feature_importances(results[ticker]['importances_df'], ticker, close=False)
    lgf.plot_metric_bar(r2_sorted['Sector'].tolist(), r2_sorted['R2'].tolist(),
                        'OOS R2 by Sector', 'R2', close=False)
    lgf.plot_metric_bar(acc_sorted['Sector'].tolist(), acc_sorted['Accuracy'].tolist(),
                        'Accuracy by Sector', 'Accuracy', color='mediumpurple',
                        hline=0.5, hline_label='0.50', close=False)
    lgf.plot_metric_bar(summary.index.tolist(), summary['Sharpe Ratio'].tolist(),
                        'Sharpe by Sector', 'Sharpe', color='mediumseagreen',
                        hline=1.0, hline_label='1.0', close=False)
    lgf.plot_equity_curve(backtests, close=False)

    # Keep every window open together until the user closes them. Under a headless
    # (Agg) backend this returns immediately, so batch asset generation still completes.
    plt.show(block=True)
    print("Pipeline complete.")


if __name__ == '__main__':
    main()
