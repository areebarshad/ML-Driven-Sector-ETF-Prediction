"""
LGBM_portfolio_backtest.py — Section 3.3: Simulated Portfolio Backtest Generation.

Converts the model's out-of-sample (out-of-fold) return predictions into a tradeable
cross-sectional long-short strategy and reports an UN-LEAKED walk-forward trading
Sharpe ratio.

Why this is un-leaked:
  - Every prediction used here is an out-of-fold prediction from the walk-forward
    TimeSeriesSplit loop in train_evaluate_model — i.e. it was produced by a model
    that never saw the corresponding date during training.
  - Portfolio weights formed at date t are applied to the NEXT-day realized return
    (returns.shift(-1)), so the position earns only future P&L.

Two allocation schemes are evaluated (framework Eq. 8):
  - Proportional:  w_{i,t} = yhat_{i,t} / sum_j |yhat_{j,t}|
  - Top-N long-short: equal-weight long the top-N and short the bottom-N sectors.

Theory: the Sharpe ratio (annualized by sqrt(252)) is the risk-adjusted objective;
a dollar-neutral long-short book isolates cross-sectional sector-selection skill from
broad-market beta. See the QuantFinance wiki: Sharpe Ratio, Walk-Forward Backtesting,
Mean-Variance Optimization.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bootstrap: import the shared library from the sibling LGBM_model_functions folder.
_FUNCTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LGBM_model_functions')
)
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import LGBM_functions as lgf  # noqa: E402


def run_backtest(top_n=3, rf=0.0):
    """Execute the full walk-forward prediction + cross-sectional backtest pipeline."""
    data_path = os.path.join(lgf.get_data_dir(), 'sp500_sector_prices.csv')
    if not os.path.exists(data_path):
        print("Price data not found; downloading from Yahoo Finance ...")
        lgf.download_prices()
    returns = lgf.load_returns(data_path)

    # Hoist the macro download once (shared across all sectors).
    macro_z = lgf.download_macro_zscores()

    # 1. Walk-forward OOF predictions for every non-benchmark sector.
    oof_pred = {}
    metrics = []
    for ticker in lgf.SECTORS:
        print(f"\nRunning walk-forward model for {ticker} ...")
        features = lgf.generate_features(returns, ticker, lgf.TICKERS, macro_z=macro_z)
        target = lgf.make_forward_target(returns, ticker, horizon=lgf.FORWARD_HORIZON)
        result = lgf.train_evaluate_model(features, target, ticker)
        oof_pred[ticker] = result['y_pred']
        metrics.append({
            'Sector': ticker,
            'R2': round(result['r2'], 4),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6),
            'Directional_Sharpe': round(result['sharpe'], 4),
        })

    metrics_df = pd.DataFrame(metrics)
    print("\nPer-sector OOS metrics:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(
        os.path.join(lgf.get_data_dir(), 'sector_model_summary.csv'), index=False
    )

    # 2. Assemble the cross-sectional prediction matrix (dates x sectors), keeping
    #    only dates for which every sector has an OOF prediction.
    pred_matrix = pd.DataFrame(oof_pred).dropna()

    # 3. Next-day realized returns aligned to the same sector columns. Using the
    #    1-day-ahead return as the tradeable proxy avoids the overlapping-window
    #    autocorrelation that a 5-day forward P&L would introduce, while still being
    #    driven purely by the walk-forward allocations.
    realized_next = returns[lgf.SECTORS].shift(-1)

    # 4. Build weights and backtest both schemes.
    results = {}
    for scheme_label, scheme in [('Proportional', 'proportional'),
                                 (f'Top-{top_n} Long-Short', 'long_short')]:
        weights = lgf.build_weights(pred_matrix, scheme=scheme, top_n=top_n)
        res = lgf.backtest_portfolio(weights, realized_next, rf=rf)
        results[scheme_label] = res
        print(f"\n[{scheme_label}] Un-leaked walk-forward trading Sharpe: {res['sharpe']:.4f}")
        print(f"    Annualized return: {res['ann_return']:.4%} | "
              f"Annualized vol: {res['ann_vol']:.4%} | "
              f"Max drawdown: {res['max_drawdown']:.4%}")

    # 5. Persist the equity curve plot and the strategy return series.
    plot_path = os.path.join(lgf.get_plots_dir('Portfolio_Backtest'), 'equity_curve.png')
    lgf.plot_equity_curve(results, save_path=plot_path)

    equity_df = pd.DataFrame({
        label: res['returns'] for label, res in results.items()
    })
    equity_df.to_csv(os.path.join(lgf.get_data_dir(), 'portfolio_backtest_returns.csv'))

    summary = pd.DataFrame([
        {
            'Strategy': label,
            'Sharpe': round(res['sharpe'], 4),
            'Annual_Return': round(res['ann_return'], 4),
            'Annual_Vol': round(res['ann_vol'], 4),
            'Max_Drawdown': round(res['max_drawdown'], 4),
        }
        for label, res in results.items()
    ])
    summary.to_csv(
        os.path.join(lgf.get_data_dir(), 'portfolio_backtest_summary.csv'), index=False
    )
    print("\nPortfolio backtest summary:")
    print(summary.to_string(index=False))
    print("\nPortfolio backtest complete.")
    return results


if __name__ == '__main__':
    run_backtest()
