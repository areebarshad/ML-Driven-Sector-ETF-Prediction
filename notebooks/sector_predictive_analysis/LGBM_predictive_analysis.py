"""
LGBM_predictive_analysis.py — Per-sector predictive analysis driver.

Trains and evaluates the LightGBM model for every sector using the leakage-free
walk-forward pipeline in LGBM_functions.py, then plots predictions and feature
importances and writes the OOS metrics summary.

This script imports the shared library rather than re-inlining feature engineering
or the CV loop, so there is exactly one implementation of the leakage-sensitive
logic (feature/target alignment, per-fold PCA, per-fold scaling, ADF screening).
"""

import os
import sys

import pandas as pd

_FUNCTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LGBM_model_functions')
)
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import LGBM_functions as lgf  # noqa: E402


def main():
    data_path = os.path.join(lgf.get_data_dir(), 'sp500_sector_prices.csv')
    if not os.path.exists(data_path):
        print("Price data not found; downloading from Yahoo Finance ...")
        lgf.download_prices()
    returns = lgf.load_returns(data_path)

    # Hoist the macro z-score download once for all sectors.
    macro_z = lgf.download_macro_zscores()

    sector_results = []
    for ticker in lgf.SECTORS:
        print(f"\nRunning model pipeline for {ticker} ...")

        features = lgf.generate_features(returns, ticker, lgf.TICKERS, macro_z=macro_z)
        # CORRECTED target: k-day FORWARD average return (.shift(-k)), not a
        # backward-looking rolling mean of the past.
        target = lgf.make_forward_target(returns, ticker, horizon=lgf.FORWARD_HORIZON)

        result = lgf.train_evaluate_model(features, target, ticker)

        sector_results.append({
            'Sector': ticker,
            'R2': round(result['r2'], 4),
            'RMSE': round(result['rmse'], 6),
            'MAE': round(result['mae'], 6),
            'Directional_Sharpe': round(result['sharpe'], 4),
        })

        pred_dir = lgf.get_plots_dir(os.path.join('Sector_Predictions', ticker))
        imp_dir = lgf.get_plots_dir(os.path.join('Feature_Importances', ticker))
        lgf.plot_actual_vs_predicted(
            result['y_test'], result['y_pred'], ticker,
            save_path=os.path.join(pred_dir, 'prediction.png'),
        )
        lgf.plot_feature_importances(
            result['importances_df'], ticker,
            save_path=os.path.join(imp_dir, 'feature_importance.png'),
        )

    results_df = pd.DataFrame(sector_results)
    print("\nSector Performance Summary (out-of-sample):")
    print(results_df.to_string(index=False))
    results_df.to_csv(
        os.path.join(lgf.get_data_dir(), 'sector_model_summary.csv'), index=False
    )
    print("Sector predictive analysis complete.")


if __name__ == '__main__':
    main()
