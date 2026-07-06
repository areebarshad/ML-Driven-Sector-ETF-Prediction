"""
LGBM_sector_data_collection.py — Data ingestion stage.

Downloads adjusted closing prices for the nine S&P 500 sector ETFs plus SPY over the
full 2008-2025 sample and persists them to data/sp500_sector_prices.csv. The window
deliberately spans multiple macro regimes (GFC, COVID-19 shock, post-ZIRP tightening)
so the downstream walk-forward evaluation is exposed to regime diversity.
"""

import os
import sys

_FUNCTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'LGBM_model_functions')
)
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

import LGBM_functions as lgf  # noqa: E402


def main():
    data = lgf.download_prices(lgf.TICKERS, lgf.START_DATE, lgf.END_DATE, save=True)
    print(f"Data collected and shaped: {data.shape}")
    print(f"Saved to: {os.path.join(lgf.get_data_dir(), 'sp500_sector_prices.csv')}")
    print(data.head())


if __name__ == '__main__':
    main()
