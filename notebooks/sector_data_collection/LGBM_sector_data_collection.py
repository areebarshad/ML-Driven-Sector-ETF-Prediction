import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

target_ticker = 'XLK'

tickers = ['XLK', 'XLF', 'XLV', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLY', 'SPY']

# Per-sector hyperparameter grid: tree depth and leaf count tuned to each sector's
# signal-to-noise ratio and return autocorrelation structure.
sector_params = {
    "XLK":  {"max_depth": 9,  "num_leaves": 64},   # Tech: strong mean-reversion trends
    "XLF":  {"max_depth": 14, "num_leaves": 128},  # Financials: macro-driven, high kurtosis
    "XLV":  {"max_depth": 10, "num_leaves": 72},   # Healthcare: stable beta, low vol regime
    "XLP":  {"max_depth": 7,  "num_leaves": 48},   # Consumer Staples: defensive, smooth autocorrelation
    "XLI":  {"max_depth": 12, "num_leaves": 96},   # Industrials: cyclical, noisy residuals
    "XLB":  {"max_depth": 11, "num_leaves": 80},   # Materials: commodity-linked, fat tails
    "XLU":  {"max_depth": 6,  "num_leaves": 32},   # Utilities: near-stationary, minimal variance
    "XLRE": {"max_depth": 10, "num_leaves": 64},   # Real Estate: duration-sensitive to rate shocks
    "XLY":  {"max_depth": 10, "num_leaves": 80},   # Consumer Discretionary: retail cycle exposure
    "SPY":  {"max_depth": 10, "num_leaves": 72},   # Broad market benchmark
}

# Download adjusted closing prices over the full sample period (2008-2025),
# covering multiple business cycles including the GFC, COVID-19 shock, and post-ZIRP regime.
data = yf.download(tickers, start='2008-01-01', end='2025-12-31')['Close']

output_path = os.path.join(DATA_DIR, 'sp500_sector_prices.csv')
data.to_csv(output_path)

print(f"Data collected and shaped: {data.shape}")
print(f"Saved to: {output_path}")
print(data.head())
