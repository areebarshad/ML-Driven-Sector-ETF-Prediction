# S&P 500 Sector Prediction - Model Report

## Overview

- Objective: Predict returns for sector ETFs using historical price, volatility and macroeconomic data.
- Approach: A feature engineered LightGBM model trained sector-wise with out-of-time evaluation from 2021-2025.
- Sectors Modeled: XLK, XLF, XLV, XLI, XLP, XLB, XLU, XLRE, XLY (SPY used as a benchmark).

## Data Summary

- Time Period: 2008-2025
- Sources: - ETF Price Data: Yahoo Finance (`yFinance`)
           - VIX -> Volatility Index (`^VIX`)
           - TNX -> 10-Year Treasury Yield (`^TNX`)

## Feature Engineering

- Rolling Statistics: Mean, Volatility, Skewness, Kurtosis
- Momentum Features: Raw and volatility-adjusted
- PCA Compression: Applied to 10 lags
- Technical Indicators: Moving averages, volatility ratios
- Macroeconomic Factors: VIX/TNX z-scores and regime shifts
- Temporal Features: Month, week, weekday, month-end/start
- Correlation/Interaction Features: With SPY and other sectors

## Model Architecture

- Model Type: LightGBM Regressor
- Parameters: Tuned per sector (`max_depth`, `num_leaves`, ...)
- Feature Selection: Top 35% features by importance
- Scaling: StandardScaler used
- Training Period: 2010-2020
- Testing Period: 2021-2025 (Out-of-time validation)

## Performance Metrics 

| Sector | R^2 | RMSE | MAE |
| ------ | --- | ---- | --- |
| `XLK` | 0.794 | 0.00303 | 0.00218 |
| `XLF` | 0.867 | 0.00203 | 0.00141 |
| `XLV` | 0.877 | 0.00140 | 0.00103 |
| `XLI` | 0.851 | 0.00187 | 0.00133 |
| `XLP` | 0.802 | 0.00155 | 0.00113 |
| `XLB` | 0.853 | 0.00208 | 0.00151 |
| `XLU` | 0.814 | 0.00213 | 0.00157 |
| `XLRE` | 0.843 | 0.00219 | 0.00161 |
| `XLY` | 0.841 | 0.00269 | 0.00199 |

Average R^2 across Sectors: **0.8380**
Average RMSE across Sectors: **0.00211**
Average MAE across Sectors: **0.00153**

## Insights

- Best Performing Sector: XLV (Healthcare), providing strong, stable returns.
- Most Difficult Sector: XLK (Tech), higher volatility reduced predictive power.
- Model Strength: A strong, not (over OR under)fit model. Pefroms well across sectors with interpretable outputs.

## Limitations

- Out-of-sample performance may vary in real-time due to regime shifts.
- Model assumes stationarity in macro-financial relationships.
- Results depend on the assumption that past volatility/momentum patterns persist.

## Future work

- Incorporate alternative data (earnings, news sentiment)
- Explore deep learning models (For example: LSTM for time series)
- Add hyperparameter optimization with Optuna or GridSearchCV.

## Visual References

Refer to the `plots/` directory:
- Sector Predictions
- R^2 By Sector
- Sharpe Ratio by Sector
- Feature Importances
- Price Trends and Return Distributions
- Return Correlations Heatmap
  
