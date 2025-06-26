# Notebooks Directory

This folder contains the Google Colab notebooks for the ML-Driven-Sector-ETF-Prediction project using LightGBM.
The notebooks are structured to handle data collection, feature engineering, model training, evaluation, and analysis.

## Notebook Breakdown

| Notebook | Description |
| -------- | ----------- | 
| `LGBM_Prediction_Model/` | Main notebook that trains the LightGBM model on the sector ETFs data and evaluates performance. |
| `LGBM_model_functions/` | Contains reusable functions and classes for the model pipeline, such as data preprocessing, feature engineering, and evaluation. |
| `sector_data_collection/` | Downloads and processes the raw data from Yahoo Finance (`sp500_sector_prices.csv`, VIX, TNX) and prepares it for modeling. |
| `sector_eda/` | Exploratory Data Analysis (EDA) for the sector ETFs, including price trends, return distributions, and correlation heatmaps. |
| `sector_predictive_analysis/` | Analyzes model predictions, including performance metrics (R², RMSE, MAE) and comparison across sectors. |
| `sector_risk_analysis/` | Conducts risk analysis on the sector ETFs, computing Sharpe ratios, annual return, and volatility. |

## Notes 

- All data files required for the notebooks are located in the `data/` directory. The raw sector data is fetched via `yFinance` from Yahoo Finance.
- The notebooks are executed in sequence, starting from data collection, to model training, and finally to risk analysis.
- Output from model evaluation and risk metrics are saved in the `data/` folder as CSV files.

## Related Information

- The data folder contains the raw and processed data used in the notebooks.
- The plots generated during model evaluation and risk analysis are saved in the `plots/` folder.
