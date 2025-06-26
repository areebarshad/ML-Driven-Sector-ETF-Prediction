# S%P 500 Sector Prediction: Machine Learning for Financial Forecasting

## Overview 

- The S&P 500 Sector Prediction project aims to evaluate and predict the returns of various S%P 500 sector ETFs.
- The core goal is to identify which sectors offer the best risk-adjusted returns and explore whether predictive models can accurately forecast sector performance using historical price and volatility data.
- This project utilises *LightGBM*, a powerful gradient boosting framework, to build and evaluate return predictive models across multiple sectors.

---

## Project Goals

- **Compare Sector Performance**: Evaluate different S%P 500 sectors in terms of return and risk.
- **Build predictive models**: Use historical price and volatility data to forecast the short-term returns of sector ETFs.
- **Evaluate model performance**: Assess the models based on metrics like R², RMSE, MAE, and Sharpe Ratio.
- **Identify the best models**: Understand which sectors benefit from predictive modeling and where the models are most effective.

---

## Methodology:

### Data Collection
- **Source**: Data was collected using the Yahoo Finance API through the `yfinance` Python library.
- **Time Period**: The historical data spans from **January 1, 2008, to December 31, 2025**.
- **Tickers**: The project covers 10 S&P 500 sector ETFs, as well as **SPY** (S&P 500 Index) for benchmarking.
- **Features**: Various technical features, including **lagged returns**, **moving averages**, **volatility measures**, and **macroeconomic indicators** like **VIX** and **TNX**, are used in model training.

### Feature Engineering
The features engineered include:
- Rolling statistical measures: **mean**, **volatility**, **skew**, and **kurtosis** over specified windows.
- **Momentum** and **volatility ratios** to capture short-term trends.
- **PCA transformation** for reducing multicollinearity in lag features.
- **VIX** and **TNX** data to account for macroeconomic conditions such as market volatility and interest rates.

### Model Training & Evaluation
- **Model**: The **LightGBM Regressor** is used to predict short-term returns.
- **Evaluation Metrics**: The models are evaluated using R², RMSE, MAE, and the **Sharpe ratio** (for risk-adjusted performance).
- **Cross-validation**: Models are trained on data from **2010 to 2020** and evaluated on out-of-time (OOT) samples from **2021 to 2025**.

---

## Results

- The predictive models demonstrated strong performance across various sectors, with **R² scores** ranging from **0.79 to 0.87**, indicating good predictive accuracy in most cases.
- The models also identified sectors with the best **risk-adjusted returns** based on **Sharpe ratios**.

---

## Folder Structure

```plaintext
/
├── data/
│   ├── sp500_sector_prices.csv             
│   ├── sector_model_summary.csv            
│   ├── lgbm_risk_summary.csv
|   ├── README.md                       
├── notebooks/
│   ├── sector_data_collection      
│   ├── sector_eda                  
│   ├── sector_predictive_analysis
│   ├── sector_risk_analysis
|   ├── LGBM_model_functions
|   ├── LGBM_Prediction_Model
|   ├── README.md     
├── plots/
│   ├── Feature_Importances/               
│   ├── Price_Trends_&_Return_Distributions/ 
│   ├── Returns_Correlation_Heatmap/       
│   ├── R²_By_Sector/                     
│   ├── Sector_Predictions/               
│   ├── Sharpe_Ratio_By_Sector/
|   ├── README.md           
├── reports/
│   ├── Model_Reports/                    
│   ├── Risk_Assessment_Reports/          
│   ├── Sharpe_Ratio_Summary/
|   ├── README.md            
└── README.md                             
