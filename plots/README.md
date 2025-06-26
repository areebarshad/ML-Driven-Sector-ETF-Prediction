# Plots Directory

This folder contains plots generated from the S&P 5OO Sector Prediction project. 
The plots are mainly used for model evaluation, performance evaluation, and risk assessment across different sector ETFs.

## Folder Breakdown

| File | Description |
| ---- | ----------- |
| `Feature_Importances/` | Contains plots visualizing the feature importance for each sector. These bar charts indicate the key features that most influence the LightGBM model's predictions for each sector. |
| `Price_Trends_&_Return_Distributions/` | Contains plots showing the price trends and return distributions of sector ETFs over time. These help visualize historical performance and the distribution of sector returns. |
| `Returns_Correlation_Heatmap/` | Contains correlation heatmaps for sector returns, showing how the returns of different sectors relate to each other. This is useful for understanding relationships and interdependencies across sectors. |
| `R^2_By_Sector/` | Contains bar charts showing the R² (coefficient of determination) for each sector model. R² measures how well the model explains the variance in sector returns, with higher values indicating better fit. |
| `Sector_Predictions/` | Contains plots that compare actual vs predicted returns for each sector. These plots visually demonstrate the model's performance over time and its ability to predict sector returns. |
| `Sharpe_Ratio_By_Sector/` | Contains bar plots comparing the Sharpe ratio across different sectors. The Sharpe ratio is a measure of risk-adjusted return, and this folder helps visualize which sectors offer the best risk-return tradeoff. |

## Notes

- All plots are generated from the `LGBM_Prediction_Model/` notebook, using code from the `sector_predictive_analysis/` notebook and the `sector_risk_analsys/` notebook.
- The plot filenames include the sector ticker ('XLK', 'XLV', ...) to easily identify which plot correlates to which sector.

## Related Information

- The data used for generating these plots is stored in the `data/` directory.
- The code for training the model and generating the plots is stored in the `notebooks/` directory.
