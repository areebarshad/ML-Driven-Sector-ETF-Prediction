# Reports Directory

This directory contains structured analytical reports derived from the ML-Driven Sector ETF Prediction pipeline. Reports synthesise quantitative outputs from model evaluation and risk attribution into narrative summaries suitable for review, reproducibility documentation, and investment thesis communication.

---

## Folder Breakdown

| Folder | Description |
|--------|-------------|
| `Model_Reports/` | Detailed technical report covering the LightGBM model architecture, feature engineering methodology, out-of-sample evaluation metrics (R², RMSE, MAE), cross-validation protocol, and per-sector predictive performance analysis. Includes discussion of model limitations, potential sources of residual variance, and directions for future research. |
| `Risk_Assessment_Summary/` | Quantitative risk attribution report providing sector-level analysis of annualised return, annualised volatility (σ × √252), and Sharpe ratio ((μ − r_f) / σ) computed from historical daily return series over the full 2008–2025 sample. Contextualises each sector's risk-return profile within the broader macro regime history and offers portfolio construction guidance based on risk-adjusted performance rankings. |

---

## Related Directories

- `../data/` — Source CSVs (`lgbm_risk_summary.csv`, `sector_model_summary.csv`) underpinning these reports
- `../plots/` — Visualisations referenced within the reports
- `../notebooks/` — Source code that generates all reported metrics
