# Reports Directory

This directory contains structured analytical reports derived from the ML-Driven Sector ETF Prediction pipeline. Reports synthesise quantitative outputs from model evaluation and risk attribution into narrative summaries suitable for review, reproducibility documentation, and investment thesis communication.

---

## Folder Breakdown

| Folder | Description |
|--------|-------------|
| `Model_Reports/` | Detailed technical report covering the hybrid directional-classifier architecture, feature engineering, out-of-sample metrics (accuracy, ROC-AUC, mapped R², directional Sharpe), the walk-forward cross-validation protocol, per-sector performance, and the portfolio backtest. Includes limitations and future directions. **Contains the current live-run metric tables.** |
| `Risk_Assessment_Summary/` | Quantitative risk attribution report providing sector-level analysis of annualised return, annualised volatility (σ × √252), and Sharpe ratio ((μ − r_f) / σ) computed from historical daily returns over the full 2008–2025 sample, benchmarked against passive SPY. Offers portfolio-construction guidance based on risk-adjusted rankings. |
| `Methodology_Enhancements/` | The leakage-audit and framework-derivation report: the four look-ahead leakage fixes, walk-forward validation, the feature-space expansion (ADF stationarity, 252-day macro z-scores, cross-sector spreads), the hybrid classification engine with continuous R² mapping, the two-stage non-blocking plotting architecture, and the verification/leakage-guard tests — each grounded in the QuantFinance knowledge base. |

---

## Related Directories

- `../data/` — Source CSVs (`lgbm_risk_summary.csv`, `sector_model_summary.csv`) underpinning these reports
- `../plots/` — Visualisations referenced within the reports
- `../notebooks/` — Source code that generates all reported metrics
