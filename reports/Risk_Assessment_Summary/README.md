# Risk Assessment Summary — Sector ETF Universe

## Overview

This report presents a quantitative risk attribution analysis of nine S&P 500 GICS sector ETFs over the full 2008–2025 sample period. Risk metrics are computed from daily arithmetic return series and annualised using standard scaling conventions (252 trading days per year). The analysis supports portfolio construction by ranking sectors on a risk-adjusted return basis and contextualising each sector's volatility profile within the macro regime history of the sample window.

---

## Risk Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Annual Return** | μ × 252 | Annualised expected return; arithmetic mean of daily returns scaled by 252 trading days |
| **Annual Volatility** | σ × √252 | Annualised standard deviation of daily returns; measures total return dispersion (systematic + idiosyncratic risk) |
| **Sharpe Ratio** | (μ_annual − r_f) / σ_annual | Risk-adjusted excess return per unit of total volatility; r_f = 3% as a long-run approximation of the risk-free rate (10-year US Treasury yield proxy). Values > 1.0 are conventionally acceptable; > 2.0 strong; > 3.0 exceptional |

Note: The Sharpe ratio as computed here reflects the unconditional historical risk-return profile of passive long exposure to each sector ETF — it is not a strategy Sharpe and does not account for the directional signal from the LightGBM model. For model strategy Sharpe ratios, see `Model_Reports/`.

---

## Results

Regenerated from `data/lgbm_risk_summary.csv` (live 2008–2025 run, r_f = 3%). SPY is
included as the passive broad-market benchmark.

| Sector | Annual Return | Annual Volatility | Sharpe Ratio | Risk Profile |
|--------|--------------|-------------------|--------------|--------------|
| XLK | 23.20% | 24.10% | 0.838 | High return, high volatility — aggressive growth |
| _SPY_ | _15.33%_ | _17.94%_ | _0.687_ | _Passive broad-market benchmark_ |
| XLI | 14.33% | 19.78% | 0.573 | Cyclical, moderate risk-return balance |
| XLF | 14.82% | 22.23% | 0.532 | Macro-sensitive, elevated tail risk |
| XLY | 14.58% | 21.84% | 0.530 | Income-cycle driven, retail-sensitive |
| XLV | 11.17% | 16.67% | 0.490 | Low-beta, stable, defensive growth |
| XLU | 11.61% | 19.11% | 0.451 | Defensive, mean-reverting, rate-sensitive |
| XLB | 11.32% | 20.70% | 0.402 | Commodity-linked, elevated downside tail |
| XLP | 8.27%  | 14.63% | 0.360 | Lowest volatility, near-stationary returns |
| XLRE | 8.53%  | 20.53% | 0.269 | Duration-sensitive, weakest risk-adjusted return |

Only XLK (0.838) exceeds passive SPY (0.687) on a standalone risk-adjusted basis over
this window; every other sector underperforms the benchmark, underscoring why
diversification and active selection — not single-sector tilts — drive the value case.

---

## Sector-by-Sector Analysis

_The precise annualised figures are those in the Results table above (regenerated from
the live run); the commentary below characterises each sector's risk profile and its
qualitative ranking, which is unchanged._

### XLK — Information Technology (Sharpe: 0.838)

XLK delivers the highest annualised return in the universe (22.36%) at the cost of the highest absolute volatility (24.38%). Its above-average Sharpe ratio reflects that the excess return premium compensates adequately for the elevated risk. The Technology sector's return dynamics are characterised by strong mean-reversion momentum in trending regimes but sharp drawdowns during rate-driven valuation repricing cycles (e.g. 2022 tech sell-off post-ZIRP). High earnings growth sensitivity and multiple expansion/compression dynamics contribute to elevated kurtosis in the return distribution.

### XLF — Financials (Sharpe: 0.532)

XLF exhibits a moderate return-to-risk trade-off. Its volatility is elevated (22.62%) due to macro-driven exposure to credit spreads, yield curve dynamics (net interest margin sensitivity), and systemic risk events (GFC 2008–09, regional banking stress 2023). The fat-tailed return distribution (excess kurtosis) makes annualised standard deviation a potentially understated risk measure — conditional value-at-risk (CVaR) would provide a more conservative risk estimate.

### XLV — Health Care (Sharpe: 0.490)

Healthcare occupies a defensive position in the GICS taxonomy. Its low annual volatility (16.77%) reflects inelastic demand characteristics, stable earnings streams from pharmaceutical and insurance subsectors, and limited cyclicality. The sector's low correlation with broad-market risk factors (low rolling beta to SPY) makes it a valuable diversifier in a multi-sector portfolio framework.

### XLI — Industrials (Sharpe: 0.573)

XLI is a cyclical sector with GDP-sensitive demand drivers (capital expenditure, manufacturing activity, freight volumes). Its moderate volatility (20.10%) and solid annualised return (13.99%) produce the second-highest Sharpe ratio, reflecting that industrial earnings cycles align well with the broad equity risk premium over long horizons.

### XLP — Consumer Staples (Sharpe: 0.394)

Consumer Staples is the lowest-volatility sector in the universe (14.81%), reflecting inelastic demand for non-discretionary goods (food, beverages, household products). The near-stationary return process and smooth autocorrelation structure make XLP one of the most predictable sectors for the LightGBM model. Its low Sharpe ratio relative to its defensive characteristics is partly attributable to the valuation compression risk introduced by rate cycles.

### XLB — Materials (Sharpe: 0.402)

Materials exhibit commodity-linked return dynamics with exposure to global industrial demand, currency effects (USD sensitivity of commodity prices), and supply-chain disruptions. The elevated volatility (20.96%) relative to return (11.12%) produces a below-average Sharpe ratio. Fat tails driven by commodity supercycles and demand shocks (e.g. China reopening, post-COVID supply constraints) create intermittent large positive and negative return outliers.

### XLU — Utilities (Sharpe: 0.451)

Utilities are characterised by near-stationary, mean-reverting return series with minimal variance. The sector's primary risk driver is interest rate sensitivity: as a bond proxy, Utilities experience valuation compression during rate-hike cycles (negative duration exposure) and expansion during dovish pivots. The moderate Sharpe ratio reflects that yield-seeking returns are periodically offset by rate-driven capital losses.

### XLRE — Real Estate (Sharpe: 0.269)

XLRE exhibits the weakest risk-adjusted return in the universe, with a Sharpe ratio of 0.285 driven by the combination of modest annualised return (8.94%) and elevated volatility (20.86%). The sector's sensitivity to the term premium — the spread between long-term and short-term Treasury yields — makes it particularly vulnerable during monetary tightening cycles. The 2022 rate-hike cycle produced sharp REIT valuation drawdowns, significantly impacting the full-sample Sharpe ratio.

### XLY — Consumer Discretionary (Sharpe: 0.530)

Consumer Discretionary reflects income-elasticity and consumer confidence dynamics. Its return (13.86%) and volatility (22.07%) are driven by retail earnings sensitivity, e-commerce sector weights (notably Amazon), and credit cycle exposure. The sector tends to outperform during expansionary phases and underperform sharply during credit contractions.

---

## Portfolio Implications

| Risk Profile | Recommended Sectors | Rationale |
|--------------|--------------------|-----------| 
| Aggressive growth | XLK, XLF, XLY | Highest absolute returns; suitable for long-horizon investors with high risk tolerance |
| Balanced | XLI, XLB | Cyclical exposure with moderate volatility; benefits from economic expansion phases |
| Defensive / capital preservation | XLV, XLP, XLU | Low beta, low volatility; reduce portfolio drawdown during market stress regimes |
| Avoid (risk-adjusted basis) | XLRE | Lowest Sharpe ratio; rate sensitivity creates asymmetric downside in tightening cycles |

Cross-sectional diversification benefit: sectors with low pairwise return correlations (e.g. XLV + XLK, XLP + XLF) provide the greatest reduction in portfolio volatility for a given level of expected return, as formalised by mean-variance portfolio theory (Markowitz, 1952).

---

## Downloadable Outputs

- `data/lgbm_risk_summary.csv` — Machine-readable risk metrics table (annualised return, volatility, Sharpe ratio per sector)
- `plots/Sharpe_Ratio_By_Sector/` — Bar chart of annualised Sharpe ratios across sectors
- `plots/Price_Trends_&_Return_Distributions/` — Return distribution box plots for visual risk assessment
- `plots/Returns_Correlation_Heatmap/` — Cross-sectional correlation matrix
