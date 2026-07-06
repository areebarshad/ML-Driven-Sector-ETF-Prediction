"""
app.py — MSPM Factor Engine

An institutional-style quantitative trading terminal for the ML-Driven Sector ETF
Prediction model. It drives the project's real LightGBM hybrid directional pipeline
(notebooks/LGBM_model_functions/LGBM_functions.py) live: it loads the cached price
history, engineers the leakage-free feature matrix, runs the walk-forward hybrid
classifier for the selected sector/horizon, explains the fitted booster with SHAP,
and renders the diagnostics as a dark terminal.

Design notes
------------
- Every expensive step (price load, macro download, feature build, walk-forward
  training, SHAP) is wrapped in @st.cache_data keyed on primitive arguments, so
  toggling a widget never retrains or reloads the dataset.
- The historical *data layer* is served from the committed CSV snapshot so the app
  boots instantly on Streamlit Cloud; the LightGBM training and SHAP attribution
  stay fully dynamic per user selection.
- Macro (VIX / TNX) ingestion is network-bound. It is attempted only when
  st.secrets opts in, and it degrades to a neutral (zero z-score) macro frame on
  any failure so a sandboxed container still produces a forecast.

Run:  streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Import the project's single source of truth for the modelling pipeline ───────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FUNCTIONS_DIR = os.path.join(_HERE, 'notebooks', 'LGBM_model_functions')
if _FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, _FUNCTIONS_DIR)

try:
    import LGBM_functions as lgf
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced in the UI, never crashes
    lgf = None
    _IMPORT_ERROR = exc


# ══ Palette / static config ══════════════════════════════════════════════════════

TERM = {
    "bg":        "#0A0E14",
    "panel":     "#121821",
    "panel_hi":  "#161D28",
    "grid":      "#1E2A38",
    "phosphor":  "#E8B923",   # amber terminal accent (branding only)
    "up":        "#2FD98A",   # long / bullish signal
    "down":      "#FF5C5C",   # short / bearish signal
    "text":      "#C9D4E0",
    "dim":       "#6B7A8D",
}

SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLP": "Consumer Staples", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLY": "Consumer Discretionary",
}

HORIZON_MAP = {"1D": 1, "5D": 5, "21D": 21}
DATA_PATH = os.path.join(_HERE, 'data', 'sp500_sector_prices.csv')


st.set_page_config(
    page_title="MSPM · Factor Engine",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══ Global theme injection ═══════════════════════════════════════════════════════

def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {{
            --bg: {TERM['bg']}; --panel: {TERM['panel']}; --panel-hi: {TERM['panel_hi']};
            --grid: {TERM['grid']}; --phosphor: {TERM['phosphor']};
            --up: {TERM['up']}; --down: {TERM['down']};
            --text: {TERM['text']}; --dim: {TERM['dim']};
            --mono: 'IBM Plex Mono','Courier New',monospace;
            --label: 'Space Grotesk','Segoe UI',sans-serif;
        }}

        .stApp {{ background: var(--bg); color: var(--text); }}
        .block-container {{ padding: 0.6rem 1.4rem 2.2rem 1.4rem; max-width: 1600px; }}
        header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
        #MainMenu, footer {{ visibility: hidden; }}

        html, body, [class*="css"] {{ font-family: var(--mono); }}

        /* Eyebrow section labels */
        .eyebrow {{
            font-family: var(--label); font-size: 0.66rem; font-weight: 600;
            letter-spacing: 0.28em; text-transform: uppercase; color: var(--dim);
            margin: 0.4rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.6rem;
        }}
        .eyebrow::after {{ content:""; flex:1; height:1px; background: var(--grid); }}
        .eyebrow .idx {{ color: var(--phosphor); }}

        /* ── Signal strip (signature) ───────────────────────────────────────── */
        .signal-strip {{
            position: relative; overflow: hidden; border: 1px solid var(--grid);
            background:
              repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0 1px, transparent 1px 3px),
              linear-gradient(90deg, var(--panel-hi), var(--panel));
            padding: 0.9rem 1.3rem; display: flex; align-items: center;
            justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;
        }}
        .brand {{
            font-family: var(--label); font-weight: 700; font-size: 1.05rem;
            letter-spacing: 0.04em; color: var(--phosphor);
            text-shadow: 0 0 14px rgba(232,185,35,0.45); white-space: nowrap;
        }}
        .brand .tick {{ color: var(--dim); font-weight: 500; }}
        .breadcrumb {{
            font-size: 0.74rem; color: var(--dim); letter-spacing: 0.12em;
            white-space: nowrap;
        }}
        .breadcrumb b {{ color: var(--text); font-weight: 600; }}
        .verdict {{ display: flex; align-items: baseline; gap: 0.75rem; white-space: nowrap; }}
        .verdict .dir {{ font-size: 1.7rem; font-weight: 700; letter-spacing: 0.06em; }}
        .verdict .prob {{ font-size: 0.8rem; color: var(--dim); }}
        .verdict .prob b {{ color: var(--text); }}
        .conv-track {{
            width: 150px; height: 6px; background: var(--grid); position: relative;
            margin-top: 4px;
        }}
        .conv-fill {{ position:absolute; top:0; bottom:0; }}
        .up-c {{ color: var(--up); }} .down-c {{ color: var(--down); }}

        /* ── KPI cards ──────────────────────────────────────────────────────── */
        .kpi {{
            border: 1px solid var(--grid); background: var(--panel);
            padding: 0.85rem 1rem 0.9rem 1rem; height: 100%;
            border-top: 2px solid var(--phosphor);
        }}
        .kpi .k-label {{
            font-family: var(--label); font-size: 0.62rem; font-weight: 600;
            letter-spacing: 0.18em; text-transform: uppercase; color: var(--dim);
        }}
        .kpi .k-val {{
            font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin-top: 0.35rem;
            font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
        }}
        .kpi .k-sub {{ font-size: 0.68rem; color: var(--dim); margin-top: 0.15rem;
            font-variant-numeric: tabular-nums; }}
        .kpi .k-sub .pos {{ color: var(--up); }} .kpi .k-sub .neg {{ color: var(--down); }}

        /* Panels around charts */
        .panel-frame {{ border: 1px solid var(--grid); background: var(--panel);
            padding: 0.4rem 0.6rem 0.2rem 0.6rem; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--grid); }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
        .stRadio label, .stSelectbox label, .stSlider label {{
            font-family: var(--label) !important; font-size: 0.66rem !important;
            letter-spacing: 0.16em !important; text-transform: uppercase; color: var(--dim) !important;
        }}

        /* Dataframe → terminal ledger */
        [data-testid="stDataFrame"] {{ border: 1px solid var(--grid); }}
        [data-testid="stDataFrame"] * {{ font-family: var(--mono) !important; }}

        /* Spinner accent */
        .stSpinner > div > div {{ border-top-color: var(--phosphor) !important; }}
        a {{ color: var(--phosphor); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══ Cached data + inference layer ════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_returns():
    """Load the committed price snapshot and return daily arithmetic returns."""
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return data.pct_change().dropna()


@st.cache_data(show_spinner=False)
def get_macro(index_key, allow_live):
    """
    Return a macro z-score frame aligned to the sample.

    Live VIX/TNX ingestion is network-bound, so it is attempted only when the
    operator opts in via st.secrets. Any failure falls back to a neutral
    (zero-shock) macro frame with the exact columns generate_features expects, so
    the pipeline still runs inside a locked-down container.
    """
    returns = load_returns()
    if allow_live:
        try:
            macro = lgf.download_macro_zscores()
            if not macro.dropna(how="all").empty:
                return macro
        except Exception:
            pass  # fall through to neutral frame

    cols = ["vix_zscore", "tnx_zscore", "market_vol_regime"]
    cols += [f"vix_zscore_lag{i}" for i in range(1, 6)]
    cols += [f"tnx_zscore_lag{i}" for i in range(1, 6)]
    neutral = pd.DataFrame(0.0, index=returns.index, columns=cols)
    return neutral


def _shap_global(model, X_explain, max_rows=800):
    """Mean |SHAP| per feature for the fitted classifier, as a tidy DataFrame."""
    import shap

    sample = X_explain.tail(max_rows)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    # LightGBM binary classifiers return either a list [class0, class1] or a
    # single (n, features) / (n, features, classes) array across shap versions.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": sample.columns, "Attribution": mean_abs})
        .sort_values("Attribution", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def run_engine(ticker, horizon, lookback, allow_live_macro):
    """
    Run the full hybrid pipeline for one sector and return only picklable results.

    Cached on the primitive selections, so re-selecting a prior configuration is
    instant and no retraining occurs on unrelated widget changes.
    """
    returns = load_returns()
    macro_z = get_macro(str(returns.index[-1].date()), allow_live_macro)

    features = lgf.generate_features(
        returns, ticker, lgf.TICKERS, macro_z=macro_z, rolling_window=lookback
    )
    y_continuous = lgf.make_forward_target(returns, ticker, horizon=horizon)
    vol_series = returns[ticker].shift(1).rolling(lookback).std()

    res = lgf.train_evaluate_hybrid(features, y_continuous, ticker, vol_series)

    y_test = res["y_test"]
    y_pred = res["y_pred"]
    prob = res["prob"]

    # Realized daily sector return over the out-of-sample window (for the equity view).
    realized = returns[ticker].reindex(y_test.index)

    # Information coefficient: rank correlation of forecast vs realized forward return.
    ic = float(pd.Series(y_pred.values).corr(pd.Series(y_test.values), method="spearman"))

    shap_df = _shap_global(res["model"], res["X_explain"])

    return {
        "accuracy": float(res["accuracy"]),
        "auc": float(res["auc"]),
        "sharpe": float(res["sharpe"]),
        "r2": float(res["r2"]),
        "ic": ic,
        "n_features": int(res["X_explain"].shape[1]),
        "n_obs": int(len(y_test)),
        "y_test": y_test,
        "y_pred": y_pred,
        "prob": prob,
        "realized": realized,
        "shap_df": shap_df,
        "latest_prob": float(prob.iloc[-1]),
        "latest_alpha_bps": float(y_pred.iloc[-1] * 1e4),
        "mean_alpha_bps": float(y_pred.mean() * 1e4),
    }


# ══ Presentational helpers ═══════════════════════════════════════════════════════

def eyebrow(idx, text):
    st.markdown(f'<div class="eyebrow"><span class="idx">{idx}</span>{text}</div>',
                unsafe_allow_html=True)


def kpi_card(label, value, sub_html=""):
    return (
        f'<div class="kpi"><div class="k-label">{label}</div>'
        f'<div class="k-val">{value}</div>'
        f'<div class="k-sub">{sub_html}</div></div>'
    )


def signal_strip(ticker, horizon, lookback, prob):
    """The signature element: a phosphor command-line verdict bar."""
    is_long = prob >= 0.5
    dir_txt = "LONG" if is_long else "SHORT"
    cls = "up-c" if is_long else "down-c"
    col = TERM["up"] if is_long else TERM["down"]
    conviction = abs(prob - 0.5) * 2.0        # 0 (coin-flip) → 1 (max conviction)
    fill_w = 4 + conviction * 96
    fill_from = 50 if is_long else 50 - conviction * 50
    st.markdown(
        f"""
        <div class="signal-strip">
          <div>
            <div class="brand">MSPM <span class="tick">▸</span> FACTOR ENGINE</div>
            <div class="breadcrumb">systematic sector rotation · lgbm hybrid classifier ·
              <b>{len(SECTOR_NAMES)}</b> sectors live</div>
          </div>
          <div class="breadcrumb" style="text-align:center">
            TARGET <b>{ticker}</b> · {SECTOR_NAMES[ticker]}<br>
            HORIZON <b>H+{horizon}</b> · LOOKBACK <b>{lookback}D</b>
          </div>
          <div class="verdict">
            <div>
              <div class="dir {cls}">{dir_txt}</div>
              <div class="conv-track">
                <div class="conv-fill" style="left:{fill_from}%;width:{fill_w/2}%;background:{col};box-shadow:0 0 10px {col}"></div>
              </div>
            </div>
            <div class="prob">P(up)<br><b>{prob:.1%}</b><br>
              <span style="color:{col}">conv {conviction:.0%}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _base_layout(fig, height):
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=TERM["panel"],
        font=dict(family="IBM Plex Mono, monospace", color=TERM["text"], size=11),
        margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=TERM["grid"], zeroline=False, linecolor=TERM["grid"])
    fig.update_yaxes(gridcolor=TERM["grid"], zeroline=False, linecolor=TERM["grid"])
    return fig


def performance_chart(realized, y_pred, ticker):
    """Realized cumulative return vs cumulative model alpha projection (dual axis)."""
    cum_real = (1 + realized.fillna(0)).cumprod() - 1
    cum_alpha = y_pred.cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=cum_real.index, y=cum_real.values * 100, name=f"{ticker} realized (cum %)",
        line=dict(color=TERM["text"], width=1.6), fill="tozeroy",
        fillcolor="rgba(201,212,224,0.06)"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cum_alpha.index, y=cum_alpha.values * 1e4, name="LGBM expected alpha (cum bps)",
        line=dict(color=TERM["phosphor"], width=1.8)), secondary_y=True)

    _base_layout(fig, 360)
    fig.update_yaxes(title_text="Realized cum. return %", secondary_y=False,
                     title_font=dict(size=9, color=TERM["dim"]))
    fig.update_yaxes(title_text="Model alpha (cum bps)", secondary_y=True,
                     showgrid=False, title_font=dict(size=9, color=TERM["phosphor"]))
    return fig


def shap_chart(shap_df, top_n=12):
    top = shap_df.head(top_n).iloc[::-1]
    vmax = top["Attribution"].max() or 1.0
    colors = [
        f"rgba(232,185,35,{0.35 + 0.65 * (v / vmax):.3f})" for v in top["Attribution"]
    ]
    fig = go.Figure(go.Bar(
        x=top["Attribution"], y=top["Feature"], orientation="h",
        marker=dict(color=colors, line=dict(color=TERM["phosphor"], width=0.6)),
        hovertemplate="%{y}<br>mean|SHAP| %{x:.4f}<extra></extra>"))
    _base_layout(fig, 360)
    fig.update_layout(margin=dict(l=8, r=8, t=10, b=8))
    fig.update_xaxes(title_text="mean |SHAP| (factor attribution)",
                     title_font=dict(size=9, color=TERM["dim"]))
    return fig


def build_audit(prob, y_pred, ticker, n=14):
    tail = prob.tail(n).iloc[::-1]
    rows = []
    for ts, p in tail.items():
        if p >= 0.55:
            status = "CONVERGED · LONG"
        elif p <= 0.45:
            status = "CONVERGED · SHORT"
        else:
            status = "PENDING · FLAT"
        rows.append({
            "Timestamp": ts.strftime("%Y-%m-%d"),
            "Asset": ticker,
            "Raw Prediction Value": round(float(p), 4),
            "Mapped Alpha (bps)": round(float(y_pred.loc[ts]) * 1e4, 2),
            "Execution Convergence Status": status,
        })
    return pd.DataFrame(rows)


# ══ App ══════════════════════════════════════════════════════════════════════════

def main():
    inject_theme()

    if lgf is None:
        st.error(
            "Could not import the modelling library `LGBM_functions`. Expected it at "
            f"`{_FUNCTIONS_DIR}`.\n\nUnderlying error: `{_IMPORT_ERROR}`"
        )
        st.stop()

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Price snapshot not found at `{DATA_PATH}`. Run the data-collection "
            "driver (`notebooks/sector_data_collection/LGBM_sector_data_collection.py`) "
            "or `LGBM_functions.download_prices()` to generate it."
        )
        st.stop()

    # Operator opt-in for live macro ingestion (network) via Streamlit secrets.
    try:
        allow_live_macro = bool(st.secrets.get("allow_live_macro", False))
    except Exception:
        allow_live_macro = False

    # ── Sidebar (macro inputs) ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="brand" style="font-size:0.95rem">MSPM <span class="tick">▸</span> CONTROLS</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        ticker = st.selectbox(
            "Target sector ETF",
            options=list(SECTOR_NAMES.keys()),
            format_func=lambda t: f"{t} · {SECTOR_NAMES[t]}",
        )
        horizon_label = st.radio(
            "Forward horizon", options=list(HORIZON_MAP.keys()),
            index=1, horizontal=True,
        )
        horizon = HORIZON_MAP[horizon_label]
        lookback = st.slider(
            "Lookback / lag window (days)", min_value=5, max_value=30, value=10, step=1,
        )

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
        macro_state = "LIVE (VIX/TNX)" if allow_live_macro else "NEUTRAL fallback"
        st.markdown(
            f'<div class="breadcrumb" style="white-space:normal;line-height:1.6">'
            f'ENGINE · walk-forward TimeSeriesSplit(5)<br>'
            f'LEAKAGE GUARDS · per-fold PCA + scaler + ADF<br>'
            f'MACRO FEED · <b>{macro_state}</b></div>',
            unsafe_allow_html=True,
        )
        st.caption("Set `allow_live_macro = true` in secrets to fetch VIX/TNX live.")

    # ── Run the pipeline defensively ─────────────────────────────────────────────
    try:
        returns = load_returns()
    except Exception as exc:
        st.error(f"Failed to read the price snapshot: `{exc}`")
        st.stop()

    if returns.empty or ticker not in returns.columns:
        st.error(
            f"The returns frame is empty or is missing column `{ticker}`. "
            f"Available columns: {list(returns.columns)}."
        )
        st.stop()

    with st.spinner(f"Running walk-forward hybrid classifier for {ticker} · H+{horizon} …"):
        try:
            R = run_engine(ticker, horizon, lookback, allow_live_macro)
        except Exception as exc:
            st.error(
                "The modelling pipeline raised an exception before producing a "
                f"forecast for `{ticker}` (H+{horizon}, lookback {lookback}).\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            st.stop()

    if R["n_obs"] == 0 or R["y_pred"].empty:
        st.warning(
            f"The walk-forward run for {ticker} produced no out-of-sample "
            "observations at this configuration. Try a shorter lookback or a "
            "different horizon."
        )
        st.stop()

    # ── Signature signal strip ───────────────────────────────────────────────────
    signal_strip(ticker, horizon, lookback, R["latest_prob"])

    # ── Row 1 · performance cards ────────────────────────────────────────────────
    eyebrow("01", "Validation Scorecard · out-of-sample")
    edge = (R["accuracy"] - 0.5) * 100
    edge_cls = "pos" if edge >= 0 else "neg"
    alpha_cls = "pos" if R["mean_alpha_bps"] >= 0 else "neg"
    ic_cls = "pos" if R["ic"] >= 0 else "neg"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(
        "LGBM Directional Accuracy", f"{R['accuracy']*100:.1f}%",
        f'<span class="{edge_cls}">{edge:+.1f} pts</span> vs coin-flip · AUC {R["auc"]:.3f}'),
        unsafe_allow_html=True)
    c2.markdown(kpi_card(
        "Active Expected Alpha", f"{R['mean_alpha_bps']:+.1f}",
        f'bps/day mean · latest <span class="{alpha_cls}">{R["latest_alpha_bps"]:+.1f}</span> bps'),
        unsafe_allow_html=True)
    c3.markdown(kpi_card(
        "Information Coefficient", f"{R['ic']:+.3f}",
        f'<span class="{ic_cls}">rank IC</span> · dir. Sharpe {R["sharpe"]:.2f}'),
        unsafe_allow_html=True)
    c4.markdown(kpi_card(
        "Feature Space Dimensions", f"{R['n_features']}",
        f'active factors · {R["n_obs"]:,} OOS obs'),
        unsafe_allow_html=True)

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)

    # ── Row 2 · analytics grid ───────────────────────────────────────────────────
    left, right = st.columns([2, 1])
    with left:
        eyebrow("02", f"Realized vs Projected Alpha · {ticker}")
        st.plotly_chart(performance_chart(R["realized"], R["y_pred"], ticker),
                        use_container_width=True, config={"displayModeBar": False})
    with right:
        eyebrow("03", "SHAP Factor Attribution")
        st.plotly_chart(shap_chart(R["shap_df"]),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Row 3 · audit trail ──────────────────────────────────────────────────────
    eyebrow("04", "Signal Ledger · raw prediction audit trail")
    audit = build_audit(R["prob"], R["y_pred"], ticker)
    st.dataframe(
        audit, use_container_width=True, hide_index=True,
        column_config={
            "Raw Prediction Value": st.column_config.NumberColumn(format="%.4f"),
            "Mapped Alpha (bps)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Continuous R² is negative by construction — a directional classifier "
        "mapped to the return scale is not a magnitude regressor. Accuracy, AUC, "
        "rank IC and directional Sharpe are the honest skill metrics. "
        "Out-of-sample via leakage-free walk-forward TimeSeriesSplit(5)."
    )


if __name__ == "__main__":
    main()
