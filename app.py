import os
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="NSE Institutional Research Engine",
    page_icon="📈",
    layout="wide"
)

DATA_DIR = "data"
EOD_PARQUET = os.path.join(DATA_DIR, "eod_stocks.parquet")
SECTORS_PARQUET = os.path.join(DATA_DIR, "ranked_sectors.parquet")
INDUSTRIES_PARQUET = os.path.join(DATA_DIR, "ranked_industries.parquet")
STOCKS_PARQUET = os.path.join(DATA_DIR, "ranked_stocks.parquet")


@st.cache_data(ttl=600)
def load_ranked_data():
    con = duckdb.connect()

    sectors = con.execute(
        f"SELECT * FROM '{SECTORS_PARQUET}'"
    ).df()
    industries = con.execute(
        f"SELECT * FROM '{INDUSTRIES_PARQUET}'"
    ).df()
    stocks = con.execute(
        f"SELECT * FROM '{STOCKS_PARQUET}'"
    ).df()

    required_columns = {
        "sectors": {
            "SECTOR",
            "ROTATION_STAGE",
            "GROUP_SCORE",
            "AVG_RET_1M",
            "AVG_RET_3M",
            "BREADTH_ABOVE_50DMA",
            "RS_1M",
            "RS_3M",
        },
        "industries": {
            "NSE_INDUSTRY",
            "ROTATION_STAGE",
            "GROUP_SCORE",
            "STOCK_COUNT",
            "AVG_RET_1W",
            "AVG_RET_1M",
            "AVG_RET_3M",
            "BREADTH_ABOVE_50DMA",
            "AVG_DELIV_SPIKE",
        },
        "stocks": {
            "TRADE_DATE",
            "SYMBOL",
            "COMPANY_NAME",
            "SECTOR",
            "NSE_INDUSTRY",
            "SIGNAL",
            "MASTER_SCORE",
            "FUND_SCORE",
            "TECH_SCORE",
            "CLOSE_PRICE",
            "RET_1M",
            "RET_3M",
            "RSI_14",
            "DELIV_PER",
            "DELIV_SPIKE_RATIO",
            "FUNDAMENTALS_AVAILABLE",
            "PROMOTER_HOLDING",
            "PROMOTER_AS_OF",
            "ROE",
            "PE_RATIO",
            "DEBT_TO_EQUITY",
            "REV_GROWTH_YOY",
            "EARNINGS_GROWTH_YOY",
            "FUNDAMENTALS_COVERAGE_PCT"
        },
    }

    frames = {
        "sectors": sectors,
        "industries": industries,
        "stocks": stocks,
    }

    for name, frame in frames.items():
        missing = sorted(required_columns[name] - set(frame.columns))

        if missing:
            raise ValueError(
                f"{name.title()} data is missing required columns: "
                + ", ".join(missing)
                + ". Re-run scripts/compute_signals.py."
            )

    return sectors, industries, stocks


@st.cache_data(ttl=600)
def load_stock_history(symbol: str):
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TRADE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE,
               TTL_TRD_QNTY, DELIV_QTY, DELIV_PER
        FROM '{EOD_PARQUET}'
        WHERE SYMBOL = ? AND SERIES = 'EQ'
        ORDER BY TRADE_DATE
    """, [symbol]).df()
    df["SMA_50"] = df["CLOSE_PRICE"].rolling(50).mean()
    df["SMA_200"] = df["CLOSE_PRICE"].rolling(200).mean()
    return df


required_data_files = {
    "EOD price history": EOD_PARQUET,
    "ranked sectors": SECTORS_PARQUET,
    "ranked industries": INDUSTRIES_PARQUET,
    "ranked stocks": STOCKS_PARQUET,
}

missing_files = [
    label
    for label, path in required_data_files.items()
    if not os.path.exists(path)
]

if missing_files:
    st.error(
        "Required data files are missing: "
        + ", ".join(missing_files)
        + ". Run the data backfill and pipeline scripts first."
    )
    st.stop()

try:
    sectors_df, industries_df, stocks_df = load_ranked_data()
except Exception as exc:
    st.error(f"Could not load valid ranked data: {exc}")
    st.stop()

# --- HEADER ---
st.title("🇮🇳 NSE Top-Down Sector, Industry & Stock Intelligence")

latest_trade_date = pd.to_datetime(
    stocks_df["TRADE_DATE"],
    errors="coerce",
).max()

if pd.isna(latest_trade_date):
    st.error("Ranked stock data has no valid trade date. Re-run the data pipeline.")
    st.stop()

latest_trade_date = pd.Timestamp(latest_trade_date).normalize()
latest_date = latest_trade_date.strftime("%d %b %Y")

today_ist = (
    pd.Timestamp.now(tz="Asia/Kolkata")
    .normalize()
    .tz_localize(None)
)

weekdays_since_snapshot = len(
    pd.bdate_range(
        start=latest_trade_date + pd.Timedelta(days=1),
        end=today_ist,
    )
)

st.caption(
    f"**Data Snapshot:** EOD {latest_date} | "
    f"**Weekdays since snapshot:** {weekdays_since_snapshot} | "
    "**Universe:** Nifty 500 | "
    "**Model:** 40% Fundamental + 40% Technical + 20% Sector/Industry Tailwind"
)

if weekdays_since_snapshot > 5:
    st.warning(
        "This stock snapshot is more than five weekdays old. "
        "Signals may be stale; check the latest data pipeline run."
    )

# --- SIDEBAR FILTERS ---
st.sidebar.header("🎯 Screener Filters")
selected_signals = st.sidebar.multiselect(
    "Signal Filter",
    options=stocks_df["SIGNAL"].unique().tolist(),
    default=[
    "🔥 STRONG BUY",
    "✅ ACCUMULATE",
    "👀 WATCHLIST — FUNDAMENTALS UNAVAILABLE",
],
)
selected_sectors = st.sidebar.multiselect(
    "Sector Filter",
    options=sectors_df["SECTOR"].tolist(),
    default=sectors_df["SECTOR"].tolist()
)
min_master = st.sidebar.slider("Minimum Master Score", 0, 100, 60)
min_roe = st.sidebar.slider("Minimum ROE (%)", -20, 50, 10)
min_deliv_spike = st.sidebar.slider("Min Delivery Volume Spike (x Avg)", 0.0, 5.0, 0.0, 0.1)

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "🧭 1. Sector & Industry Rotation",
    "🚀 2. Stock Screener (Fund + Tech)",
    "🔍 3. Company Deep-Dive & Delivery Chart"
])

# ==========================================
# TAB 1: SECTOR & INDUSTRY ROTATION
# ==========================================
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Relative Rotation Map (Sectors vs. Market)")
        fig_rrg = px.scatter(
            sectors_df,
            x="RS_3M",
            y="RS_1M",
            size="BREADTH_ABOVE_50DMA",
            color="ROTATION_STAGE",
            text="SECTOR",
            hover_data=["GROUP_SCORE", "AVG_RET_1M", "AVG_RET_3M", "BREADTH_ABOVE_50DMA"],
            color_discrete_map={
                "🟢 LEADING": "#00C853",
                "🔵 IMPROVING": "#2979FF",
                "🟡 WEAKENING": "#FFD600",
                "🔴 LAGGING": "#FF1744"
            },
            labels={
                "RS_3M": "3-Month Relative Strength vs Market (%)",
                "RS_1M": "1-Month Relative Momentum vs Market (%)"
            },
            height=480
        )
        fig_rrg.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rrg.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_rrg.update_traces(textposition="top center")
        st.plotly_chart(fig_rrg, use_container_width=True)

    with col2:
        st.subheader("Sector Scorecard")
        st.dataframe(
            sectors_df[[
                "SECTOR", "ROTATION_STAGE", "GROUP_SCORE",
                "AVG_RET_1M", "AVG_RET_3M", "BREADTH_ABOVE_50DMA"
            ]],
            use_container_width=True,
            hide_index=True,
            height=440
        )

    st.divider()
    st.subheader("🏭 Top Performing NSE Industries (Drill-Down)")
    st.dataframe(
        industries_df[[
            "NSE_INDUSTRY", "ROTATION_STAGE", "GROUP_SCORE", "STOCK_COUNT",
            "AVG_RET_1W", "AVG_RET_1M", "AVG_RET_3M", "BREADTH_ABOVE_50DMA", "AVG_DELIV_SPIKE"
        ]],
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 2: MASTER STOCK SCREENER
# ==========================================
with tab2:
    missing_fundamentals = ~stocks_df["FUNDAMENTALS_AVAILABLE"]

    filtered = stocks_df[
        (stocks_df["SIGNAL"].isin(selected_signals))
        & (
            stocks_df["SECTOR"].isin(selected_sectors)
            | missing_fundamentals
        )
        & (
            (stocks_df["MASTER_SCORE"] >= min_master)
            | missing_fundamentals
        )
        & (
            (stocks_df["ROE"].fillna(0) >= min_roe)
            | missing_fundamentals
        )
        & (
            stocks_df["DELIV_SPIKE_RATIO"].fillna(0)
            >= min_deliv_spike
        )
    ].copy()

    st.subheader(f"Shortlisted High-Conviction Stocks ({len(filtered)} matches)")
    display_cols = [
        "SYMBOL", "COMPANY_NAME", "SECTOR", "NSE_INDUSTRY",
        "FUNDAMENTALS_AVAILABLE", "PROMOTER_HOLDING", "PROMOTER_AS_OF", "SIGNAL",
        "MASTER_SCORE", "FUND_SCORE", "TECH_SCORE", "CLOSE_PRICE",
        "RET_1M", "RET_3M", "RSI_14", "DELIV_PER", "DELIV_SPIKE_RATIO",
        "ROE", "PE_RATIO", "DEBT_TO_EQUITY", "REV_GROWTH_YOY",
        "EARNINGS_GROWTH_YOY", "FUNDAMENTALS_COVERAGE_PCT",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True, height=600)

# ==========================================
# TAB 3: COMPANY DEEP-DIVE & CHARTS
# ==========================================
with tab3:
    selected_sym = st.selectbox(
        "Select a Stock to Inspect Price, Moving Averages & Institutional Delivery:",
        options=stocks_df["SYMBOL"].tolist()
    )

    row = stocks_df[stocks_df["SYMBOL"] == selected_sym].iloc[0]
    hist_df = load_stock_history(selected_sym)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Signal", row["SIGNAL"])
    m2.metric("Master Score", f"{row['MASTER_SCORE']} / 100")
    m3.metric("Fundamental Score", f"{row['FUND_SCORE']} / 100")
    m4.metric("Technical Score", f"{row['TECH_SCORE']} / 100")
    m5.metric("ROE / P/E", f"{row['ROE']}% | {round(row['PE_RATIO'] or 0, 1)}x")
    m6.metric("Delivery % (Spike)", f"{row['DELIV_PER']}% ({row['DELIV_SPIKE_RATIO']}x)")

    # Candlestick + 50/200 DMA + Delivery % Subplot
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06, row_heights=[0.7, 0.3],
        subplot_titles=(f"{selected_sym} — Price & 50/200-DMA", "Daily Traded Volume & NSE Delivery %")
    )

    fig.add_trace(go.Candlestick(
        x=hist_df["TRADE_DATE"], open=hist_df["OPEN_PRICE"],
        high=hist_df["HIGH_PRICE"], low=hist_df["LOW_PRICE"],
        close=hist_df["CLOSE_PRICE"], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist_df["TRADE_DATE"], y=hist_df["SMA_50"],
        line=dict(color="#2979FF", width=1.5), name="50-DMA"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist_df["TRADE_DATE"], y=hist_df["SMA_200"],
        line=dict(color="#FF9100", width=1.8), name="200-DMA"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=hist_df["TRADE_DATE"], y=hist_df["TTL_TRD_QNTY"],
        name="Total Volume", marker_color="rgba(120, 120, 120, 0.5)"
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=hist_df["TRADE_DATE"], y=hist_df["DELIV_QTY"],
        name="Delivery Volume (NSE)", marker_color="#00C853"
    ), row=2, col=1)

    fig.update_layout(height=650, xaxis_rangeslider_visible=False, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    # Add this inside Tab 3 of app.py
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📰 Recent News & Catalysts")
        st.info(row.get("LATEST_NEWS", "No catalysts available"))
    with c2:
        st.markdown("### 📑 Balance Sheet & Cash Flow Health")
        st.write(f"- **Operating Cash Flow Positive Years:** {row.get('CFO_POSITIVE_YEARS', 'N/A')} / 3")
        st.write(f"- **Interest Coverage Ratio:** {row.get('INTEREST_COVERAGE', 'N/A')}x")
        st.write(f"- **Free Cash Flow Status:** {row.get('FCF_STATUS', 'N/A')}")
        st.write(f"- **Consistent Revenue Growth:** {row.get('CONSISTENT_REVENUE_GROWTH', 'N/A')}")