import os
import duckdb
import numpy as np
import pandas as pd

DATA_DIR = "data"
EOD_PARQUET = os.path.join(DATA_DIR, "eod_stocks.parquet")
FUND_PARQUET = os.path.join(DATA_DIR, "fundamentals.parquet")

OUT_SECTORS = os.path.join(DATA_DIR, "ranked_sectors.parquet")
OUT_INDUSTRIES = os.path.join(DATA_DIR, "ranked_industries.parquet")
OUT_STOCKS = os.path.join(DATA_DIR, "ranked_stocks.parquet")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Wilder's 14-day RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).round(2)


def load_and_calculate_technicals() -> pd.DataFrame:
    """Loads EOD price/delivery data and computes moving averages, RSI, and Delivery spikes."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TRADE_DATE, SYMBOL, CLOSE_PRICE, HIGH_PRICE, LOW_PRICE,
               TTL_TRD_QNTY, TURNOVER_LACS, DELIV_QTY, DELIV_PER
        FROM '{EOD_PARQUET}'
        WHERE SERIES = 'EQ'
        ORDER BY SYMBOL, TRADE_DATE
    """).df()

    print(f"[1/4] Computing technicals & delivery footprints across {df['SYMBOL'].nunique()} stocks...")
    g = df.groupby("SYMBOL", group_keys=False)

    # Moving Averages
    df["SMA_20"] = g["CLOSE_PRICE"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["SMA_50"] = g["CLOSE_PRICE"].transform(lambda x: x.rolling(50, min_periods=25).mean())
    df["SMA_150"] = g["CLOSE_PRICE"].transform(lambda x: x.rolling(150, min_periods=75).mean())
    df["SMA_200"] = g["CLOSE_PRICE"].transform(lambda x: x.rolling(200, min_periods=100).mean())

    # Multi-timeframe Returns (%)
    df["RET_1W"] = g["CLOSE_PRICE"].transform(lambda x: (x / x.shift(5) - 1) * 100)
    df["RET_1M"] = g["CLOSE_PRICE"].transform(lambda x: (x / x.shift(21) - 1) * 100)
    df["RET_3M"] = g["CLOSE_PRICE"].transform(lambda x: (x / x.shift(63) - 1) * 100)
    df["RET_6M"] = g["CLOSE_PRICE"].transform(lambda x: (x / x.shift(126) - 1) * 100)

    # 14-Day RSI
    df["RSI_14"] = g["CLOSE_PRICE"].transform(compute_rsi)

    # 52-Week High Proximity
    df["HIGH_52W"] = g["HIGH_PRICE"].transform(lambda x: x.rolling(252, min_periods=60).max())
    df["PCT_FROM_52W_HIGH"] = ((df["CLOSE_PRICE"] / df["HIGH_52W"]) - 1) * 100

    # Volume & NSE Delivery Accumulation Metrics (using recent valid delivery rows)
    df["VOL_SMA_20"] = g["TTL_TRD_QNTY"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["VOL_RATIO"] = (df["TTL_TRD_QNTY"] / df["VOL_SMA_20"].replace(0, np.nan)).fillna(1.0).round(2)

    # Replace 0 delivery (from historical yfinance rows) with NaN so rolling average only uses true NSE Bhavcopy days
    valid_deliv = df["DELIV_QTY"].replace(0, np.nan)
    df["DELIV_QTY_SMA_15"] = valid_deliv.groupby(df["SYMBOL"]).transform(
        lambda x: x.rolling(15, min_periods=3).mean()
    )
    df["DELIV_SPIKE_RATIO"] = (valid_deliv / df["DELIV_QTY_SMA_15"]).fillna(1.0).round(2)

    # Keep only the latest trading date snapshot for each stock
    latest_df = df.sort_values("TRADE_DATE").groupby("SYMBOL").tail(1).copy()
    return latest_df


def score_fundamentals(df: pd.DataFrame) -> pd.Series:
    """Calculates a 0-100 Fundamental Quality Score including Balance Sheet & Cash Flow audits."""
    score = pd.Series(0.0, index=df.index)
    is_financial = df["SECTOR"].astype(str).str.contains("Financial|Bank", case=False, na=False)

    # 1. Capital Efficiency (Max 25 pts)
    roe = df["ROE"].fillna(0)
    roa = df["ROA"].fillna(0)
    score += np.where(roe >= 20, 15, np.where(roe >= 15, 10, np.where(roe >= 10, 5, 0)))
    score += np.where(
        is_financial,
        np.where(roa >= 1.5, 10, np.where(roa >= 1.0, 6, 0)),
        np.where(df["OPERATING_MARGIN"].fillna(0) >= 18, 10, np.where(df["OPERATING_MARGIN"].fillna(0) >= 12, 6, 0))
    )

    # 2. Earnings & Revenue Statement Growth (Max 30 pts)
    rev_g = df["REV_GROWTH_YOY"].fillna(0)
    eps_g = df["EARNINGS_GROWTH_YOY"].fillna(0)
    consistent_rev = df["CONSISTENT_REVENUE_GROWTH"].fillna(False)
    
    score += np.where(consistent_rev, 15, np.where(rev_g >= 12, 10, 0))
    score += np.where(eps_g >= 25, 15, np.where(eps_g >= 15, 10, np.where(eps_g > 5, 5, 0)))

    # 3. Balance Sheet Solvency & Cash Flow Audit (Max 25 pts)
    dte = df["DEBT_TO_EQUITY"].fillna(100)
    interest_cov = df["INTEREST_COVERAGE"].fillna(0)
    cfo_pos = df["CFO_POSITIVE_YEARS"].fillna(0)

    score += np.where(is_financial, 10, np.where((dte <= 30) & (interest_cov >= 4.0), 10, np.where(dte <= 70, 5, 0)))
    score += np.where(cfo_pos >= 2, 15, np.where(cfo_pos == 1, 8, 0))

    # 4. Valuation & Institutional Skin-in-the-Game (Max 20 pts)
    pe = df["PE_RATIO"].fillna(999)
    score += np.where((pe > 0) & (pe <= 28), 10, np.where((pe > 28) & (pe <= 45), 5, 0))
    strong_hands = df["PROMOTER_HOLDING"].fillna(0) + df["INSTITUTIONAL_HOLDING"].fillna(0)
    score += np.where(strong_hands >= 65, 10, np.where(strong_hands >= 50, 6, 2))

    return score.clip(0, 100).round(1)


def score_technicals(df: pd.DataFrame) -> pd.Series:
    """Calculates a 0-100 Technical & Institutional Accumulation Score."""
    score = pd.Series(0.0, index=df.index)

    # 1. Trend Alignment - Minervini Stage 2 (Max 30 pts)
    above_50 = df["CLOSE_PRICE"] > df["SMA_50"]
    above_150 = df["CLOSE_PRICE"] > df["SMA_150"]
    above_200 = df["CLOSE_PRICE"] > df["SMA_200"]
    ma_stacked = (df["SMA_50"] > df["SMA_150"]) & (df["SMA_150"] > df["SMA_200"])

    score += np.where(above_50, 8, 0)
    score += np.where(above_150, 7, 0)
    score += np.where(above_200, 7, 0)
    score += np.where(ma_stacked, 8, 0)

    # 2. Momentum & Relative StrengthPercentile (Max 30 pts)
    rs_3m_pct = df["RET_3M"].rank(pct=True).fillna(0) * 100
    rs_1m_pct = df["RET_1M"].rank(pct=True).fillna(0) * 100
    score += (rs_3m_pct * 0.18) + (rs_1m_pct * 0.12)

    # 3. 52-Week High Proximity & RSI Sweet Spot (Max 20 pts)
    pct_high = df["PCT_FROM_52W_HIGH"].fillna(-50)
    score += np.where(pct_high >= -5, 10, np.where(pct_high >= -12, 8, np.where(pct_high >= -20, 4, 0)))

    rsi = df["RSI_14"].fillna(50)
    score += np.where((rsi >= 55) & (rsi <= 72), 10, np.where((rsi > 72) & (rsi <= 80), 6, np.where((rsi >= 48) & (rsi < 55), 5, 0)))

    # 4. NSE Delivery Spike & Volume Footprint (Max 20 pts)
    deliv_spike = df["DELIV_SPIKE_RATIO"].fillna(1.0)
    deliv_pct = df["DELIV_PER"].fillna(0)
    pos_day = df["RET_1W"].fillna(0) > 0

    score += np.where(pos_day & (deliv_spike >= 2.0), 12, np.where(pos_day & (deliv_spike >= 1.3), 8, np.where(deliv_spike >= 1.1, 4, 0)))
    score += np.where(deliv_pct >= 60, 8, np.where(deliv_pct >= 45, 5, 2))

    return score.clip(0, 100).round(1)


def rank_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregates stock metrics into Equal-Weighted Sector or Industry Rankings."""
    mkt_1m = df["RET_1M"].median()
    mkt_3m = df["RET_3M"].median()

    grp = df.groupby(group_col).agg(
        STOCK_COUNT=("SYMBOL", "count"),
        AVG_RET_1W=("RET_1W", "mean"),
        AVG_RET_1M=("RET_1M", "mean"),
        AVG_RET_3M=("RET_3M", "mean"),
        BREADTH_ABOVE_50DMA=("CLOSE_PRICE", lambda x: round((x > df.loc[x.index, "SMA_50"]).mean() * 100, 1)),
        BREADTH_ABOVE_200DMA=("CLOSE_PRICE", lambda x: round((x > df.loc[x.index, "SMA_200"]).mean() * 100, 1)),
        AVG_DELIV_SPIKE=("DELIV_SPIKE_RATIO", "mean"),
        AVG_FUND_SCORE=("FUND_SCORE", "mean"),
        AVG_TECH_SCORE=("TECH_SCORE", "mean"),
    ).reset_index()

    # Filter out tiny industries with only 1 stock if ranking industries
    if group_col == "NSE_INDUSTRY":
        grp = grp[grp["STOCK_COUNT"] >= 2].copy()

    # Relative Strength vs Market Benchmark
    grp["RS_1M"] = (grp["AVG_RET_1M"] - mkt_1m).round(2)
    grp["RS_3M"] = (grp["AVG_RET_3M"] - mkt_3m).round(2)

    # Sector/Industry Rotation Score (0-100)
    grp["GROUP_SCORE"] = (
        grp["RS_3M"].rank(pct=True) * 35 +
        grp["RS_1M"].rank(pct=True) * 25 +
        (grp["BREADTH_ABOVE_50DMA"] / 100) * 25 +
        (grp["BREADTH_ABOVE_200DMA"] / 100) * 15
    ).round(1)

    # Assign RRG Rotation Quadrant
    conditions = [
        (grp["RS_3M"] >= 0) & (grp["RS_1M"] >= 0),
        (grp["RS_3M"] < 0) & (grp["RS_1M"] >= 0),
        (grp["RS_3M"] >= 0) & (grp["RS_1M"] < 0),
    ]
    choices = ["🟢 LEADING", "🔵 IMPROVING", "🟡 WEAKENING"]
    grp["ROTATION_STAGE"] = np.select(conditions, choices, default="🔴 LAGGING")

    for col in ["AVG_RET_1W", "AVG_RET_1M", "AVG_RET_3M", "AVG_DELIV_SPIKE", "AVG_FUND_SCORE", "AVG_TECH_SCORE"]:
        grp[col] = grp[col].round(2)

    return grp.sort_values("GROUP_SCORE", ascending=False).reset_index(drop=True)


def run_engine():
    latest_tech = load_and_calculate_technicals()
    con = duckdb.connect()
    fund_df = con.execute(f"SELECT * FROM '{FUND_PARQUET}'").df()

    print("[2/4] Merging Fundamental & Technical datasets...")
    merged = pd.merge(latest_tech, fund_df, on="SYMBOL", how="inner")

    merged["FUND_SCORE"] = score_fundamentals(merged)
    merged["TECH_SCORE"] = score_technicals(merged)

    print("[3/4] Ranking Sectors & Industries (Relative Rotation + Market Breadth)...")
    sector_ranks = rank_groups(merged, "SECTOR")
    industry_ranks = rank_groups(merged, "NSE_INDUSTRY")

    # Map Sector & Industry strength back into individual stocks for the Master Composite Score
    sec_map = sector_ranks.set_index("SECTOR")["GROUP_SCORE"].to_dict()
    ind_map = industry_ranks.set_index("NSE_INDUSTRY")["GROUP_SCORE"].to_dict()
    stage_map = sector_ranks.set_index("SECTOR")["ROTATION_STAGE"].to_dict()

    merged["SECTOR_SCORE"] = merged["SECTOR"].map(sec_map).fillna(50)
    merged["INDUSTRY_SCORE"] = merged["NSE_INDUSTRY"].map(ind_map).fillna(50)
    merged["SECTOR_STAGE"] = merged["SECTOR"].map(stage_map).fillna("🔴 LAGGING")

    # Master Investment Score: 40% Fundamental + 40% Technical + 20% Sector/Industry Tailwind
    merged["MASTER_SCORE"] = (
        merged["FUND_SCORE"] * 0.40 +
        merged["TECH_SCORE"] * 0.40 +
        (merged["SECTOR_SCORE"] * 0.10 + merged["INDUSTRY_SCORE"] * 0.10)
    ).round(1)

    # Actionable Signal Tag
    merged["SIGNAL"] = np.where(
        (merged["MASTER_SCORE"] >= 72) & (merged["TECH_SCORE"] >= 65) & (merged["SECTOR_STAGE"].isin(["🟢 LEADING", "🔵 IMPROVING"])),
        "🔥 STRONG BUY",
        np.where(
            (merged["MASTER_SCORE"] >= 62) & (merged["CLOSE_PRICE"] > merged["SMA_50"]),
            "✅ ACCUMULATE",
            np.where(merged["CLOSE_PRICE"] < merged["SMA_200"], "⚠️ AVOID / WEAK", "👀 WATCHLIST")
        )
    )

    merged = merged.sort_values("MASTER_SCORE", ascending=False).reset_index(drop=True)

    # Save outputs to Parquet
    print("[4/4] Saving ranked tables & printing executive summary...")
    for table_df, path in [(sector_ranks, OUT_SECTORS), (industry_ranks, OUT_INDUSTRIES), (merged, OUT_STOCKS)]:
        con.register("tmp_view", table_df)
        con.execute(f"COPY tmp_view TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # --- PRINT TERMINAL REPORT ---
    print("\n" + "=" * 85)
    print("📊 STAGE 1: TOP SECTORS TO INVEST IN RIGHT NOW")
    print("=" * 85)
    print(sector_ranks[["SECTOR", "ROTATION_STAGE", "GROUP_SCORE", "AVG_RET_1M", "AVG_RET_3M", "BREADTH_ABOVE_50DMA"]].to_string(index=False))

    print("\n" + "=" * 85)
    print("🏭 STAGE 2: TOP 10 LEADING INDUSTRIES (NSE CLASSIFICATION)")
    print("=" * 85)
    print(industry_ranks.head(10)[["NSE_INDUSTRY", "ROTATION_STAGE", "GROUP_SCORE", "STOCK_COUNT", "AVG_RET_1M", "BREADTH_ABOVE_50DMA"]].to_string(index=False))

    print("\n" + "=" * 85)
    print("🚀 STAGE 3: TOP 15 HIGH-CONVICTION STOCKS (FUNDAMENTAL + TECHNICAL + SECTOR TAILWIND)")
    print("=" * 85)
    top_cols = ["SYMBOL", "NSE_INDUSTRY", "SIGNAL", "MASTER_SCORE", "FUND_SCORE", "TECH_SCORE", "CLOSE_PRICE", "DELIV_PER", "DELIV_SPIKE_RATIO"]
    print(merged.head(15)[top_cols].to_string(index=False))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_engine()