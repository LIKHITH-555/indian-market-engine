import io
import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
import duckdb

# --- DIRECTORY & STORAGE DEFINITIONS ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

EOD_PARQUET = os.path.join(DATA_DIR, "eod_stocks.parquet")
SECTOR_PARQUET = os.path.join(DATA_DIR, "sector_indices.parquet")

# Sector mapping on Yahoo Finance
SECTOR_INDICES = {
    "NIFTY_50": "^NSEI",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "AUTOBEES.NS",       # Nippon India Nifty Auto ETF (Full 494d history)
    "NIFTY_FMCG": "FMCGIETF.NS",       # ICICI Pru Nifty FMCG ETF (Full 494d history)
    "NIFTY_METAL": "METALIETF.NS",     # ICICI Pru Nifty Metal ETF (Full 494d history)
    "NIFTY_PSU_BANK": "PSUBNKBEES.NS", # Nippon India Nifty PSU Bank ETF (Full 494d history)
    "NIFTY_INFRA": "INFRABEES.NS",     # Nippon India Nifty Infra ETF (Full 494d history)
    "NIFTY_REALTY": "MOREALTY.NS",     # Motilal Oswal Nifty Realty ETF
    "NIFTY_ENERGY": "MOENERGY.NS",     # Motilal Oswal Nifty Energy ETF
    "NIFTY_DEFENCE": "MODEFENCE.NS",   # Bonus: Nifty India Defence ETF
}

def fetch_nse_bhavcopy(target_date: datetime) -> pd.DataFrame:
    """Downloads and cleans NSE Full Bhavcopy (includes Delivery %)."""
    date_str = target_date.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/all-reports",
    }
    session = requests.Session()
    try:
        response = session.get(url, headers=headers, timeout=20)
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()
    except Exception:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(response.text), dtype=str)
    df.columns = [col.strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()

    df = df[df["SERIES"].isin(["EQ", "BE"])].copy()
    df["TRADE_DATE"] = pd.to_datetime(df["DATE1"], format="%d-%b-%Y", errors="coerce")
    df = df[df["TRADE_DATE"] == pd.Timestamp(target_date.date())]

    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
        "LAST_PRICE", "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY",
        "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace("-", "0"), errors="coerce").fillna(0.0)

    df["AVG_TRADE_SIZE"] = (df["TTL_TRD_QNTY"] / df["NO_OF_TRADES"].replace(0, 1)).round(2)

    keep_cols = [
        "TRADE_DATE", "SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE",
        "LOW_PRICE", "CLOSE_PRICE", "PREV_CLOSE", "TTL_TRD_QNTY",
        "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER", "AVG_TRADE_SIZE"
    ]
    return df[keep_cols]

def fetch_sector_indices(days_back: int = 10) -> pd.DataFrame:
    """Pulls recent EOD data for Nifty 50 and Sectoral Indices."""
    records = []
    for sector_name, ticker in SECTOR_INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period=f"{days_back}d", auto_adjust=True)
            if not hist.empty:
                hist = hist.reset_index()
                hist["TRADE_DATE"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
                hist["SECTOR"] = sector_name
                hist = hist.rename(columns={
                    "Open": "OPEN", "High": "HIGH", "Low": "LOW",
                    "Close": "CLOSE", "Volume": "VOLUME"
                })
                records.append(hist[["TRADE_DATE", "SECTOR", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]])
        except Exception:
            pass
        time.sleep(0.3)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def upsert_parquet(new_df: pd.DataFrame, parquet_path: str, key_cols: list):
    """Deduplicates and appends new data into Parquet using DuckDB."""
    if new_df.empty:
        return
    con = duckdb.connect()
    if os.path.exists(parquet_path):
        existing_df = con.execute(f"SELECT * FROM '{parquet_path}'").df()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(by=key_cols).reset_index(drop=True)
    con.register("combined_view", combined)
    con.execute(f"COPY combined_view TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[SAVED] {parquet_path} ({len(combined):,} total rows)")

def repair_corporate_actions(today_bhav: pd.DataFrame, parquet_path: str):
    """Adjusts split/bonus discrepancies against historical data."""
    if today_bhav.empty or not os.path.exists(parquet_path):
        return

    con = duckdb.connect()
    latest_date_str = pd.to_datetime(today_bhav["TRADE_DATE"].iloc[0]).strftime("%Y-%m-%d")
    prev_df = con.execute(f"""
        SELECT SYMBOL, CLOSE_PRICE AS DB_PREV_CLOSE
        FROM '{parquet_path}'
        WHERE TRADE_DATE < '{latest_date_str}' AND SERIES = 'EQ'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY SYMBOL ORDER BY TRADE_DATE DESC) = 1
    """).df()

    if prev_df.empty:
        return

    eq_bhav = today_bhav[today_bhav["SERIES"] == "EQ"][["SYMBOL", "PREV_CLOSE"]]
    merged = pd.merge(eq_bhav, prev_df, on="SYMBOL", how="inner")
    merged["DIFF_PCT"] = ((merged["PREV_CLOSE"] - merged["DB_PREV_CLOSE"]).abs() / merged["DB_PREV_CLOSE"])
    split_symbols = merged[merged["DIFF_PCT"] > 0.25]["SYMBOL"].tolist()

    if not split_symbols:
        return

    print(f"[SPLIT/BONUS DETECTED] Adjusting historical prices for: {split_symbols}")
    full_db = con.execute(f"SELECT * FROM '{parquet_path}'").df()
    full_db["TRADE_DATE"] = pd.to_datetime(full_db["TRADE_DATE"]).dt.normalize()

    for sym in split_symbols:
        try:
            hist = yf.Ticker(f"{sym}.NS").history(period="18mo", auto_adjust=True)
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist["TRADE_DATE"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
            price_sub = hist[["TRADE_DATE", "Open", "High", "Low", "Close"]].copy()
            price_sub.columns = ["TRADE_DATE", "ADJ_OPEN", "ADJ_HIGH", "ADJ_LOW", "ADJ_CLOSE"]

            sym_mask = full_db["SYMBOL"] == sym
            stock_data = full_db[sym_mask].drop(columns=["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"])
            repaired = pd.merge(stock_data, price_sub, on="TRADE_DATE", how="left")
            repaired = repaired.rename(columns={
                "ADJ_OPEN": "OPEN_PRICE",
                "ADJ_HIGH": "HIGH_PRICE",
                "ADJ_LOW": "LOW_PRICE",
                "ADJ_CLOSE": "CLOSE_PRICE"
            })
            for p_col in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"]:
                repaired[p_col] = repaired[p_col].round(2)
            full_db = pd.concat([full_db[~sym_mask], repaired], ignore_index=True)
        except Exception as e:
            print(f"  [WARN] Failed to repair {sym}: {e}")

    con.register("repaired_view", full_db)
    con.execute(f"COPY repaired_view TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

if __name__ == "__main__":
    today = datetime.now()
    print(f"Running EOD pipeline for {today.strftime('%Y-%m-%d')}...")
    bhav_df = fetch_nse_bhavcopy(today)
    if not bhav_df.empty:
        repair_corporate_actions(bhav_df, EOD_PARQUET)
        upsert_parquet(bhav_df, EOD_PARQUET, ["TRADE_DATE", "SYMBOL", "SERIES"])
    
    sector_df = fetch_sector_indices(days_back=10)
    if not sector_df.empty:
        upsert_parquet(sector_df, SECTOR_PARQUET, ["TRADE_DATE", "SECTOR"])