import io
import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
import duckdb
import tempfile

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

# Update this set each year from NSE's official equities holiday list.
NSE_HOLIDAYS = {
    datetime.strptime(date, "%Y-%m-%d").date()
    for date in [
        "2026-01-15",
        "2026-01-26",
        "2026-03-03",
        "2026-03-26",
        "2026-03-31",
        "2026-04-03",
        "2026-04-14",
        "2026-05-01",
        "2026-05-28",
        "2026-06-26",
        "2026-09-14",
        "2026-10-02",
        "2026-10-20",
        "2026-11-10",
        "2026-11-24",
        "2026-12-25",
    ]
}

def fetch_nse_bhavcopy(target_date: datetime) -> pd.DataFrame:
    """Download and validate NSE Full Bhavcopy for one date."""
    date_str = target_date.strftime("%d%m%Y")
    url = (
        "https://nsearchives.nseindia.com/products/content/"
        f"sec_bhavdata_full_{date_str}.csv"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/all-reports",
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)

        if response.status_code == 404:
            return pd.DataFrame()

        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Failed to fetch NSE bhavcopy for "
            f"{target_date:%Y-%m-%d}: {exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Network error fetching NSE bhavcopy for "
            f"{target_date:%Y-%m-%d}: {exc}"
        ) from exc

    try:
        df = pd.read_csv(io.StringIO(response.text), dtype=str)
    except pd.errors.ParserError as exc:
        raise RuntimeError(
            f"Invalid NSE bhavcopy CSV for "
            f"{target_date:%Y-%m-%d}: {exc}"
        ) from exc

    # Strip header whitespace before checking the expected columns.
    df.columns = [str(column).strip() for column in df.columns]

    required_columns = {
        "DATE1",
        "SYMBOL",
        "SERIES",
        "PREV_CLOSE",
        "OPEN_PRICE",
        "HIGH_PRICE",
        "LOW_PRICE",
        "LAST_PRICE",
        "CLOSE_PRICE",
        "AVG_PRICE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER",
    }
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"NSE bhavcopy for {target_date:%Y-%m-%d} "
            f"is missing columns: {missing}"
        )

    for column in df.columns:
        df[column] = df[column].str.strip()

    df = df[df["SERIES"].isin(["EQ", "BE"])].copy()
    df["TRADE_DATE"] = pd.to_datetime(
        df["DATE1"],
        format="%d-%b-%Y",
        errors="coerce",
    )
    df = df[df["TRADE_DATE"] == pd.Timestamp(target_date.date())]

    if df.empty:
        return pd.DataFrame()

    numeric_columns = [
        "PREV_CLOSE",
        "OPEN_PRICE",
        "HIGH_PRICE",
        "LOW_PRICE",
        "LAST_PRICE",
        "CLOSE_PRICE",
        "AVG_PRICE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column].replace("-", "0"),
            errors="coerce",
        ).fillna(0.0)

    df["AVG_TRADE_SIZE"] = (
        df["TTL_TRD_QNTY"]
        / df["NO_OF_TRADES"].replace(0, 1)
    ).round(2)

    output_columns = [
        "TRADE_DATE",
        "SYMBOL",
        "SERIES",
        "OPEN_PRICE",
        "HIGH_PRICE",
        "LOW_PRICE",
        "CLOSE_PRICE",
        "PREV_CLOSE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER",
        "AVG_TRADE_SIZE",
    ]

    return df[output_columns]

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
    """Deduplicate and atomically append data to a Parquet file."""
    if new_df.empty:
        return

    output_dir = os.path.dirname(parquet_path) or "."
    temp_path = None
    con = duckdb.connect()

    try:
        if os.path.exists(parquet_path):
            existing_df = con.execute(
                f"SELECT * FROM '{parquet_path.replace(chr(39), chr(39) * 2)}'"
            ).df()

            combined = pd.concat(
                [existing_df, new_df],
                ignore_index=True,
            )
            combined = combined.drop_duplicates(
                subset=key_cols,
                keep="last",
            )
        else:
            combined = new_df.copy()

        combined = combined.sort_values(
            by=key_cols,
        ).reset_index(drop=True)

        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix="market-data-",
            dir=output_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        # DuckDB must create the Parquet file itself.
        os.remove(temp_path)

        con.register("combined_view", combined)
        try:
            sql_temp_path = temp_path.replace("'", "''")
            con.execute(
                f"COPY combined_view TO '{sql_temp_path}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            con.unregister("combined_view")

        # Replace the existing file only after the new file is complete.
        os.replace(temp_path, parquet_path)
        temp_path = None

        print(f"[SAVED] {parquet_path} ({len(combined):,} total rows)")

    finally:
        con.close()

        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

def repair_corporate_actions(today_bhav: pd.DataFrame, parquet_path: str):
    """
    Do not rewrite historical NSE prices using Yahoo-adjusted data.

    Corporate actions must be handled by a separate, explicitly sourced
    adjustment process. This function remains as a compatibility no-op so
    existing callers do not silently replace historical OHLC values.
    """
    if today_bhav.empty or not os.path.exists(parquet_path):
        return

    print(
        "[INFO] Skipping automatic corporate-action repair. "
        "Historical prices remain unchanged; review corporate actions "
        "before calculating long-range returns."
    )

if __name__ == "__main__":
    today = datetime.now()
    print(f"Running EOD pipeline for {today.strftime('%Y-%m-%d')}...")
    bhav_df = fetch_nse_bhavcopy(today)
    if not bhav_df.empty:
        repair_corporate_actions(bhav_df, EOD_PARQUET)
        upsert_parquet(bhav_df, EOD_PARQUET, ["TRADE_DATE", "SYMBOL", "SERIES"])
    if bhav_df.empty and today.weekday() < 5:
        if today.date() in NSE_HOLIDAYS:
            print(
                f"[INFO] {today:%Y-%m-%d} is an NSE trading holiday. "
                "Skipping the daily pipeline."
            )
            raise SystemExit(78)

        raise RuntimeError(
            f"No NSE bhavcopy data found for {today:%Y-%m-%d}, "
            "and the date is not in the configured NSE holiday list."
        )
    
    sector_df = fetch_sector_indices(days_back=10)
    if not sector_df.empty:
        upsert_parquet(sector_df, SECTOR_PARQUET, ["TRADE_DATE", "SECTOR"])