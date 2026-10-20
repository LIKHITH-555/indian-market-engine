import io
import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
import duckdb
import tempfile

# --- PATHS & CONSTANTS ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
EOD_PARQUET = os.path.join(DATA_DIR, "eod_stocks.parquet")
SECTOR_PARQUET = os.path.join(DATA_DIR, "sector_indices.parquet")

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

    invalid_rows = (
        (df["OPEN_PRICE"] <= 0)
        | (df["HIGH_PRICE"] <= 0)
        | (df["LOW_PRICE"] <= 0)
        | (df["CLOSE_PRICE"] <= 0)
        | (df["HIGH_PRICE"] < df[["OPEN_PRICE", "LOW_PRICE", "CLOSE_PRICE"]].max(axis=1))
        | (df["LOW_PRICE"] > df[["OPEN_PRICE", "HIGH_PRICE", "CLOSE_PRICE"]].min(axis=1))
        | (df["TTL_TRD_QNTY"] < 0)
        | (df["DELIV_QTY"] < 0)
        | (df["DELIV_QTY"] > df["TTL_TRD_QNTY"])
        | (df["DELIV_PER"] < 0)
        | (df["DELIV_PER"] > 100)
    )

    if invalid_rows.any():
        invalid_symbols = df.loc[invalid_rows, "SYMBOL"].astype(str).tolist()

        print(
            f"[WARN] Dropping {len(invalid_symbols)} NSE rows with "
            f"invalid price, volume, or delivery values: "
            f"{', '.join(invalid_symbols[:10])}"
        )

        df = df.loc[~invalid_rows].copy()

        if df.empty:
            raise RuntimeError(
                f"All NSE rows for {target_date:%Y-%m-%d} failed validation."
            )

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

def get_nifty500_symbols() -> list:
    """Downloads the official Nifty 500 constituent list from NSE Archives."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        df = pd.read_csv(io.StringIO(res.text))
        symbols = [s.strip() for s in df["Symbol"].dropna().unique().tolist() if "DUMMY" not in s]
        print(f"[INFO] Loaded {len(symbols)} symbols from Nifty 500.")
        return symbols
    except Exception as e:
        print(f"[WARN] Could not fetch Nifty 500 list ({e}). Using core list.")
        return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN", "LT", "ITC"]

def backfill_sectors():
    """Downloads 2 years of daily history for all NSE Sectoral Indices."""
    print("\n--- Step 1/3: Backfilling Sectoral Indices (2 Years) ---")
    records = []
    for sector_name, ticker in SECTOR_INDICES.items():
        try:
            hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
            if not hist.empty:
                hist = hist.reset_index()
                hist["TRADE_DATE"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
                hist["SECTOR"] = sector_name
                hist = hist.rename(columns={
                    "Open": "OPEN", "High": "HIGH", "Low": "LOW",
                    "Close": "CLOSE", "Volume": "VOLUME"
                })
                records.append(hist[["TRADE_DATE", "SECTOR", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]])
                print(f"  ✓ {sector_name}: {len(hist)} days")
        except Exception as e:
            print(f"  [WARN] Failed to fetch sector {sector_name}: {e}")
        time.sleep(0.3)

    if records:
        df = pd.concat(records, ignore_index=True)
        upsert_parquet(df, SECTOR_PARQUET, ["TRADE_DATE", "SECTOR"])

def backfill_stock_ohlcv(symbols: list):
    """Batch-downloads 1.5 years of split-adjusted OHLCV for stocks."""
    print(f"\n--- Step 2/3: Backfilling 1.5 Years of Split-Adjusted Prices for {len(symbols)} Stocks ---")
    batch_size = 50
    all_rows = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        yf_tickers = [f"{sym}.NS" for sym in batch]
        print(f"  Downloading batch {i // batch_size + 1}/{(len(symbols) - 1) // batch_size + 1}...")

        try:
            data = yf.download(
            yf_tickers,
            period="18mo",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
            )
            for sym in batch:
                tkr = f"{sym}.NS"
                if tkr not in data.columns.get_level_values(0):
                    continue
                sub = data[tkr].dropna(subset=["Close"]).reset_index()
                if sub.empty:
                    continue

                sub["TRADE_DATE"] = pd.to_datetime(sub["Date"]).dt.tz_localize(None).dt.normalize()
                sub["SYMBOL"] = sym
                sub["SERIES"] = "EQ"
                sub["OPEN_PRICE"] = sub["Open"].round(2)
                sub["HIGH_PRICE"] = sub["High"].round(2)
                sub["LOW_PRICE"] = sub["Low"].round(2)
                sub["CLOSE_PRICE"] = sub["Close"].round(2)
                sub["PREV_CLOSE"] = sub["CLOSE_PRICE"].shift(1).round(2)
                sub["TTL_TRD_QNTY"] = sub["Volume"].fillna(0).astype("int64")
                sub["TURNOVER_LACS"] = ((sub["CLOSE_PRICE"] * sub["TTL_TRD_QNTY"]) / 1e5).round(2)
                sub["NO_OF_TRADES"] = 0
                sub["DELIV_QTY"] = 0
                sub["DELIV_PER"] = 0.0
                sub["AVG_TRADE_SIZE"] = 0.0

                keep_cols = [
                    "TRADE_DATE", "SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE",
                    "LOW_PRICE", "CLOSE_PRICE", "PREV_CLOSE", "TTL_TRD_QNTY",
                    "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER", "AVG_TRADE_SIZE"
                ]
                all_rows.append(sub[keep_cols])
        except Exception as e:
            print(f"  [WARN] Batch failed: {e}")
        time.sleep(1)

    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
        upsert_parquet(full_df, EOD_PARQUET, ["TRADE_DATE", "SYMBOL", "SERIES"])

def backfill_recent_bhavcopies(days_back: int = 20):
    """Overlay recent official NSE bhavcopy data and report failed fetches."""
    print(
        f"\n--- Step 3/3: Overlaying Last {days_back} Calendar Days "
        "of Official NSE Bhavcopy ---"
    )

    today = datetime.now()
    failures = []

    for day_offset in range(days_back, -1, -1):
        target = today - timedelta(days=day_offset)

        if target.weekday() >= 5:
            continue

        try:
            bhav = fetch_nse_bhavcopy(target)

            if bhav.empty:
                print(
                    f"  [INFO] No bhavcopy for "
                    f"{target:%Y-%m-%d} (possibly a holiday or not yet published)."
                )
            else:
                upsert_parquet(
                    bhav,
                    EOD_PARQUET,
                    ["TRADE_DATE", "SYMBOL", "SERIES"],
                )
                print(f"  [OK] Added NSE bhavcopy for {target:%Y-%m-%d}")

        except Exception as exc:
            message = f"{target:%Y-%m-%d}: {exc}"
            failures.append(message)
            print(f"  [ERROR] Failed to fetch NSE bhavcopy: {message}")

        time.sleep(1.0)

    if failures:
        failure_list = "\n".join(f"  - {item}" for item in failures)
        raise RuntimeError(
            "Backfill finished with NSE bhavcopy failures:\n"
            f"{failure_list}"
        )

if __name__ == "__main__":
    backfill_sectors()
    universe = get_nifty500_symbols()
    backfill_stock_ohlcv(universe)
    backfill_recent_bhavcopies(days_back=20)
    print("\n✅ Backfill Complete! Your database is fully populated.")