import io
import os
import time
import urllib.parse
import duckdb
import feedparser
import pandas as pd
import requests
import yfinance as yf
import tempfile
import numpy as np

DATA_DIR = "data"
FUND_PARQUET = os.path.join(DATA_DIR, "fundamentals.parquet")

def get_nifty500_metadata() -> pd.DataFrame:
    """Download and validate the official Nifty 500 constituent list."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"Invalid NSE Nifty 500 CSV: {exc}") from exc

    df.columns = [str(column).strip() for column in df.columns]

    required_columns = {"Symbol", "Company Name", "Industry"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"NSE Nifty 500 CSV is missing columns: {missing}"
        )

    df["Symbol"] = df["Symbol"].astype("string").str.strip()
    df["Company Name"] = df["Company Name"].astype("string").str.strip()
    df["Industry"] = df["Industry"].astype("string").str.strip()

    df = df.dropna(subset=["Symbol"])
    df = df[df["Symbol"] != ""]
    df = df[~df["Symbol"].str.contains("DUMMY", na=False)]

    if df.empty:
        raise RuntimeError("NSE Nifty 500 CSV contained no usable symbols.")

    if df["Symbol"].duplicated().any():
        duplicates = sorted(
            df.loc[df["Symbol"].duplicated(), "Symbol"].unique().tolist()
        )
        raise RuntimeError(
            "NSE Nifty 500 CSV contains duplicate symbols: "
            + ", ".join(duplicates)
        )

    if len(df) < 450:
        raise RuntimeError(
            f"NSE constituent list has only {len(df)} symbols; "
            "expected approximately 500. Refusing to continue."
        )

    return df[["Symbol", "Company Name", "Industry"]].rename(
        columns={
            "Symbol": "SYMBOL",
            "Company Name": "NSE_COMPANY_NAME",
            "Industry": "NSE_INDUSTRY",
        }
    )

def load_promoter_holdings() -> pd.DataFrame:
    """Load normalized promoter holdings from official exchange filings."""
    path = os.path.join(DATA_DIR, "shareholding.csv")
    columns = ["SYMBOL", "PROMOTER_HOLDING", "PROMOTER_AS_OF"]

    if not os.path.exists(path):
        print(
            f"[WARN] {path} not found. "
            "Promoter holdings will be unavailable."
        )
        return pd.DataFrame(columns=columns)

    holdings = pd.read_csv(path, dtype={"SYMBOL": "string"})
    holdings.columns = [str(column).strip().upper() for column in holdings.columns]

    required = {
        "SYMBOL",
        "PROMOTER_HOLDING",
        "AS_OF_DATE",
        "SUBMISSION_DATE",
    }
    missing = required.difference(holdings.columns)

    if missing:
        raise RuntimeError(
            f"{path} is missing columns: "
            + ", ".join(sorted(missing))
        )

    holdings["SYMBOL"] = holdings["SYMBOL"].str.strip().str.upper()
    holdings["PROMOTER_HOLDING"] = pd.to_numeric(
        holdings["PROMOTER_HOLDING"],
        errors="coerce",
    )
    holdings["AS_OF_DATE"] = pd.to_datetime(
        holdings["AS_OF_DATE"],
        errors="coerce",
    )
    holdings["SUBMISSION_DATE"] = pd.to_datetime(
        holdings["SUBMISSION_DATE"],
        errors="coerce",
    )

    invalid_rows = (
        holdings["SYMBOL"].isna()
        | holdings["SYMBOL"].eq("")
        | holdings["AS_OF_DATE"].isna()
        | holdings["PROMOTER_HOLDING"].isna()
        | ~holdings["PROMOTER_HOLDING"].between(0, 100)
    )

    if invalid_rows.any():
        bad_count = int(invalid_rows.sum())
        raise RuntimeError(
            f"{path} contains {bad_count} invalid row(s). "
            "Check symbols, dates, and promoter percentages (0 to 100)."
        )

    latest = (
        holdings.sort_values(
            ["AS_OF_DATE", "SUBMISSION_DATE"],
            na_position="first",
        )
        .drop_duplicates("SYMBOL", keep="last")
        .rename(columns={"AS_OF_DATE": "PROMOTER_AS_OF"})
    )

    today = (
        pd.Timestamp.now(tz="Asia/Kolkata")
        .normalize()
        .tz_localize(None)
    )
    age_days = (today - latest["PROMOTER_AS_OF"]).dt.days
    stale = age_days > 180

    if stale.any():
        stale_count = int(stale.sum())
        print(
            f"[WARN] {stale_count} promoter holding record(s) "
            "are more than 180 days old; marking them unavailable."
        )
        latest.loc[stale, "PROMOTER_HOLDING"] = pd.NA

    return latest[columns]

def analyze_financial_statements(ticker_obj) -> dict:
    """Performs statement health checks on P&L, Balance Sheet, and Cash Flows."""
    res = {
        "CFO_POSITIVE_YEARS": None,
        "CONSISTENT_REVENUE_GROWTH": None,
        "INTEREST_COVERAGE": None,
        "FCF_STATUS": "Unavailable",
    }
    try:
        cf = ticker_obj.cash_flow
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.financials

        # Operating cash flow history
        cfo_series = pd.Series(dtype="float64")

        if not cf.empty and "Operating Cash Flow" in cf.index:
            cfo_series = cf.loc["Operating Cash Flow"].dropna()
            res["CFO_POSITIVE_YEARS"] = int((cfo_series > 0).sum())

        # Free cash flow = operating cash flow minus capital expenditure.
        # Yahoo Finance usually reports capital expenditure as a negative value.
        if (
            not cf.empty
            and "Operating Cash Flow" in cf.index
            and "Capital Expenditure" in cf.index
        ):
            cash_flow = pd.concat(
                [
                    cf.loc["Operating Cash Flow"],
                    cf.loc["Capital Expenditure"],
                ],
                axis=1,
                keys=["CFO", "CAPEX"],
            ).dropna()

            if not cash_flow.empty:
                latest = cash_flow.iloc[0]
                free_cash_flow = latest["CFO"] - abs(latest["CAPEX"])

                if free_cash_flow > 0:
                    res["FCF_STATUS"] = "Positive"
                elif free_cash_flow < 0:
                    res["FCF_STATUS"] = "Negative"
                else:
                    res["FCF_STATUS"] = "Break-even"

        # Revenue consistency check: positive YoY growth in each of
        # the three most recent comparable annual periods.
        if not inc.empty and "Total Revenue" in inc.index:
            revenue = inc.loc["Total Revenue"].dropna()

            # yfinance usually orders annual statement columns newest first.
            if len(revenue) >= 4:
                recent_revenue = revenue.iloc[:4].astype(float).to_numpy()

                current_revenue = recent_revenue[:-1]
                prior_revenue = recent_revenue[1:]

                valid_comparisons = (
                    np.isfinite(current_revenue)
                    & np.isfinite(prior_revenue)
                    & (prior_revenue > 0)
                )

                if valid_comparisons.all():
                    yoy_growth = (
                        current_revenue / prior_revenue
                    ) - 1

                    res["CONSISTENT_REVENUE_GROWTH"] = bool(
                        (yoy_growth > 0).all()
                    )
                else:
                    res["CONSISTENT_REVENUE_GROWTH"] = None

        # Interest Coverage Check (EBIT / Interest Expense)
        if not inc.empty and "EBIT" in inc.index and "Interest Expense" in inc.index:
            ebit = inc.loc["EBIT"].iloc[0]
            interest = abs(inc.loc["Interest Expense"].iloc[0])
            if interest > 0:
                res["INTEREST_COVERAGE"] = round(float(ebit / interest), 2)
            else:
                res["INTEREST_COVERAGE"] = 50.0  # Virtually debt-free
    except Exception:
        pass
    return res

def fetch_company_catalysts(company_name: str) -> str:
    """Pulls recent news headlines via Google RSS to catch triggers."""
    try:
        query = urllib.parse.quote(f"{company_name} stock news NSE India")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        headlines = [entry.title for entry in feed.entries[:3]]
        return " | ".join(headlines) if headlines else "No recent catalysts found"
    except Exception:
        return "No recent catalysts found"

def percent_or_none(info: dict, key: str):
    """Convert a Yahoo ratio to percent, preserving unavailable values."""
    value = info.get(key)

    if value is None or pd.isna(value):
        return None

    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None    

def fetch_fundamentals():
    meta_df = get_nifty500_metadata()
    symbols = meta_df["SYMBOL"].tolist()
    print(f"Fetching deep fundamentals & financial statements for {len(symbols)} Nifty 500 companies...")

    rows = []
    for idx, row in meta_df.iterrows():
        sym = row["SYMBOL"]
        comp_name = row["Company Name"] if "Company Name" in row else row["NSE_COMPANY_NAME"]
        try:
            tkr = yf.Ticker(f"{sym}.NS")
            info = tkr.info
            statements = analyze_financial_statements(tkr)
            latest_news = fetch_company_catalysts(comp_name)

            rows.append(
                {
                    "SYMBOL": sym,
                    "COMPANY_NAME": info.get("longName") or comp_name,
                    "SECTOR": info.get("sector") or row["NSE_INDUSTRY"],
                    "INDUSTRY": info.get("industry") or row["NSE_INDUSTRY"],
                    "NSE_INDUSTRY": row["NSE_INDUSTRY"],
                    "MARKET_CAP_CR": round((info.get("marketCap", 0) or 0) / 1e7, 2),
                    "PE_RATIO": info.get("trailingPE"),
                    "PB_RATIO": info.get("priceToBook"),
                    "DEBT_TO_EQUITY": info.get("debtToEquity"),
                    "ROE": percent_or_none(info, "returnOnEquity"),
                    "ROA": percent_or_none(info, "returnOnAssets"),
                    "REV_GROWTH_YOY": percent_or_none(info, "revenueGrowth"),
                    "EARNINGS_GROWTH_YOY": percent_or_none(info, "earningsGrowth"),
                    "OPERATING_MARGIN": percent_or_none(info, "operatingMargins"),
                    "INSTITUTIONAL_HOLDING": percent_or_none( info,"heldPercentInstitutions",),
                    "INSIDER_HOLDING": percent_or_none(info,"heldPercentInsiders",),
                    # New Balance Sheet & Statement metrics
                    "CFO_POSITIVE_YEARS": statements["CFO_POSITIVE_YEARS"],
                    "CONSISTENT_REVENUE_GROWTH": statements[ "CONSISTENT_REVENUE_GROWTH"],
                    "INTEREST_COVERAGE": statements["INTEREST_COVERAGE"],
                    "FCF_STATUS": statements["FCF_STATUS"],
                    "LATEST_NEWS": latest_news,
                    "UPDATED_AT": pd.Timestamp.now().normalize(),
                }
            )
        except Exception as e:
            print(f"  [SKIP] {sym}: {e}")

        count = len(rows)
        if count > 0 and count % 50 == 0:
            print(f"  ✓ Processed {count}/{len(symbols)} companies...")
            time.sleep(1.0)
        else:
            time.sleep(0.1)

        fund_df = pd.DataFrame(rows)

    if fund_df.empty:
        raise RuntimeError(
            "No company fundamentals were fetched; "
            "the existing fundamentals file was not replaced."
        )

    promoter_df = load_promoter_holdings()
    fund_df = fund_df.merge(
        promoter_df,
        on="SYMBOL",
        how="left",
        validate="one_to_one",
    )

    fetched_symbols = set(fund_df["SYMBOL"])
    expected_symbols = set(meta_df["SYMBOL"])
    missing_symbols = sorted(expected_symbols - fetched_symbols)

    print(
        f"\n[SUMMARY] Fetched fundamentals for {len(fetched_symbols)} "
        f"of {len(expected_symbols)} companies."
    )

    if missing_symbols:
        print(
            f"[WARN] {len(missing_symbols)} companies are missing: "
            f"{', '.join(missing_symbols)}"
        )

    output_dir = os.path.dirname(FUND_PARQUET) or "."
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix="fundamentals-",
            dir=output_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        # DuckDB needs to create the output file itself.
        os.remove(temp_path)

        con = duckdb.connect()
        con.register("fund_view", fund_df)

        # Escape quotes in the generated path before embedding it in SQL.
        sql_temp_path = temp_path.replace("'", "''")

        con.execute(
            f"COPY fund_view TO '{sql_temp_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        os.replace(temp_path, FUND_PARQUET)
        temp_path = None

    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"\n[SAVED] {FUND_PARQUET} ({len(fund_df)} companies)")

if __name__ == "__main__":
    fetch_fundamentals()
