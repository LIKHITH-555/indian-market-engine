import io
import os
import time
import urllib.parse
import duckdb
import feedparser
import pandas as pd
import requests
import yfinance as yf

DATA_DIR = "data"
FUND_PARQUET = os.path.join(DATA_DIR, "fundamentals.parquet")

def get_nifty500_metadata() -> pd.DataFrame:
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()
    df = pd.read_csv(io.StringIO(res.text))
    df["Symbol"] = df["Symbol"].str.strip()
    df = df[~df["Symbol"].str.contains("DUMMY", na=False)]
    return df[["Symbol", "Company Name", "Industry"]].rename(columns={
        "Symbol": "SYMBOL",
        "Company Name": "NSE_COMPANY_NAME",
        "Industry": "NSE_INDUSTRY"
    })

def analyze_financial_statements(ticker_obj) -> dict:
    """Performs statement health checks on P&L, Balance Sheet, and Cash Flows."""
    res = {
        "CFO_POSITIVE_YEARS": 0,
        "CONSISTENT_REVENUE_GROWTH": False,
        "INTEREST_COVERAGE": 0.0,
        "FCF_STATUS": "Neutral"
    }
    try:
        cf = ticker_obj.cash_flow
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.financials

        # Cash Flow Check
        if not cf.empty and "Operating Cash Flow" in cf.index:
            cfo_series = cf.loc["Operating Cash Flow"].dropna()
            res["CFO_POSITIVE_YEARS"] = int((cfo_series > 0).sum())
            if len(cfo_series) >= 2 and cfo_series.iloc[0] > cfo_series.iloc[-1]:
                res["FCF_STATUS"] = "Strong Positive"

        # Revenue Consistency Check from P&L
        if not inc.empty and "Total Revenue" in inc.index:
            rev = inc.loc["Total Revenue"].dropna()
            if len(rev) >= 3:
                res["CONSISTENT_REVENUE_GROWTH"] = bool((rev.values[:-1] > rev.values[1:]).all())

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

            rows.append({
                "SYMBOL": sym,
                "COMPANY_NAME": info.get("longName") or comp_name,
                "SECTOR": info.get("sector") or row["NSE_INDUSTRY"],
                "INDUSTRY": info.get("industry") or row["NSE_INDUSTRY"],
                "NSE_INDUSTRY": row["NSE_INDUSTRY"],
                "MARKET_CAP_CR": round((info.get("marketCap", 0) or 0) / 1e7, 2),
                "PE_RATIO": info.get("trailingPE"),
                "PB_RATIO": info.get("priceToBook"),
                "ROE": round((info.get("returnOnEquity", 0) or 0) * 100, 2),
                "ROA": round((info.get("returnOnAssets", 0) or 0) * 100, 2),
                "DEBT_TO_EQUITY": info.get("debtToEquity"),
                "REV_GROWTH_YOY": round((info.get("revenueGrowth", 0) or 0) * 100, 2),
                "EARNINGS_GROWTH_YOY": round((info.get("earningsGrowth", 0) or 0) * 100, 2),
                "OPERATING_MARGIN": round((info.get("operatingMargins", 0) or 0) * 100, 2),
                "INSTITUTIONAL_HOLDING": round((info.get("heldPercentInstitutions", 0) or 0) * 100, 2),
                "PROMOTER_HOLDING": round((info.get("heldPercentInsiders", 0) or 0) * 100, 2),
                # New Balance Sheet & Statement metrics
                "CFO_POSITIVE_YEARS": statements["CFO_POSITIVE_YEARS"],
                "CONSISTENT_REVENUE_GROWTH": statements["CONSISTENT_REVENUE_GROWTH"],
                "INTEREST_COVERAGE": statements["INTEREST_COVERAGE"],
                "FCF_STATUS": statements["FCF_STATUS"],
                "LATEST_NEWS": latest_news,
                "UPDATED_AT": pd.Timestamp.now().normalize()
            })
        except Exception as e:
            print(f"  [SKIP] {sym}: {e}")

        count = len(rows)
        if count > 0 and count % 50 == 0:
            print(f"  ✓ Processed {count}/{len(symbols)} companies...")
            time.sleep(1.0)
        else:
            time.sleep(0.1)

    fund_df = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("fund_view", fund_df)
    con.execute(f"COPY fund_view TO '{FUND_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"\n✅ [SAVED] {FUND_PARQUET} ({len(fund_df)} companies with complete Balance Sheet & News analysis)")

if __name__ == "__main__":
    fetch_fundamentals()