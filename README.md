# Indian Market Engine

A Streamlit dashboard and Python pipeline for exploring NSE stocks through price trends, company fundamentals, and sector and industry performance.

The project produces screening signals from heuristic rules. These signals are experimental research indicators, not validated forecasts or personalized investment advice.

## What it does

- Downloads historical stock prices from Yahoo Finance and overlays recent NSE bhavcopy data, including delivery metrics.
- Fetches Nifty 500 company fundamentals and financial statements from Yahoo Finance.
- Reads promoter ownership from a normalized CSV based on official exchange shareholding filings.
- Calculates technical and fundamental scores, ranks sectors and industries, and writes Parquet datasets.
- Displays the results in a Streamlit dashboard with screeners and price charts.

## Requirements

- Python 3.11 or later
- Internet access for NSE and Yahoo Finance data downloads

Install dependencies:

```bash
python -m venv .venv