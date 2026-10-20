import feedparser
import urllib.parse

def fetch_company_catalysts(company_name: str) -> list:
    """Fetches recent Google News RSS headlines for a company to detect catalysts."""
    query = urllib.parse.quote(f"{company_name} stock news NSE India")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    headlines = []
    for entry in feed.entries[:5]: # Top 5 recent headlines
        headlines.append({
            "title": entry.title,
            "published": entry.published,
            "link": entry.link
        })
    return headlines