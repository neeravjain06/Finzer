import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import NewsArticle

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_BASE_URL = "https://gnews.io/api/v4/search"

# Map our symbols to search terms GNews understands
SYMBOL_QUERIES = {
    "AAPL": "Apple stock",
    "BTC-USD": "Bitcoin crypto",
}


def fetch_gnews(symbol: str, max_articles: int = 10) -> list[dict]:
    """
    Fetches news articles from GNews API for a given symbol.
    Returns a list of article dicts.
    """
    query = SYMBOL_QUERIES.get(symbol, symbol)

    params = {
        "q": query,
        "lang": "en",
        "max": max_articles,
        "apikey": GNEWS_API_KEY,
    }

    try:
        response = requests.get(GNEWS_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch news for {symbol}: {e}")
        return []

    data = response.json()
    articles = []

    for item in data.get("articles", []):
        try:
            published_at = datetime.strptime(
                item["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
            )
        except (ValueError, KeyError):
            published_at = datetime.now()

        articles.append({
            "symbol": symbol,
            "headline": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "source": item.get("source", {}).get("name", "GNews"),
            "published_at": published_at,
        })

    print(f"Found {len(articles)} articles for {symbol}")
    return articles


def save_news(articles: list[dict]) -> int:
    """
    Saves new articles to news_articles table.
    Skips duplicates by URL.
    """
    if not articles:
        return 0

    saved = 0
    db: Session = SessionLocal()

    try:
        for article in articles:
            if not article["url"] or not article["headline"]:
                continue

            exists = db.query(NewsArticle).filter(
                NewsArticle.url == article["url"]
            ).first()

            if exists:
                continue

            news = NewsArticle(
                symbol=article["symbol"],
                headline=article["headline"],
                url=article["url"],
                source=article["source"],
                published_at=article["published_at"],
            )

            db.add(news)
            saved += 1

        db.commit()
        print(f"Saved {saved} new articles")
        return saved

    except Exception as e:
        db.rollback()
        print(f"Error saving articles: {e}")
        raise

    finally:
        db.close()


def scrape_and_save(symbol: str) -> int:
    articles = fetch_gnews(symbol)
    return save_news(articles)


if __name__ == "__main__":
    scrape_and_save("AAPL")
    scrape_and_save("BTC-USD")