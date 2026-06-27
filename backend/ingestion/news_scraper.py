import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import NewsArticle

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
AV_BASE_URL = "https://www.alphavantage.co/query"

SYMBOLS = ["AAPL", "BTC-USD"]


def fetch_av_news(symbol: str, limit: int = 10) -> list[dict]:
    """
    Fetches real US market news from Alpha Vantage News Sentiment API.
    Returns structured articles with real sentiment scores.
    
    Alpha Vantage returns news specific to the ticker with
    their own sentiment scores — we store both their score
    and let FinBERT rescore later for comparison.
    """
    # Alpha Vantage uses 'CRYPTO:BTC' format for crypto
    ticker = "CRYPTO:BTC" if symbol == "BTC-USD" else symbol

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHA_VANTAGE_KEY,
    }

    try:
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch Alpha Vantage news for {symbol}: {e}")
        return []

    # Alpha Vantage returns error messages as JSON keys
    if "Information" in data or "Note" in data:
        msg = data.get("Information") or data.get("Note")
        print(f"Alpha Vantage limit hit for {symbol}: {msg}")
        return []

    articles = []
    feed = data.get("feed", [])

    for item in feed:
        try:
            # Parse Alpha Vantage date format: 20240613T143000
            published_at = datetime.strptime(
                item["time_published"], "%Y%m%dT%H%M%S"
            )
        except (ValueError, KeyError):
            published_at = datetime.now()

        # Alpha Vantage provides overall sentiment score per article
        # Range: -1.0 (bearish) to +1.0 (bullish)
        av_score = float(item.get("overall_sentiment_score", 0.0))

        articles.append({
            "symbol": symbol,
            "headline": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "source": item.get("source", "Alpha Vantage"),
            "published_at": published_at,
            # Use Alpha Vantage score directly — FinBERT will
            # rescore when score_unscored_articles() runs
            "sentiment_score": av_score,
        })

    print(f"Found {len(articles)} articles for {symbol} from Alpha Vantage")
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
                sentiment_score=article["sentiment_score"],
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
    articles = fetch_av_news(symbol)
    return save_news(articles)


if __name__ == "__main__":
    for symbol in SYMBOLS:
        scrape_and_save(symbol)