import pandas as pd
import pandas_ta as ta
import numpy as np
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import PriceBar, NewsArticle


def load_price_data(symbol: str) -> pd.DataFrame:
    """
    Loads all price bars for a symbol from the database
    and returns them as a pandas DataFrame sorted by date.
    """
    db: Session = SessionLocal()

    try:
        rows = db.query(PriceBar).filter(
            PriceBar.symbol == symbol
        ).order_by(PriceBar.timestamp.asc()).all()

        if not rows:
            print(f"No price data found for {symbol}")
            return pd.DataFrame()

        # Convert list of PriceBar objects into a DataFrame
        # Each object becomes one row
        df = pd.DataFrame([{
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        } for row in rows])

        df.set_index("timestamp", inplace=True)
        return df

    finally:
        db.close()


def load_sentiment_data(symbol: str) -> pd.DataFrame:
    """
    Loads all scored news articles for a symbol.
    Groups by date and averages the sentiment scores.
    One sentiment score per day.
    """
    db: Session = SessionLocal()

    try:
        rows = db.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.sentiment_score != None
        ).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date": row.published_at.date(),
            "sentiment_score": row.sentiment_score,
        } for row in rows])

        # Multiple articles per day — average their scores
        # One row per day is what we need to join with price data
        daily_sentiment = df.groupby("date")["sentiment_score"].mean()
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment.columns = ["date", "sentiment_score"]

        return daily_sentiment

    finally:
        db.close()


def compute_features(symbol: str) -> pd.DataFrame:
    """
    Builds the feature matrix for a symbol.

    Features (inputs to the ML model):
    - RSI: momentum indicator — overbought/oversold signal
    - SMA_10: 10-day simple moving average
    - SMA_20: 20-day simple moving average
    - price_momentum: % change over last 5 days
    - volume_change: % change in volume vs previous day
    - sentiment_score: average FinBERT score for that day

    Target (what we're predicting):
    - target: 1 if next day's close is higher, 0 if lower
    """
    price_df = load_price_data(symbol)

    if price_df.empty:
        return pd.DataFrame()

    # --- Technical indicators ---

    # RSI — Relative Strength Index
    # Measures momentum: how fast and how much price has moved
    # Above 70 = overbought (likely to fall)
    # Below 30 = oversold (likely to rise)
    # period=14 means calculated over last 14 days
    price_df["rsi"] = ta.rsi(price_df["close"], length=14)

    # Simple Moving Averages
    # Average closing price over last N days
    # When short MA crosses above long MA = bullish signal
    price_df["sma_10"] = ta.sma(price_df["close"], length=10)
    price_df["sma_20"] = ta.sma(price_df["close"], length=20)

    # Price momentum — % change over last 5 days
    # Positive = upward trend, negative = downward trend
    price_df["price_momentum"] = price_df["close"].pct_change(periods=5)

    # Volume change — % change vs previous day
    # Unusual volume often signals a significant move
    price_df["volume_change"] = price_df["volume"].pct_change(periods=1)

    # --- Target variable ---
    # shift(-1) moves the next row's close into the current row
    # So for each day, next_close is tomorrow's closing price
    price_df["next_close"] = price_df["close"].shift(-1)

    # 1 if tomorrow's price is higher than today's, else 0
    # This is what we're training the model to predict
    price_df["target"] = (
        price_df["next_close"] > price_df["close"]
    ).astype(int)

    # --- Join sentiment ---
    price_df = price_df.reset_index()
    price_df["date"] = price_df["timestamp"].dt.date

    sentiment_df = load_sentiment_data(symbol)

    if not sentiment_df.empty:
        price_df = price_df.merge(sentiment_df, on="date", how="left")
    else:
        price_df["sentiment_score"] = 0.0

    # Fill missing sentiment with 0 (neutral)
    price_df["sentiment_score"] = price_df["sentiment_score"].fillna(0.0)

    # Drop rows with NaN — happens at the edges due to rolling calculations
    # e.g. SMA_20 needs 20 days of data so first 19 rows will be NaN
    feature_cols = [
        "rsi", "sma_10", "sma_20",
        "price_momentum", "volume_change",
        "sentiment_score", "target"
    ]

    df = price_df[["timestamp"] + feature_cols].dropna()

    print(f"Feature matrix for {symbol}: {len(df)} rows, {len(feature_cols)-1} features")
    return df


if __name__ == "__main__":
    df = compute_features("AAPL")
    print(df.tail())

    df2 = compute_features("BTC-USD")
    print(df2.tail())