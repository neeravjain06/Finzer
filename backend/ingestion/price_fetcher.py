import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import PriceBar


def fetch_price_data(symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance for a given symbol.
    
    symbol:   ticker string e.g. 'AAPL', 'BTC-USD', 'ETH-USD'
    period:   how far back to fetch — '1d', '5d', '1mo', '3mo', '1y'
    interval: candle size — '1m', '5m', '1h', '1d'
    
    Returns a pandas DataFrame with columns:
    Open, High, Low, Close, Volume — indexed by datetime
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        print(f"No data returned for {symbol}")
        return df
    
    print(f"Fetched {len(df)} rows for {symbol}")
    return df


def save_price_data(symbol: str, df: pd.DataFrame) -> int:
    """
    Takes a DataFrame from fetch_price_data and writes each row
    to the price_bars table as a PriceBar object.
    
    Returns the number of rows saved.
    """
    if df.empty:
        return 0

    saved = 0
    db: Session = SessionLocal()

    try:
        for timestamp, row in df.iterrows():
            # Convert the DataFrame index timestamp to a timezone-naive datetime
            # Postgres stores timestamps without timezone by default
            # pandas returns timezone-aware timestamps from yfinance
            # We strip the timezone info with replace(tzinfo=None)
            if hasattr(timestamp, 'to_pydatetime'):
                dt = timestamp.to_pydatetime().replace(tzinfo=None)
            else:
                dt = timestamp.replace(tzinfo=None)

            # Check if this bar already exists to avoid duplicates
            # If you run the fetcher twice, you don't want duplicate rows
            exists = db.query(PriceBar).filter(
                PriceBar.symbol == symbol,
                PriceBar.timestamp == dt
            ).first()

            if exists:
                continue

            # Create a PriceBar object — one Python object = one database row
            bar = PriceBar(
                symbol=symbol,
                timestamp=dt,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
            )

            # Add to session — staged for saving, not saved yet
            db.add(bar)
            saved += 1

        # Commit — this is when all staged rows actually write to Postgres
        db.commit()
        print(f"Saved {saved} new rows for {symbol}")
        return saved

    except Exception as e:
        # Something went wrong — roll back everything staged in this session
        # This means no partial data gets saved — all or nothing
        db.rollback()
        print(f"Error saving data: {e}")
        raise

    finally:
        db.close()


def fetch_and_save(symbol: str, period: str = "1mo", interval: str = "1d") -> int:
    """
    Convenience function — fetch and save in one call.
    This is what the automation layer will call on a schedule.
    """
    df = fetch_price_data(symbol, period, interval)
    return save_price_data(symbol, df)


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # e.g. python -m backend.ingestion.price_fetcher
    # It does NOT run when the file is imported by another module
    fetch_and_save("AAPL", period="2y", interval="1d")
    fetch_and_save("BTC-USD", period="2y", interval="1d")