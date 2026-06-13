from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .connection import Base


class PriceBar(Base):
    __tablename__ = "price_bars"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    def __repr__(self):
        return f"<PriceBar {self.symbol} @ {self.timestamp} close={self.close}>"


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    headline = Column(Text, nullable=False)
    source = Column(String(100))
    url = Column(Text)
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    sentiment_score = Column(Float, nullable=True)

    def __repr__(self):
        return f"<NewsArticle {self.symbol} — {self.headline[:50]}>"