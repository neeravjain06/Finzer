from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import PriceBar, NewsArticle
from backend.ml.trainer import predict_signal, train_model

# FastAPI() creates the application instance
# This object is what uvicorn runs
app = FastAPI(
    title="AlphaSignal API",
    description="ML-powered trading signal platform",
    version="1.0.0"
)

# CORS — Cross Origin Resource Sharing
# Browsers block requests from one domain to another by default
# Our Next.js frontend runs on localhost:3000
# Our API runs on localhost:8000
# Without CORS middleware, the browser blocks every request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "AlphaSignal API running"}


@app.get("/signals/{symbol}")
def get_signal(symbol: str):
    """
    Returns the current trading signal for a symbol.
    Calls the trained XGBoost model and returns prediction + confidence.
    """
    symbol = symbol.upper()
    result = predict_signal(symbol)

    if result["signal"] == "unknown":
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {symbol}"
        )

    return result


@app.get("/prices/{symbol}")
def get_prices(symbol: str, limit: int = 30):
    """
    Returns the most recent price bars for a symbol.
    limit parameter controls how many bars to return.
    """
    symbol = symbol.upper()
    db = SessionLocal()

    try:
        bars = db.query(PriceBar).filter(
            PriceBar.symbol == symbol
        ).order_by(
            PriceBar.timestamp.desc()
        ).limit(limit).all()

        if not bars:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {symbol}"
            )

        return [{
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        } for bar in bars]

    finally:
        db.close()


@app.get("/news/{symbol}")
def get_news(symbol: str, limit: int = 10):
    """
    Returns the most recent scored news articles for a symbol.
    """
    symbol = symbol.upper()
    db = SessionLocal()

    try:
        articles = db.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.sentiment_score != None
        ).order_by(
            NewsArticle.published_at.desc()
        ).limit(limit).all()

        if not articles:
            raise HTTPException(
                status_code=404,
                detail=f"No news found for {symbol}"
            )

        return [{
            "headline": article.headline,
            "source": article.source,
            "published_at": article.published_at.isoformat(),
            "sentiment_score": article.sentiment_score,
            "url": article.url,
        } for article in articles]

    finally:
        db.close()


@app.post("/train/{symbol}")
def trigger_training(symbol: str):
    """
    Triggers model retraining for a symbol.
    In production this would be called by the scheduler,
    but exposing it as an endpoint lets you trigger it manually.
    """
    symbol = symbol.upper()

    try:
        model = train_model(symbol)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data to train for {symbol}"
            )
        return {"status": "success", "message": f"Model retrained for {symbol}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))