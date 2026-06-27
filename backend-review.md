# Backend Review

This is a read-only review of the backend folder. No backend source files were changed.

## Overall Read

The backend is organized cleanly into four concerns:

- `backend/api` exposes the FastAPI endpoints.
- `backend/db` owns the SQLAlchemy engine, session factory, and models.
- `backend/ingestion` pulls in market and news data.
- `backend/ml` builds features, trains the classifier, and serves predictions.

The codebase is coherent and the data flow is easy to follow: ingest raw data, store it in Postgres, derive features, train XGBoost, then expose predictions through the API.

## Concrete Errors / Issues

1. `requirement.txt` is not install-safe as written.
   - `yfinance==1.4.1.` has a trailing period, which can break pip parsing.
   - Several imported backend dependencies are missing from the file, including `fastapi`, `pandas_ta`, `xgboost`, `scikit-learn`, and `transformers`.
   - This means a fresh environment will likely fail before the backend can run.

2. `backend/api/main.py` has one lint-style issue reported by the tool in the `/prices/{symbol}` query block.
   - The tool classified it as a Sourcery suggestion rather than a syntax or runtime error.
   - I did not find a real compile-time problem in the backend Python files.

3. `backend/db/connection.py` creates the SQLAlchemy engine at import time.
   - If the database environment variables are missing or invalid, importing the module can fail immediately.
   - That is acceptable for a controlled deployment, but it is a startup risk.

4. `backend/ml/sentiment.py` loads FinBERT on demand.
   - The first scoring call will be slow and memory-heavy because the model must be downloaded and initialized.
   - This is an operational cost, not a code error.

## File-by-File Summary

- `backend/api/main.py`: FastAPI routes for health, signals, prices, news, and training.
- `backend/db/connection.py`: Builds the Postgres engine and session factory from environment variables.
- `backend/db/models.py`: Defines `PriceBar` and `NewsArticle` tables.
- `backend/ingestion/price_fetcher.py`: Fetches Yahoo Finance OHLCV data and stores it in Postgres.
- `backend/ingestion/news_scraper.py`: Fetches GNews articles and stores unique items by URL.
- `backend/ml/sentiment.py`: Scores stored headlines with FinBERT and writes sentiment values back to the database.
- `backend/ml/signals.py`: Builds the feature matrix from price and sentiment data.
- `backend/ml/trainer.py`: Trains and loads XGBoost models, then produces bullish or bearish predictions.

## Bottom Line

The backend structure is solid, but the dependency file needs attention before deployment. The main actionable problem is the incomplete and slightly malformed `requirement.txt`; after that, the backend should be much closer to reproducible startup.