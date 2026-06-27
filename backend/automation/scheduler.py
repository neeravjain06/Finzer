from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.ingestion.price_fetcher import fetch_and_save
from backend.ingestion.news_scraper import scrape_and_save

SYMBOLS = ["AAPL", "BTC-USD"]


def daily_pipeline():
    """
    Runs the full data pipeline once per day.
    Sentiment and training run as subprocesses to avoid
    PyTorch + XGBoost memory conflicts in the same process.
    """
    logger.info("Starting daily pipeline...")

    # Step 1 — fetch latest price data
    for symbol in SYMBOLS:
        try:
            saved = fetch_and_save(symbol, period="5d", interval="1d")
            logger.info(f"Price fetch {symbol}: {saved} new bars")
        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")

    # Step 2 — fetch latest news
    for symbol in SYMBOLS:
        try:
            saved = scrape_and_save(symbol)
            logger.info(f"News fetch {symbol}: {saved} new articles")
        except Exception as e:
            logger.error(f"News fetch failed for {symbol}: {e}")

    # Step 3 — score new articles with FinBERT (subprocess)
    try:
        subprocess.run(
            [sys.executable, "-m", "backend.ml.sentiment"],
            check=True
        )
        logger.info("Sentiment scoring complete")
    except Exception as e:
        logger.error(f"Sentiment scoring failed: {e}")

    # Step 4 — retrain models on fresh data (subprocess)
    try:
        subprocess.run(
            [sys.executable, "-m", "backend.ml.trainer"],
            check=True
        )
        logger.info("Model retraining complete")
    except Exception as e:
        logger.error(f"Training failed: {e}")

    logger.info("Daily pipeline complete.")


if __name__ == "__main__":
    scheduler = BlockingScheduler()

    scheduler.add_job(
        daily_pipeline,
        CronTrigger(hour=17, minute=0, day_of_week="mon-fri"),
        id="daily_pipeline",
        name="Daily market data pipeline",
        replace_existing=True,
    )

    logger.info("Scheduler started. Pipeline runs weekdays at 5:00 PM ET.")
    logger.info("Running pipeline now for immediate test...")

    daily_pipeline()

    scheduler.start()