from transformers import pipeline
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal
from backend.db.models import NewsArticle


def load_model():
    """
    Loads the FinBERT sentiment analysis pipeline.

    FinBERT is a BERT model pre-trained on financial text.
    'pipeline' is HuggingFace's high-level API — it wraps the model
    and tokenizer into one object you can call directly with text.

    task: 'text-classification' — classify text into categories
    model: 'ProsusAI/finbert' — the specific model on HuggingFace Hub

    First run downloads the model (~400MB) and caches it locally.
    Subsequent runs load from cache instantly.
    """
    print("Loading FinBERT model...")
    sentiment_pipeline = pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
    )
    print("Model loaded.")
    return sentiment_pipeline


def score_headline(pipeline, headline: str) -> float:
    """
    Runs a single headline through FinBERT and returns a float score.

    FinBERT returns one of three labels:
    - 'positive' — bullish sentiment
    - 'negative' — bearish sentiment
    - 'neutral'  — no strong signal

    We convert this to a float between -1.0 and 1.0:
    - positive → +score  (e.g. +0.92)
    - negative → -score  (e.g. -0.87)
    - neutral  → 0.0

    The 'score' field from FinBERT is the model's confidence (0 to 1).
    Multiplying by the direction gives us a signed sentiment score.
    """
    # pipeline() returns a list of dicts, one per input
    # e.g. [{'label': 'positive', 'score': 0.923}]
    result = pipeline(headline[:512])[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        return confidence
    elif label == "negative":
        return -confidence
    else:
        return 0.0


def score_unscored_articles() -> int:
    """
    Reads all NewsArticle rows where sentiment_score IS NULL,
    scores each headline with FinBERT, and writes the score back.

    Returns the number of articles scored.
    """
    db: Session = SessionLocal()
    scored = 0

    try:
        # Query only articles that haven't been scored yet
        # This is the two-phase write pattern — scraper saves raw data,
        # this function enriches it with ML scores
        unscored = db.query(NewsArticle).filter(
            NewsArticle.sentiment_score == None
        ).all()

        if not unscored:
            print("No unscored articles found.")
            return 0

        print(f"Scoring {len(unscored)} articles...")

        # Load model once — expensive operation, don't load inside the loop
        model = load_model()

        for article in unscored:
            score = score_headline(model, article.headline)

            # Write the score directly onto the SQLAlchemy object
            # SQLAlchemy tracks this change automatically
            article.sentiment_score = score
            scored += 1

            print(f"[{score:+.3f}] {article.headline[:60]}")

        # One commit saves all changes at once
        db.commit()
        print(f"\nScored {scored} articles successfully.")
        return scored

    except Exception as e:
        db.rollback()
        print(f"Error scoring articles: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    score_unscored_articles()