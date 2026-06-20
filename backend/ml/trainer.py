import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from backend.ml.signals import compute_features

# Directory where trained models get saved
MODEL_DIR = "backend/ml/models"


def prepare_data(symbol: str):
    """
    Loads feature matrix for a symbol and splits it into
    training and test sets.

    Returns X_train, X_test, y_train, y_test
    X = features (inputs)
    y = target (what we're predicting)
    """
    df = compute_features(symbol)

    if df.empty or len(df) < 10:
        print(f"Not enough data to train for {symbol}")
        return None, None, None, None

    # Feature columns — everything except timestamp and target
    feature_cols = [
        "rsi", "sma_10", "sma_20",
        "price_momentum", "volume_change",
        "sentiment_score"
    ]

    X = df[feature_cols]
    y = df["target"]

    # train_test_split splits data into training and test sets
    # test_size=0.2 means 20% of data held out for testing
    # shuffle=False is CRITICAL for time series — we must NOT
    # shuffle the data because future data cannot be used to
    # predict the past. Order must be preserved.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    print(f"{symbol} — train: {len(X_train)} rows, test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def train_model(symbol: str) -> XGBClassifier:
    """
    Trains an XGBoost classifier on the feature matrix
    for a given symbol and saves the model to disk.

    Returns the trained model.
    """
    X_train, X_test, y_train, y_test = prepare_data(symbol)

    if X_train is None:
        return None

    # XGBClassifier — gradient boosted decision tree classifier
    # n_estimators: number of trees to build
    # max_depth: how deep each tree can grow
    # learning_rate: how much each tree corrects the previous one
    # use_label_encoder=False: suppress a deprecation warning
    # eval_metric='logloss': use log loss to evaluate during training
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    # Train the model on training data
    # XGBoost builds 100 trees sequentially, each one correcting
    # the errors of the previous one — this is gradient boosting
    model.fit(X_train, y_train)

    # Evaluate on test data — data the model has NEVER seen
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{symbol} Model Performance:")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred,
          target_names=["Bearish", "Bullish"],
          zero_division=0))

    # Save model to disk so we don't retrain every time
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f"{MODEL_DIR}/{symbol}_xgb.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")
    return model


def load_model(symbol: str) -> XGBClassifier:
    """
    Loads a previously trained model from disk.
    Returns None if no model exists yet.
    """
    model_path = f"{MODEL_DIR}/{symbol}_xgb.pkl"

    if not os.path.exists(model_path):
        print(f"No trained model found for {symbol}")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {model_path}")
    return model


def predict_signal(symbol: str) -> dict:
    """
    Loads the latest features for a symbol and runs the
    trained model to produce a trading signal.

    Returns a dict with symbol, signal, and confidence.
    """
    model = load_model(symbol)

    if model is None:
        return {"symbol": symbol, "signal": "unknown", "confidence": 0.0}

    df = compute_features(symbol)

    if df.empty:
        return {"symbol": symbol, "signal": "unknown", "confidence": 0.0}

    feature_cols = [
        "rsi", "sma_10", "sma_20",
        "price_momentum", "volume_change",
        "sentiment_score"
    ]

    # Take the most recent row — latest available features
    latest = df[feature_cols].iloc[[-1]]

    # predict_proba returns probability for each class
    # [0] = probability of bearish, [1] = probability of bullish
    proba = model.predict_proba(latest)[0]
    prediction = model.predict(latest)[0]

    signal = "bullish" if prediction == 1 else "bearish"
    confidence = float(proba[prediction])

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(confidence, 3),
        "bullish_prob": round(float(proba[1]), 3),
        "bearish_prob": round(float(proba[0]), 3),
    }


if __name__ == "__main__":
    # Train models
    print("Training AAPL model...")
    train_model("AAPL")

    print("\nTraining BTC-USD model...")
    train_model("BTC-USD")

    # Generate signals
    print("\n--- Current Signals ---")
    print(predict_signal("AAPL"))
    print(predict_signal("BTC-USD"))