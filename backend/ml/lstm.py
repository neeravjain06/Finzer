import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from backend.ml.signals import compute_features

MODEL_DIR = "backend/ml/models"
SEQUENCE_LENGTH = 30  # how many days the LSTM looks back


class PriceSequenceDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.

    Instead of feeding the model one row at a time like XGBoost,
    we feed it sequences of SEQUENCE_LENGTH consecutive days.

    For each position i in the data, we create:
    X[i] = features from day i to day i+SEQUENCE_LENGTH
    y[i] = target for day i+SEQUENCE_LENGTH (what we're predicting)

    This is called a sliding window approach.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    Two-layer LSTM for binary price direction classification.

    Architecture:
    Input (batch, sequence_length, n_features)
        ↓
    LSTM Layer 1: 64 hidden units
        ↓
    Dropout 0.2
        ↓
    LSTM Layer 2: 32 hidden units
        ↓
    Dropout 0.2
        ↓
    Dense Layer: 16 units, ReLU
        ↓
    Dense Layer: 1 unit, Sigmoid
        ↓
    Output: probability (0-1)
    """

    def __init__(self, input_size: int, hidden_size1: int = 64,
                 hidden_size2: int = 32, dropout: float = 0.2):
        # Always call super().__init__() first in PyTorch modules
        # This sets up all the internal PyTorch machinery
        super(LSTMModel, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        # First LSTM layer
        # input_size: number of features per time step (6)
        # hidden_size: number of LSTM units (neurons)
        # batch_first=True: input shape is (batch, seq, features)
        # not (seq, batch, features) — more intuitive
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer
        # input_size is now hidden_size1 — takes output of first LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            batch_first=True,
        )

        self.dropout2 = nn.Dropout(dropout)

        # Dense layers
        # Takes the final hidden state of LSTM2 (hidden_size2 values)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()

        # Final output: 1 value, sigmoid squashes it to 0-1
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass — defines how data flows through the network.
        PyTorch calls this automatically when you do model(input).

        x shape: (batch_size, sequence_length, input_size)
        """
        # LSTM1 returns output for every time step + final hidden state
        # We only need the output sequence to feed into LSTM2
        # out1 shape: (batch_size, sequence_length, hidden_size1)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        # LSTM2 takes the full output sequence from LSTM1
        # out2 shape: (batch_size, sequence_length, hidden_size2)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)

        # Take only the LAST time step's output
        # This represents the network's final state after seeing
        # all 30 days — the most informed prediction
        # Shape: (batch_size, hidden_size2)
        last_hidden = out2[:, -1, :]

        # Pass through dense layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        # Squeeze removes the extra dimension: (batch, 1) → (batch,)
        return out.squeeze()


def prepare_sequences(symbol: str):
    """
    Builds sliding window sequences from the feature matrix.

    Returns X_train, X_test, y_train, y_test as numpy arrays.
    X shape: (n_samples, SEQUENCE_LENGTH, n_features)
    y shape: (n_samples,)
    """
    df = compute_features(symbol)

    if df.empty or len(df) < SEQUENCE_LENGTH + 10:
        print(f"Not enough data for {symbol} LSTM")
        return None, None, None, None

    feature_cols = [
        "rsi", "sma_10", "sma_20",
        "price_momentum", "volume_change",
        "sentiment_score"
    ]

    X_raw = df[feature_cols].values
    y_raw = df["target"].values

    # Scale features to [0, 1] range
    # Neural networks train much better on normalised data
    # Large unscaled values (like SMA_20 = 300) cause exploding gradients
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Build sliding windows
    X_sequences = []
    y_sequences = []

    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        # Window: rows i to i+SEQUENCE_LENGTH
        X_sequences.append(X_scaled[i:i + SEQUENCE_LENGTH])
        # Target: the day after the window ends
        y_sequences.append(y_raw[i + SEQUENCE_LENGTH])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Train/test split — no shuffling, preserve time order
    split = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split]
    X_test = X_sequences[split:]
    y_train = y_sequences[:split]
    y_test = y_sequences[split:]

    print(f"{symbol} LSTM — train: {len(X_train)} sequences, test: {len(X_test)} sequences")
    print(f"Sequence shape: {X_train[0].shape}")

    return X_train, X_test, y_train, y_test, scaler


def train_lstm(symbol: str, epochs: int = 50, lr: float = 0.001) -> LSTMModel:
    """
    Trains the LSTM model and saves it to disk.

    epochs: how many times to pass through the full training set
    lr: learning rate — how big each gradient descent step is
    """
    result = prepare_sequences(symbol)

    if result[0] is None:
        return None

    X_train, X_test, y_train, y_test, scaler = result

    # Create PyTorch datasets and dataloaders
    # DataLoader handles batching — instead of passing all data at once,
    # we pass small batches of 32 sequences at a time
    # This is called mini-batch gradient descent
    train_dataset = PriceSequenceDataset(X_train, y_train)
    test_dataset = PriceSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialise model
    n_features = X_train.shape[2]  # 6 features
    model = LSTMModel(input_size=n_features)

    # Loss function: binary cross entropy
    # Measures how wrong our probability predictions are
    # Perfect prediction → loss near 0
    # Terrible prediction → loss near infinity
    criterion = nn.BCELoss()

    # Adam optimiser — adaptive learning rate gradient descent
    # Better than plain SGD for most deep learning tasks
    # It adapts the learning rate per parameter based on history
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining LSTM for {symbol}...")
    model.train()  # sets model to training mode (enables dropout)

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            # Zero gradients from previous batch
            # If you don't do this, gradients accumulate
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)

            # Calculate loss
            loss = criterion(predictions, y_batch)

            # Backward pass — compute gradients via backpropagation
            loss.backward()

            # Gradient clipping — prevents exploding gradients in RNNs
            # If any gradient exceeds 1.0, scale them all down
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    # Evaluate on test set
    model.eval()  # sets model to evaluation mode (disables dropout)

    all_preds = []
    all_targets = []

    with torch.no_grad():  # don't compute gradients during evaluation
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            # Convert probabilities to binary predictions
            # threshold: 0.5 — above = bullish, below = bearish
            binary_preds = (predictions > 0.5).float()
            all_preds.extend(binary_preds.numpy())
            all_targets.extend(y_batch.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n{symbol} LSTM Accuracy: {accuracy:.2%}")

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = f"{MODEL_DIR}/{symbol}_lstm.pkl"
    scaler_path = f"{MODEL_DIR}/{symbol}_lstm_scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"LSTM saved to {model_path}")
    return model


def predict_lstm(symbol: str) -> dict:
    """
    Loads the trained LSTM and predicts signal for the most recent
    30-day sequence.
    """
    model_path = f"{MODEL_DIR}/{symbol}_lstm.pkl"
    scaler_path = f"{MODEL_DIR}/{symbol}_lstm_scaler.pkl"

    if not os.path.exists(model_path):
        return {"symbol": symbol, "signal": "unknown", "confidence": 0.0}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = compute_features(symbol)

    if df.empty or len(df) < SEQUENCE_LENGTH:
        return {"symbol": symbol, "signal": "unknown", "confidence": 0.0}

    feature_cols = [
        "rsi", "sma_10", "sma_20",
        "price_momentum", "volume_change",
        "sentiment_score"
    ]

    # Take the most recent SEQUENCE_LENGTH rows
    recent = df[feature_cols].values[-SEQUENCE_LENGTH:]
    recent_scaled = scaler.transform(recent)

    # Add batch dimension: (30, 6) → (1, 30, 6)
    X = torch.FloatTensor(recent_scaled).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probability = model(X).item()

    signal = "bullish" if probability > 0.5 else "bearish"
    confidence = probability if probability > 0.5 else 1 - probability

    return {
        "symbol": symbol,
        "model": "LSTM",
        "signal": signal,
        "confidence": round(confidence, 3),
        "bullish_prob": round(probability, 3),
        "bearish_prob": round(1 - probability, 3),
    }


if __name__ == "__main__":
    print("Training LSTM for AAPL...")
    train_lstm("AAPL", epochs=50)

    print("\nTraining LSTM for BTC-USD...")
    train_lstm("BTC-USD", epochs=50)

    print("\n--- LSTM Signals ---")
    print(predict_lstm("AAPL"))
    print(predict_lstm("BTC-USD"))