import os
import requests
import pandas as pd
import numpy as np
import datetime
import ta  # pip install ta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set your Polygon.io API key here or via an environment variable.
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or "K0alSpz0uQvenAazXbEjkFQGthxL0eCq"


def get_sp500_tickers():
    """
    Fetch the list of S&P 500 ticker symbols from Wikipedia.
    Returns a list of tickers (with dots replaced by hyphens if necessary).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print("Error fetching S&P 500 tickers:", e)
        return []


def fetch_historical_data(ticker, days=365):
    """
    Fetch historical daily price and volume data for the given ticker from Polygon.io.
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": POLYGON_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json().get("results", []))
        if not df.empty:
            df["t"] = pd.to_datetime(df["t"], unit='ms')
            df.set_index("t", inplace=True)
        return df
    else:
        print(f"Error fetching historical data for {ticker}")
        return pd.DataFrame()


def compute_features(df):
    """
    Compute technical indicators that capture VCP characteristics.
    """
    df["SMA20"] = ta.trend.sma_indicator(df["c"], window=20)
    df["SMA50"] = ta.trend.sma_indicator(df["c"], window=50)
    df["RSI"] = ta.momentum.rsi(df["c"], window=14)
    df["MACD"] = ta.trend.macd(df["c"])
    bollinger = ta.volatility.BollingerBands(df["c"], window=20, window_dev=2)
    df["Bollinger_High"] = bollinger.bollinger_hband()
    df["Bollinger_Low"] = bollinger.bollinger_lband()
    df["Support"] = df["c"].rolling(window=20).min()
    df["Resistance"] = df["c"].rolling(window=20).max()
    df["Volume_Avg"] = df["v"].rolling(window=20).mean()
    df["Volume_Spike"] = (df["v"] > (df["Volume_Avg"] * 1.5)).astype(int)
    df["Volume_LongAvg"] = df["v"].rolling(window=40).mean()
    df["Volume_Trend"] = ((df["v"] - df["Volume_LongAvg"]) / df["Volume_LongAvg"]) * 100
    return df.dropna()


def label_data(df, breakout_threshold=0.01):
    """
    Label each day as a breakout if the next day's close exceeds current resistance by (1 + breakout_threshold).
    """
    df = df.copy()
    df["Next_Close"] = df["c"].shift(-1)
    df["Breakout"] = (df["Next_Close"] > df["Resistance"] * (1 + breakout_threshold)).astype(int)
    return df.dropna()


def prepare_sequences(df, feature_cols, sequence_length=20):
    """
    Create sequences for LSTM training.
    """
    data = df[feature_cols].values
    labels = df["Breakout"].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)


def get_training_data(tickers, days=365, sequence_length=20, breakout_threshold=0.01):
    """
    Prepare training data from historical data of given tickers.
    """
    X_list, y_list = [], []
    feature_cols = [
        'SMA20', 'SMA50', 'RSI', 'MACD',
        'Bollinger_High', 'Bollinger_Low',
        'Support', 'Resistance', 'Volume_Spike', 'Volume_Trend'
    ]
    for ticker in tickers:
        print(f"Processing {ticker}...")
        df = fetch_historical_data(ticker, days)
        if df.empty:
            continue
        df = compute_features(df)
        df = label_data(df, breakout_threshold)
        X, y = prepare_sequences(df, feature_cols, sequence_length)
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
    if X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all, feature_cols
    else:
        return None, None, feature_cols


def train_sp500_model(tickers=None, days=365, sequence_length=20, epochs=50, batch_size=32):
    """
    Train an LSTM model on S&P 500 data.
    Recommended parameters:
      - days: 365
      - sequence_length: 20
      - epochs: 50
      - batch_size: 32
    Saves the trained model as 'lstm_vcp_model.keras'.
    """
    if tickers is None:
        tickers = get_sp500_tickers()
    print("Tickers fetched:", tickers)

    X, y, feature_cols = get_training_data(tickers, days, sequence_length)
    if X is None or y is None:
        print("No training data available.")
        return None
    print("Training data shape:", X.shape, y.shape)

    model = Sequential()
    model.add(Input(shape=(sequence_length, len(feature_cols))))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop])

    # Save in the new native Keras format
    model.save("lstm_vcp_model.keras")
    print("Model trained and saved as lstm_vcp_model.keras")
    return model, history


if __name__ == "__main__":
    train_sp500_model()
