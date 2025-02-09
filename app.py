import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import time  # For progress estimation and delays
import ta  # pip install ta
import tensorflow as tf
from tensorflow.keras.models import load_model
import concurrent.futures

# ----------------------------
# CONFIGURATION & API KEYS
# ----------------------------
# Set these keys in your Streamlit Cloud Secrets or as environment variables.
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]  # Your Polygon.io API key

# Risk management parameters
RISK_PERCENTAGE = 0.02   # 2% risk per trade
RISK_REWARD_RATIO = 3    # 3:1 risk-to-reward ratio

# ----------------------------
# SECTOR MAPPING (Example)
# ----------------------------
sector_map = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLK",
    "AMZN": "XLY",
    "TSLA": "XLY"
}

# ----------------------------
# STEP 1: Technical Analysis & VCP Feature Engineering
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_historical_data(ticker, days=365):
    """
    Fetch historical daily price and volume data for a given ticker from Polygon.io.
    Cached for 1 hour.
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        df = pd.DataFrame(response.json().get("results", []))
        if not df.empty:
            df["t"] = pd.to_datetime(df["t"], unit='ms')
            df.set_index("t", inplace=True)
        return df
    else:
        st.error(f"Error fetching historical data for {ticker}")
        return pd.DataFrame()

def compute_volume_trend(df, window=20):
    """
    Compute the percentage change of volume relative to a longer-term average.
    A negative value indicates volume contraction (a key VCP signal).
    """
    df["Volume_LongAvg"] = df["v"].rolling(window=window*2).mean()
    df["Volume_Trend"] = ((df["v"] - df["Volume_LongAvg"]) / df["Volume_LongAvg"]) * 100
    return df

def compute_contraction_phases(df, window=20, smooth_window=3, threshold=0.05):
    """
    Compute the number of contraction phases in the last 'window' days.
    A contraction phase is defined when the smoothed trading range (high - low) drops by at least 'threshold'.
    (This feature is computed and displayed for informational purposes.)
    """
    if len(df) < window or ("h" not in df.columns or "l" not in df.columns):
        return 0
    recent_df = df.tail(window).copy()
    recent_df["range"] = recent_df["h"] - recent_df["l"]
    recent_df["smoothed_range"] = recent_df["range"].rolling(window=smooth_window, min_periods=1).mean()
    baseline = recent_df["smoothed_range"].iloc[0]
    phase_count = 0
    for val in recent_df["smoothed_range"].iloc[1:]:
        if baseline > 0 and ((baseline - val) / baseline) >= threshold:
            phase_count += 1
            baseline = val
    return phase_count

def perform_technical_analysis(ticker):
    """
    Compute technical indicators and VCP features (including volume contraction).
    Returns a dictionary of the latest computed values.
    """
    df = fetch_historical_data(ticker)
    if df.empty:
        return None
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
    df = compute_volume_trend(df)
    
    contraction_phases = compute_contraction_phases(df, window=20, smooth_window=3, threshold=0.05)
    
    latest = df.iloc[-1]
    return {
        "current_price": latest["c"],
        "SMA20": latest["SMA20"],
        "SMA50": latest["SMA50"],
        "RSI": latest["RSI"],
        "MACD": latest["MACD"],
        "Bollinger_High": latest["Bollinger_High"],
        "Bollinger_Low": latest["Bollinger_Low"],
        "Support": latest["Support"],
        "Resistance": latest["Resistance"],
        "volume_spike": bool(latest["Volume_Spike"]),
        "Volume_Trend": latest["Volume_Trend"],
        "Contraction_Phases": contraction_phases
    }

# ----------------------------
# STEP 2: ML Model Loading & Breakout Probability Calculation
# ----------------------------
@st.cache_resource
def load_pretrained_lstm_model():
    """
    Load the pre-trained LSTM model from disk.
    The model should be saved as 'lstm_vcp_model.keras' in your repository.
    (No message is displayed on successful load.)
    """
    try:
        model = load_model("lstm_vcp_model.keras")
        return model
    except Exception as e:
        st.error("Pre-trained LSTM model not found. Please ensure 'lstm_vcp_model.keras' is in the repository.")
        class DummyModel:
            def predict(self, X):
                return np.array([[0.5]])
        return DummyModel()

def compute_lstm_breakout_probability(ticker, sequence_length=20, breakout_threshold=0.01):
    """
    Compute the breakout probability using the pre-trained LSTM model.
    Prepares a sequence of technical indicators, feeds it to the model, and adjusts the output using
    overall market trend and sector strength.
    """
    df = fetch_historical_data(ticker)
    if df.empty:
        return 0
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
    df = compute_volume_trend(df)
    df = df.dropna()
    if df.empty or len(df) < sequence_length:
        return 0

    feature_cols = [
        'SMA20', 'SMA50', 'RSI', 'MACD',
        'Bollinger_High', 'Bollinger_Low',
        'Support', 'Resistance', 'Volume_Spike', 'Volume_Trend'
    ]
    df = df.dropna(subset=feature_cols)
    if len(df) < sequence_length:
        return 0

    X_latest = df[feature_cols].values[-sequence_length:]
    X_latest = X_latest.reshape(1, sequence_length, len(feature_cols))
    
    lstm_model = load_pretrained_lstm_model()
    ml_prob = lstm_model.predict(X_latest)[0][0]
    
    market_multiplier = 1.1 if get_market_trend() else 0.9
    sector_multiplier = get_sector_strength(ticker)
    final_prob = ml_prob * market_multiplier * sector_multiplier
    return min(final_prob, 1.0)

def get_market_trend():
    """
    Assess the overall market trend using SPY as a proxy.
    Returns True if SPY's current price is above its 50-day SMA, otherwise False.
    """
    df = fetch_historical_data("SPY", days=100)
    if df.empty:
        return True
    df["SMA50"] = ta.trend.sma_indicator(df["c"], window=50)
    latest = df.iloc[-1]
    return latest["c"] > latest["SMA50"]

def get_sector_strength(ticker):
    """
    Evaluate the strength of the stock's sector using its representative ETF.
    Returns 1.1 if the ETF's current price is above its 50-day SMA, else 0.9.
    """
    if ticker not in sector_map:
        return 1.0
    sector_etf = sector_map[ticker]
    df = fetch_historical_data(sector_etf, days=100)
    if df.empty:
        return 1.0
    df["SMA50"] = ta.trend.sma_indicator(df["c"], window=50)
    latest = df.iloc[-1]
    return 1.1 if latest["c"] > latest["SMA50"] else 0.9

# ----------------------------
# STEP 3: Trade Setup & Report Generation
# ----------------------------
def calculate_trade_levels(entry_price, stop_loss_price):
    """
    Compute the profit target based on a 3:1 risk-to-reward ratio.
    """
    risk = abs(entry_price - stop_loss_price)
    return entry_price + RISK_REWARD_RATIO * risk

def generate_breakout_report():
    """
    For each ticker provided, generate a report that includes:
      - Trade parameters: Updated Entry, Stop-Loss, and Profit Target.
      - Breakout probability (displayed as an integer percent).
      - VCP Setup confirmation ("Yes" if breakout probability > 40% and volume trend is negative; otherwise "No").
      - Contraction phases (informational), Capital Flow Strength, and Volume Trend.
    Returns a DataFrame listing all stocks that qualify as VCP setups.
    """
    report_rows = []
    for ticker in stock_list:
        tech_data = perform_technical_analysis(ticker)
        if tech_data is None:
            continue
        ml_breakout_prob = compute_lstm_breakout_probability(ticker)
        entry_price = tech_data["current_price"]
        stop_loss = tech_data["Support"] if tech_data["Support"] > 0 else entry_price * 0.98
        profit_target = calculate_trade_levels(entry_price, stop_loss)
        # A stock qualifies as a VCP setup if breakout probability > 40% and volume trend is negative.
        vcp_setup = "Yes" if (ml_breakout_prob > 0.4 and tech_data["Volume_Trend"] < 0) else "No"
        if vcp_setup == "Yes":
            report_rows.append({
                "Stock": ticker,
                "Updated Entry": round(entry_price, 2),
                "Stop-Loss": round(stop_loss, 2),
                "Profit Target": round(profit_target, 2),
                "Breakout Probability": f"{int(round(ml_breakout_prob * 100))}%",
                "Contraction Phases": tech_data.get("Contraction_Phases", 0),
                "Capital Flow Strength": "✅" if tech_data["volume_spike"] else "❌",
                "Volume Trend": f"{round(tech_data['Volume_Trend'], 1)}%"
            })
    df_report = pd.DataFrame(report_rows)
    if not df_report.empty:
        df_report["Breakout_Prob_Value"] = df_report["Breakout Probability"].str.rstrip("%").astype(float)
        df_report = df_report.sort_values(by="Breakout_Prob_Value", ascending=False)
        df_report.drop("Breakout_Prob_Value", axis=1, inplace=True)
    return df_report

# ----------------------------
# STREAMLIT APP LAYOUT
# ----------------------------
def main():
    st.title("Quantitative VCP Trading System with ML Refinement")
    st.markdown("""
    This system uses technical analysis and a pre-trained LSTM model to detect Quantitative VCP setups.
    You can import a list of stock tickers via a CSV file or manually enter them.
    The system computes technical indicators (including volume contraction) and outputs a breakout probability 
    along with suggested trade parameters.
    Only stocks that qualify as VCP setups are shown.
    """)
    
    global stock_list  # Declare stock_list as global so that it can be used in generate_breakout_report()
    
    uploaded_file = st.file_uploader("Upload CSV file with stock tickers", type=["csv"])
    if uploaded_file is not None:
        try:
            df_tickers = pd.read_csv(uploaded_file)
            stock_list = df_tickers['ticker'].dropna().astype(str).tolist()
            st.write("Tickers imported:", stock_list)
        except Exception as e:
            st.error("Error reading CSV file. Ensure it has a column named 'ticker'.")
            stock_list = []
    else:
        stock_list_input = st.text_area("Or enter comma-separated stock tickers", "AAPL, MSFT, GOOGL, AMZN, TSLA")
        stock_list = [ticker.strip().upper() for ticker in stock_list_input.split(",") if ticker.strip() != ""]
    
    if st.button("Scan for Breakout Candidates"):
        with st.spinner("Running technical analysis and ML breakout prediction..."):
            report_df = generate_breakout_report()
        
        if report_df.empty:
            st.warning("No stocks qualified as VCP setups based on the criteria.")
        else:
            st.markdown("### 📌 Stocks in VCP Setup Pattern")
            st.table(report_df)
            
            st.markdown("#### 📢 Summary")
            st.markdown("The table above lists all stocks that qualify as VCP setups with suggested trade parameters:")
            st.markdown("- **Updated Entry:** Suggested entry price based on current price.")
            st.markdown("- **Stop-Loss:** Recommended stop-loss level (below support).")
            st.markdown("- **Profit Target:** Target price calculated with a 3:1 risk-to-reward ratio.")
            st.markdown("- **Breakout Probability:** The model's breakout probability (as an integer percent).")
            st.markdown("- **Contraction Phases:** Number of contraction phases detected (informational only).")
            st.markdown("- **Capital Flow Strength & Volume Trend:** Additional indicators to support the setup.")
            
if __name__ == "__main__":
    main()

