import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import openai
import ta  # Install with: pip install ta
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIGURATION & API KEYS
# ----------------------------
# Store these keys securely. In Streamlit Cloud, set these in your Secrets.
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]  # Your Polygon.io API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]        # Your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Risk management parameters
RISK_PERCENTAGE = 0.02   # 2% risk per trade
RISK_REWARD_RATIO = 3    # 3:1 profit to risk

# ----------------------------
# SECTOR MAPPING (for demonstration)
# ----------------------------
sector_map = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLK",
    "AMZN": "XLY",
    "TSLA": "XLY"
}

# ----------------------------
# STEP 1: Rule-Based Filtering via News Sentiment
# ----------------------------
def fetch_stock_news(ticker, date):
    """
    Fetch news for a given stock ticker for today's trading session from Polygon.io.
    """
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": date.strftime("%Y-%m-%dT00:00:00Z"),
        "published_utc.lt": (date + datetime.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
        "apiKey": POLYGON_API_KEY,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error(f"Error fetching news for {ticker}")
        return []

def analyze_sentiment(article_content):
    """
    Use OpenAI API to analyze the sentiment of a news article excerpt.
    The prompt instructs the model to analyze tone, language, and context,
    and then respond with one word only: bullish, bearish, or neutral.
    """
    prompt = (
        "Analyze the following news article excerpt and determine its overall sentiment. "
        "Consider the tone, language, and context of the article. "
        "Based on your analysis, respond with one word only: bullish, bearish, or neutral.\n\n"
        f"News article excerpt:\n{article_content}\n\n"
        "Answer:"
    )
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=5,
            temperature=0.3,
        )
        return response.choices[0].text.strip().lower()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "neutral"

def scan_for_bullish_stocks(stock_list):
    """
    For each stock in the list, check if it has at least 5 bullish news articles today.
    Returns a dictionary with tickers and a count of bullish articles.
    """
    today = datetime.datetime.utcnow()
    bullish_candidates = {}
    for ticker in stock_list:
        articles = fetch_stock_news(ticker, today)
        bullish_count = 0
        for article in articles:
            content = article.get("title", "") + ". " + article.get("description", "")
            if "bullish" in analyze_sentiment(content):
                bullish_count += 1
        if bullish_count >= 5:
            bullish_candidates[ticker] = {"bullish_count": bullish_count}
            st.write(f"{ticker} has {bullish_count} bullish articles.")
    return bullish_candidates

# ----------------------------
# STEP 2: Feature Engineering for VCP (Technical Analysis & Volume Contraction)
# ----------------------------
def fetch_historical_data(ticker, days=180):
    """
    Fetch historical daily price data for the given ticker using Polygon.io.
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
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

def perform_technical_analysis(ticker):
    """
    Compute technical indicators and volume contraction using historical data.
    Returns a snapshot (latest row) of key technical values.
    """
    df = fetch_historical_data(ticker)
    if df.empty:
        return None
    # Compute technical indicators
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
    
    latest = df.iloc[-1]
    tech_data = {
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
        "Volume_Trend": latest["Volume_Trend"]
    }
    return tech_data

# ----------------------------
# STEP 3: Refinement with Machine Learning (Pre-Trained LSTM Model)
# ----------------------------
def load_pretrained_lstm_model():
    """
    Load a pre-trained LSTM model from disk.
    The model should be trained offline on historical data to learn VCP patterns.
    """
    try:
        model = load_model("lstm_vcp_model.h5")
        st.write("Pre-trained LSTM model loaded.")
        return model
    except Exception as e:
        st.warning("Pre-trained LSTM model not found. Using a dummy model for demonstration.")
        # Dummy model always returns a 0.5 probability
        class DummyModel:
            def predict(self, X):
                return np.array([[0.5]])
        return DummyModel()

def compute_lstm_breakout_probability(ticker, bullish_count, sequence_length=20, breakout_threshold=0.01):
    """
    Compute the breakout probability for the stock using the pre-trained LSTM model.
    The model refines the candidate by capturing nuanced patterns from historical data.
    """
    df = fetch_historical_data(ticker)
    if df.empty:
        return 0
    # Calculate technical indicators and volume contraction
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

    # Define the feature set (VCP-related indicators)
    feature_cols = ['SMA20', 'SMA50', 'RSI', 'MACD',
                    'Bollinger_High', 'Bollinger_Low',
                    'Support', 'Resistance', 'Volume_Spike', 'Volume_Trend']
    df = df.dropna(subset=feature_cols)
    if len(df) < sequence_length:
        return 0

    # Use the last sequence_length days as input for the LSTM model
    X_latest = df[feature_cols].values[-sequence_length:]
    X_latest = X_latest.reshape(1, sequence_length, len(feature_cols))
    
    lstm_model = load_pretrained_lstm_model()
    ml_prob = lstm_model.predict(X_latest)[0][0]
    
    # Adjust probability using bullish sentiment as a refinement signal
    sentiment_score = min(1.0, bullish_count / 10.0)
    final_prob = ml_prob * (1 + 0.3 * sentiment_score)
    
    # Further adjust based on overall market and sector strength
    market_multiplier = 1.1 if get_market_trend() else 0.9
    sector_multiplier = get_sector_strength(ticker)
    final_prob = final_prob * market_multiplier * sector_multiplier
    return min(final_prob, 1.0)

def get_market_trend():
    """
    Assess overall market trend using SPY as a proxy.
    Returns True if bullish (current price > 50-day SMA), otherwise False.
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
    Returns a multiplier (1.1 for strong sector, 0.9 for weak).
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
# STEP 4: Trade Setup & Report Generation
# ----------------------------
def calculate_trade_levels(entry_price, stop_loss_price):
    """
    Compute the profit target using a 3:1 profit-to-risk ratio.
    """
    risk = abs(entry_price - stop_loss_price)
    return entry_price + RISK_REWARD_RATIO * risk

def generate_breakout_report(bullish_candidates):
    """
    For each candidate stock, apply technical analysis and ML refinement
    to generate a report with trade parameters.
    """
    report_rows = []
    for ticker, info in bullish_candidates.items():
        tech_data = perform_technical_analysis(ticker)
        if tech_data is None:
            continue
        ml_breakout_prob = compute_lstm_breakout_probability(ticker, info["bullish_count"])
        entry_price = tech_data["current_price"]
        # Use support as stop-loss if available; else assume a 2% drop.
        stop_loss = tech_data["Support"] if tech_data["Support"] > 0 else entry_price * 0.98
        profit_target = calculate_trade_levels(entry_price, stop_loss)
        report_rows.append({
            "Stock": ticker,
            "Updated Entry": round(entry_price, 2),
            "Stop-Loss": round(stop_loss, 2),
            "Profit Target": round(profit_target, 2),
            "Breakout Probability": f"{round(ml_breakout_prob*100, 1)}%",
            "Capital Flow Strength": "✅" if tech_data["volume_spike"] else "❌",
            "Volume Trend": f"{round(tech_data['Volume_Trend'], 1)}%"
        })
    df_report = pd.DataFrame(report_rows)
    if not df_report.empty:
        df_report["Breakout_Prob_Value"] = df_report["Breakout Probability"].str.rstrip("%").astype(float)
        df_report = df_report.sort_values(by="Breakout_Prob_Value", ascending=False).head(5)
        df_report.drop("Breakout_Prob_Value", axis=1, inplace=True)
    return df_report

# ----------------------------
# STREAMLIT APP LAYOUT
# ----------------------------
def main():
    st.title("Quantitative VCP Trading System with ML Refinement")
    st.markdown("""
    This system combines rule-based filtering with machine learning refinement to detect Quantitative VCP setups.
    You can import a list of filtered stock tickers from TradingView via a CSV file or manually enter them.
    The system then computes technical indicators, applies a pre-trained LSTM model to output a breakout probability,
    and suggests trade parameters based on a 3:1 risk-to-reward ratio.
    """)
    
    # Option 1: CSV file uploader (expecting a column named "ticker")
    uploaded_file = st.file_uploader("Upload CSV file with filtered stock tickers from TradingView", type=["csv"])
    if uploaded_file is not None:
        try:
            df_tickers = pd.read_csv(uploaded_file)
            stock_list = df_tickers['ticker'].dropna().astype(str).tolist()
            st.write("Tickers imported:", stock_list)
        except Exception as e:
            st.error("Error reading CSV file. Ensure it has a column named 'ticker'.")
            stock_list = []
    else:
        # Option 2: Manual entry
        stock_list_input = st.text_area("Or enter comma-separated stock tickers", "AAPL, MSFT, GOOGL, AMZN, TSLA")
        stock_list = [ticker.strip().upper() for ticker in stock_list_input.split(",") if ticker.strip() != ""]
    
    if st.button("Scan for Breakout Candidates"):
        with st.spinner("Scanning for bullish news sentiment..."):
            bullish_candidates = scan_for_bullish_stocks(stock_list)
        if not bullish_candidates:
            st.warning("No stocks met the bullish news criteria.")
            return
        
        st.success("Candidates identified. Running technical analysis and ML breakout prediction...")
        report_df = generate_breakout_report(bullish_candidates)
        
        if report_df.empty:
            st.warning("No breakout candidates could be generated from the data.")
        else:
            st.markdown("### 📌 Top 5 Stocks Likely to Break Out (Quantitative VCP)")
            st.table(report_df)
            
            best_candidate = report_df.iloc[0]
            weakest_candidate = report_df.iloc[-1] if len(report_df) > 1 else best_candidate
                
            st.markdown("#### 📢 Key Takeaways")
            st.markdown(f"- 🚀 **Best Breakout Candidate: {best_candidate['Stock']}**. Reason: Highest breakout probability at {best_candidate['Breakout Probability']} with strong capital flow and favorable volume contraction (Volume Trend: {best_candidate['Volume Trend']}).")
            st.markdown(f"- 📉 **Stock Showing Relative Weakness: {weakest_candidate['Stock']}**. Reason: Lower breakout probability relative to peers.")
            
if __name__ == "__main__":
    main()


