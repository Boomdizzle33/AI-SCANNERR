import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import time  # For tracking elapsed time and estimating time left
import openai
import ta  # pip install ta
import tensorflow as tf
from tensorflow.keras.models import load_model
from duckduckgo_search import DDGS  # For news retrieval
import concurrent.futures

# ----------------------------
# CONFIGURATION & API KEYS
# ----------------------------
# Set these keys in your Streamlit Cloud Secrets or via environment variables.
POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]  # Your Polygon.io API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]        # Your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Risk management parameters
RISK_PERCENTAGE = 0.02   # 2% risk per trade
RISK_REWARD_RATIO = 3    # 3:1 profit-to-risk ratio

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
# STEP 1: News Sentiment Functions
# ----------------------------
def fetch_stock_news(ticker, start_date, end_date):
    """
    Fetch news articles for a given ticker between start_date and end_date using DuckDuckGo.
    Dates must be timezone-aware datetime objects.
    """
    query = f"{ticker} news"
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=20)
    filtered_results = []
    for article in results:
        if "date" in article:
            try:
                article_date = datetime.datetime.strptime(article["date"], "%Y-%m-%d")
                article_date = article_date.replace(tzinfo=datetime.timezone.utc)
                if start_date <= article_date < end_date:
                    filtered_results.append(article)
            except Exception:
                filtered_results.append(article)
        else:
            filtered_results.append(article)
    return filtered_results

def analyze_sentiment(article_content):
    """
    Use OpenAI's ChatCompletion API to analyze the sentiment of a news article excerpt.
    Returns a one-word sentiment: "bullish", "bearish", or "neutral".
    
    NOTE: If you encounter issues with openai.ChatCompletion, either run `openai migrate`
    or pin your installation to openai==0.28.0.
    """
    prompt = (
        "Analyze the following news article excerpt and determine its overall sentiment. "
        "Consider the tone, language, and context. Respond with one word only: bullish, bearish, or neutral.\n\n"
        f"News article excerpt:\n{article_content}\n\n"
        "Answer:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Provide a one-word response: bullish, bearish, or neutral."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "neutral"

def compute_sentiment_score(ticker, start_date, end_date):
    """
    For a given ticker, fetch news and compute an average sentiment score.
    Mapping: bullish = +1, neutral = 0, bearish = -1.
    Returns a score between -1 and 1.
    """
    articles = fetch_stock_news(ticker, start_date, end_date)
    if not articles:
        return 0.0
    mapping = {"bullish": 1, "neutral": 0, "bearish": -1}
    scores = []
    for article in articles:
        content = article.get("title", "") + ". " + article.get("body", "")
        sentiment = analyze_sentiment(content)
        scores.append(mapping.get(sentiment, 0))
    return np.mean(scores) if scores else 0.0

def compute_news_sentiment_scores(stock_list):
    """
    Compute the news sentiment score for each ticker in stock_list concurrently.
    Returns a dictionary mapping each ticker to its news sentiment score.
    Also displays a progress bar with an estimated time remaining.
    """
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=2)
    sentiment_dict = {}
    progress_bar = st.progress(0)
    time_text = st.empty()
    total = len(stock_list)
    completed = 0
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(compute_sentiment_score, ticker, start_date, end_date): ticker
            for ticker in stock_list
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            score = future.result()
            sentiment_dict[ticker] = {"news_sentiment_score": score}
            completed += 1
            # Update progress bar
            progress_bar.progress(completed / total)
            # Calculate estimated time left
            elapsed = time.time() - start_time
            estimated_total = (elapsed / completed) * total
            time_left = max(0, estimated_total - elapsed)
            time_text.text(f"Estimated time left: {int(time_left)} seconds")
    return sentiment_dict

# ----------------------------
# STEP 2: Technical Analysis & VCP Features
# ----------------------------
def fetch_historical_data(ticker, days=365):
    """
    Fetch historical daily price and volume data for a given ticker from Polygon.io.
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

def compute_contraction_phases(df, window=20, smooth_window=3, threshold=0.05):
    """
    Compute the number of contraction phases in the last 'window' days.
    A contraction phase is defined when the smoothed trading range (high - low) drops by at least 'threshold'.
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
    Compute technical indicators and VCP features including volume contraction and contraction phases.
    Returns a dictionary with the latest computed values.
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
# STEP 3: ML Model Loading & Breakout Probability Calculation
# ----------------------------
def load_pretrained_lstm_model():
    """
    Load the pre-trained LSTM model from disk.
    The model should be saved as 'lstm_vcp_model.keras' in your repository.
    """
    try:
        model = load_model("lstm_vcp_model.keras")
        st.write("Pre-trained LSTM model loaded.")
        return model
    except Exception as e:
        st.error("Pre-trained LSTM model not found. Please ensure 'lstm_vcp_model.keras' is in the repository.")
        class DummyModel:
            def predict(self, X):
                return np.array([[0.5]])
        return DummyModel()

def compute_lstm_breakout_probability(ticker, news_sentiment_score, sequence_length=20, breakout_threshold=0.01):
    """
    Compute the breakout probability using the pre-trained LSTM model.
    The function prepares the most recent sequence of technical indicators, feeds it to the model,
    and adjusts the output using the news sentiment score, overall market trend, and sector strength.
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
    
    sentiment_adjustment = 1 + 0.3 * news_sentiment_score
    final_prob = ml_prob * sentiment_adjustment
    
    market_multiplier = 1.1 if get_market_trend() else 0.9
    sector_multiplier = get_sector_strength(ticker)
    final_prob = final_prob * market_multiplier * sector_multiplier
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
# STEP 4: Trade Setup & Report Generation
# ----------------------------
def calculate_trade_levels(entry_price, stop_loss_price):
    """
    Compute the profit target using a 3:1 risk-to-reward ratio.
    """
    risk = abs(entry_price - stop_loss_price)
    return entry_price + RISK_REWARD_RATIO * risk

def generate_breakout_report(news_sentiment_dict):
    """
    For each ticker, generate a report with:
      - Trade parameters: entry, stop-loss, and profit target.
      - Breakout probability (displayed as an integer percent).
      - News sentiment score (average score from recent news).
      - VCP Setup confirmation ("Yes" if conditions are met, otherwise "No").
      - Contraction phases, capital flow strength, and volume trend.
    
    The VCP Setup is confirmed if:
      - Breakout probability > 50%
      - Volume trend is negative (indicating contraction)
      - Contraction phases >= 3
    """
    report_rows = []
    for ticker, sentiment_info in news_sentiment_dict.items():
        tech_data = perform_technical_analysis(ticker)
        if tech_data is None:
            continue
        news_sentiment_score = sentiment_info["news_sentiment_score"]
        ml_breakout_prob = compute_lstm_breakout_probability(ticker, news_sentiment_score)
        entry_price = tech_data["current_price"]
        stop_loss = tech_data["Support"] if tech_data["Support"] > 0 else entry_price * 0.98
        profit_target = calculate_trade_levels(entry_price, stop_loss)
        vcp_setup = "Yes" if (ml_breakout_prob > 0.5 and tech_data["Volume_Trend"] < 0 and tech_data.get("Contraction_Phases", 0) >= 3) else "No"
        report_rows.append({
            "Stock": ticker,
            "Updated Entry": round(entry_price, 2),
            "Stop-Loss": round(stop_loss, 2),
            "Profit Target": round(profit_target, 2),
            "Breakout Probability": f"{int(round(ml_breakout_prob * 100))}%",
            "News Sentiment Score": round(news_sentiment_score, 2),
            "VCP Setup": vcp_setup,
            "Contraction Phases": tech_data.get("Contraction_Phases", 0),
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
    This system combines technical analysis with a pre-trained LSTM model and a news sentiment score to detect Quantitative VCP setups.
    You can import a list of stock tickers via a CSV file or manually enter them.
    The system computes technical indicators (including volume contraction and contraction phases) and outputs a breakout probability.
    Trade parameters are suggested based on a 3:1 risk-to-reward ratio, and the output confirms whether the stock qualifies as a VCP setup.
    """)
    
    # Option 1: CSV file uploader (expects a column named "ticker")
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
        # Option 2: Manual entry
        stock_list_input = st.text_area("Or enter comma-separated stock tickers", "AAPL, MSFT, GOOGL, AMZN, TSLA")
        stock_list = [ticker.strip().upper() for ticker in stock_list_input.split(",") if ticker.strip() != ""]
    
    if st.button("Scan for Breakout Candidates"):
        with st.spinner("Computing news sentiment scores..."):
            news_sentiment_dict = compute_news_sentiment_scores(stock_list)
        st.success("News sentiment scores computed.")
        
        with st.spinner("Running technical analysis and ML breakout prediction..."):
            report_df = generate_breakout_report(news_sentiment_dict)
        
        if report_df.empty:
            st.warning("No breakout candidates could be generated from the data.")
        else:
            st.markdown("### 📌 Top 5 Stocks Likely to Break Out (Quantitative VCP)")
            st.table(report_df)
            
            best_candidate = report_df.iloc[0]
            weakest_candidate = report_df.iloc[-1] if len(report_df) > 1 else best_candidate
                
            st.markdown("#### 📢 Key Takeaways")
            st.markdown(
                f"- 🚀 **Best Breakout Candidate: {best_candidate['Stock']}**. "
                f"Breakout Probability: {best_candidate['Breakout Probability']}, "
                f"News Sentiment Score: {best_candidate['News Sentiment Score']}, "
                f"Volume Trend: {best_candidate['Volume Trend']}, "
                f"Contraction Phases: {best_candidate['Contraction Phases']}, "
                f"VCP Setup: {best_candidate['VCP Setup']}."
            )
            st.markdown(
                f"- 📉 **Stock Showing Relative Weakness: {weakest_candidate['Stock']}**. "
                f"Breakout Probability: {weakest_candidate['Breakout Probability']}, "
                f"News Sentiment Score: {weakest_candidate['News Sentiment Score']}, "
                f"VCP Setup: {weakest_candidate['VCP Setup']} "
                f"(Contraction Phases: {weakest_candidate['Contraction Phases']})."
            )
            
if __name__ == "__main__":
    main()


