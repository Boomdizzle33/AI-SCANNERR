import streamlit as st
import requests
import pandas as pd
import time
from duckduckgo_search import DDGS
import openai

# 🔹 Load API Keys Securely from Streamlit Secrets
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

openai.api_key = OPENAI_API_KEY  # Set OpenAI Key

# 🔹 UI Components for Live Updates
st.title("📊 AI Stock Scanner (Fully AI-Powered 🚀)")

# 🔹 Function to Fetch News Sentiment
def get_news_sentiment(symbol):
    """Fetches news sentiment for a stock using OpenAI and DuckDuckGo Search."""
    try:
        ddgs = DDGS()
        headlines = [result["title"] for result in ddgs.news(symbol, max_results=5)]
        
        if not headlines:
            return 0.0  

        news_text = "\n".join(headlines)
        prompt = f"""
        Analyze the sentiment of these stock news headlines for {symbol}.
        Assign a sentiment score between -1.0 (very negative) and 1.0 (very positive).
        Respond with only a single number.
        Headlines:
        {news_text}
        """

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 10}

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return max(-1.0, min(1.0, float(response.json()["choices"][0]["message"]["content"].strip())))  
    except:
        return 0.0  

# 🔹 Function to Check for Upcoming Earnings
def check_earnings_date(symbol):
    """Checks if the stock has earnings in the next 7 days using Polygon.io."""
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}/earnings?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        
        earnings_list = response.get("results", [])
        if earnings_list:
            upcoming_earnings = earnings_list[0]["reportDate"]
            days_until_earnings = (pd.to_datetime(upcoming_earnings) - pd.to_datetime("today")).days
            
            if days_until_earnings <= 7:
                return f"⚠️ Earnings in {days_until_earnings} days ({upcoming_earnings})"
            else:
                return "✅ No earnings risk"
        return "✅ No earnings risk"
    except:
        return "❓ Earnings data unavailable"

# 🔹 Function to Get Support & Resistance Levels
def get_support_resistance(symbol):
    """Fetches recent support & resistance levels using price data."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2024-02-01?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        prices = [candle["c"] for candle in response["results"]]

        support = min(prices[-10:])  
        resistance = max(prices[-10:])  

        return round(support, 2), round(resistance, 2)
    except:
        return None, None

# 🔹 Function to Calculate ATR-Based Stop-Loss
def calculate_atr_stop(symbol):
    """Calculates an ATR-based stop-loss instead of a fixed percentage."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2024-02-01?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        prices = [candle["c"] for candle in response["results"]]

        tr = [max(prices[i] - prices[i - 1], abs(prices[i] - prices[i - 2])) for i in range(2, len(prices))]
        atr = sum(tr[-14:]) / 14 if len(tr) > 0 else 1  

        return atr * 1.5  # Use 1.5x ATR for stop-loss
    except:
        return None  

# 🔹 Function to Calculate Entry, Stop-Loss & Profit Target
def calculate_trade_levels(symbol):
    """Determines best entry, stop-loss, and profit target based on AI prediction."""
    support, resistance = get_support_resistance(symbol)
    atr_stop = calculate_atr_stop(symbol)

    if support is None or resistance is None or atr_stop is None:
        return None, None, None  

    entry_price = resistance  # Enter at resistance level if AI predicts breakout
    stop_loss = entry_price - atr_stop  # ATR-based stop-loss
    profit_target = entry_price + (entry_price - stop_loss) * 3  # 3:1 reward ratio

    return round(entry_price, 2), round(stop_loss, 2), round(profit_target, 2)

# 🔹 AI Predicts Breakout Probability
def ai_predict_breakout(symbol):
    """Uses AI to analyze technical indicators & predict breakout probability."""
    try:
        return 80  # Placeholder for now
    except:
        return 50  

# 🔹 Process Stocks & Add Trade Levels
def process_stocks(tickers):
    """Processes stocks, filters weak setups, and checks for upcoming earnings reports."""
    results = []

    for ticker in tickers:
        sentiment_score = get_news_sentiment(ticker)
        breakout_probability = ai_predict_breakout(ticker)
        earnings_warning = check_earnings_date(ticker)
        entry_price, stop_loss, profit_target = calculate_trade_levels(ticker)

        ai_score = round((sentiment_score * 20) + (breakout_probability * 0.8), 2)  # FIXED SCALING

        trade_approved = "✅ Yes" if ai_score >= 75 and breakout_probability >= 80 else "❌ No"

        results.append([
            ticker, sentiment_score, breakout_probability, 
            ai_score, trade_approved, earnings_warning, entry_price, stop_loss, profit_target
        ])

    return results

# 🔹 File Upload & Scanner Execution
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()

    if st.button("Run AI Scanner"):
        st.write("🔍 **Scanning Stocks... Please Wait...**")
        final_results = process_stocks(tickers)

        st.success("✅ AI Analysis Completed!")
        st.dataframe(pd.DataFrame(final_results, columns=[
            "Stock", "Sentiment Score", "Breakout Probability",  
            "AI Score", "Trade Approved", "Earnings Alert", "Entry Price", "Stop-Loss", "Profit Target"
        ]))

