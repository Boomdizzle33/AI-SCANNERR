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

# 🔹 AI Predicts Breakout Probability
def ai_predict_breakout(symbol):
    """Uses AI to analyze technical indicators & predict breakout probability."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2024-02-01?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        prices = [candle["c"] for candle in response["results"]]

        ma_20 = sum(prices[-20:]) / 20
        ma_50 = sum(prices[-50:]) / 50
        rsi = sum([prices[i] - prices[i - 1] for i in range(1, len(prices)) if prices[i] > prices[i - 1]]) / 14 * 100
        macd = ma_20 - ma_50
        atr = max(prices[-10:]) - min(prices[-10:])
        
        ai_input = f"""
        Predict the probability of a breakout for {symbol} based on:
        - 20-day MA: {ma_20}
        - 50-day MA: {ma_50}
        - RSI: {rsi}
        - MACD: {macd}
        - ATR: {atr}
        Provide a probability (0-100). Respond with only a single number.
        """

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": ai_input}], "max_tokens": 10}

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return max(0, min(100, float(response.json()["choices"][0]["message"]["content"].strip())))
    except:
        return 50  # Default to neutral probability

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




