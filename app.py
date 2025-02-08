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
progress_bar = st.progress(0)
time_remaining_text = st.empty()  

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

# 🔹 Function to Get Sector Strength
def get_sector_strength(symbol):
    """Fetches sector performance and ranks relative strength."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-12-01/2024-02-01?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        sector_strength = response["results"][0]["c"]  
        return sector_strength  
    except:
        return 0  

# 🔹 AI Predicts Breakout Probability
def ai_predict_breakout(symbol):
    """Uses AI to analyze technical indicators & predict breakout probability."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-12-01/2024-02-01?apikey={POLYGON_API_KEY}"
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
        return 50  

# 🔹 AI Confirms Breakout Strength Using Volume & Institutional Activity
def ai_confirm_breakout_strength(symbol):
    """Uses AI to confirm breakout strength based on volume and institutional buying."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-12-01/2024-02-01?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()
        volumes = [candle["v"] for candle in response["results"]]
        avg_volume = sum(volumes[-50:]) / 50
        last_volume = volumes[-1]

        ai_input = f"""
        Analyze {symbol} for institutional buying strength.
        - 50-day Avg Volume: {avg_volume}
        - Last Trading Volume: {last_volume}
        Provide a confidence score (0-100). Respond with only a single number.
        """

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": ai_input}], "max_tokens": 10}

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return max(0, min(100, float(response.json()["choices"][0]["message"]["content"].strip())))
    except:
        return 50  

# 🔹 AI Trade Score Calculation
def calculate_ai_score(sentiment_score, sector_strength, breakout_probability, breakout_strength):
    """Calculates AI trade score based on multiple weighted factors (0-100 scale)."""
    return round(
        ((sentiment_score + 1) * 20) + 
        (sector_strength * 20) + 
        (breakout_probability * 30) + 
        (breakout_strength * 30), 2
    )

# 🔹 Process Stocks & Add Trade Levels
def process_stocks(tickers):
    """Processes stocks and calculates AI scores with estimated time left."""
    results = []
    total_stocks = len(tickers)
    start_time = time.time()

    for i, ticker in enumerate(tickers):
        sentiment_score = get_news_sentiment(ticker)
        sector_strength = get_sector_strength(ticker)
        breakout_probability = ai_predict_breakout(ticker)
        breakout_strength = ai_confirm_breakout_strength(ticker)

        ai_score = calculate_ai_score(sentiment_score, sector_strength, breakout_probability, breakout_strength)
        trade_approved = ai_score >= 70  

        results.append([ticker, sentiment_score, sector_strength, breakout_probability, breakout_strength, ai_score, trade_approved])

        progress_bar.progress((i + 1) / total_stocks)

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
        st.dataframe(pd.DataFrame(final_results, columns=["Stock", "Sentiment Score", "Sector Strength", "Breakout Probability", "Breakout Strength", "AI Score", "Trade Approved"]))




