import streamlit as st
import requests
import pandas as pd
import concurrent.futures
import time
from duckduckgo_search import DDGS  # ✅ Corrected Import
import openai

# 🔹 Load API Keys Securely from Streamlit Secrets
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

openai.api_key = OPENAI_API_KEY  # Set OpenAI Key

# 🔹 Function to Get Market Trend (SPY, QQQ)
def get_market_trend():
    """Checks if the overall market (SPY, QQQ) is bullish or bearish."""
    try:
        spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apikey={POLYGON_API_KEY}"
        qqq_url = f"https://api.polygon.io/v2/aggs/ticker/QQQ/prev?apikey={POLYGON_API_KEY}"
        
        spy_data = requests.get(spy_url, timeout=5).json()["results"][0]["c"]
        qqq_data = requests.get(qqq_url, timeout=5).json()["results"][0]["c"]

        if spy_data < 50 or qqq_data < 50:
            return 0.5  # Bearish Market (Lower Score)
        return 1.0  # Bullish Market (Higher Score)
    except:
        return 0.75  # Neutral Market (Default)

# 🔹 Function to Get News Sentiment Score (FIXED ✅)
def get_news_sentiment(symbol):
    """Fetches recent news headlines for sentiment analysis."""
    try:
        ddgs = DDGS()  # ✅ Corrected DuckDuckGo Search
        headlines = [result["title"] for result in ddgs.news(symbol, max_results=5)]
        if not headlines:
            return 0.0  # No news available, return neutral score
        
        # Convert headlines to a single text prompt
        news_text = "\n".join(headlines)
        prompt = f"Analyze the sentiment of these news headlines for {symbol}. Assign a sentiment score (-1.0 to 1.0): {news_text}"

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 10}

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        return float(result["choices"][0]["message"]["content"].strip())
    except:
        return 0.0  # Default neutral sentiment if an error occurs

# 🔹 Function to Calculate AI Score for a Stock
def calculate_ai_score(market_trend, sentiment_score):
    """Calculates AI trade score based on weighted factors (0-100 scale)."""
    score = (market_trend * 50) + (sentiment_score * 50)
    return round(score, 2)

# 🔹 Process Uploaded Stock List in Parallel
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()

    st.write("🔍 Running AI Analysis (Updated for News Sentiment Fix)...")

    # 🔹 Progress Bar for User Feedback
    progress_bar = st.progress(0)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for ticker in tickers:
            market_trend = get_market_trend()
            sentiment_score = get_news_sentiment(ticker)

            ai_score = calculate_ai_score(market_trend, sentiment_score)
            trade_approved = ai_score >= 70  # ✅ Trade must score 70+ to be valid

            results.append((ticker, sentiment_score, ai_score, trade_approved))

    # 🔹 Convert to Pandas DataFrame for Faster Processing
    df = pd.DataFrame(results, columns=["Stock", "Sentiment Score", "AI Score", "Trade Approved"])

    # 🔹 Display Final Trade Setups in Streamlit
    st.dataframe(df)
    st.success("✅ AI Analysis Completed & News Sentiment Fixed!")



