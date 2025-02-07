import streamlit as st
import requests
import pandas as pd
import concurrent.futures
import time
from duckduckgo_search import ddg_news
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

# 🔹 Function to Check Sector Strength
def get_sector_strength(symbol):
    """Assigns a strength score to the stock's sector."""
    sector_etfs = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Consumer Discretionary": "XLY"
    }

    sector = sector_etfs.get("Technology")  # Replace with actual sector lookup
    if not sector:
        return 0.5  # Neutral Score

    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{sector}/prev?apikey={POLYGON_API_KEY}"
        sector_data = requests.get(url, timeout=5).json()
        sector_performance = sector_data["results"][0]["c"]
        return 1.0 if sector_performance > 50 else 0.5  # Strong = 1.0, Weak = 0.5
    except:
        return 0.5  # Neutral Score

# 🔹 Function to Calculate Relative Strength (RS)
def get_relative_strength(symbol):
    """Calculates relative strength of a stock vs. SPY (0-1 scale)."""
    try:
        stock_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-11-01/2024-02-01?apikey={POLYGON_API_KEY}"
        spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2023-11-01/2024-02-01?apikey={POLYGON_API_KEY}"
        
        stock_data = requests.get(stock_url, timeout=5).json()
        spy_data = requests.get(spy_url, timeout=5).json()

        stock_return = (stock_data["results"][-1]["c"] - stock_data["results"][0]["c"]) / stock_data["results"][0]["c"]
        spy_return = (spy_data["results"][-1]["c"] - spy_data["results"][0]["c"]) / spy_data["results"][0]["c"]

        rs_score = (stock_return / spy_return)
        return min(1.0, max(0, rs_score))  # Normalize to 0-1 scale
    except:
        return 0.5  # Neutral Score

# 🔹 Function to Get News Sentiment Score
def get_news_sentiment(symbol):
    """Fetches news sentiment using AI analysis."""
    headlines = ddg_news(symbol, safesearch="Off", time="d", max_results=5)
    if not headlines:
        return 0.5  

    news_text = "\n".join([h['title'] for h in headlines])
    prompt = f"Analyze these stock news headlines for {symbol}. Assign a sentiment score (-1.0 to 1.0): {news_text}"

    data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 10}

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"})
        return float(response.json()["choices"][0]["message"]["content"].strip())
    except:
        return 0.5  # Neutral Score

# 🔹 AI Trade Score Calculation
def calculate_ai_score(market_trend, sector_strength, rs_score, sentiment_score):
    """Calculates AI trade score based on weighted factors (0-100 scale)."""
    score = (market_trend * 30) + (sector_strength * 15) + (rs_score * 25) + (sentiment_score * 30)
    return round(score, 2)

# 🔹 Process Uploaded Stock List in Parallel
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()

    st.write("🔍 Running AI Analysis (Optimized for Deployment)...")

    # 🔹 Progress Bar for User Feedback
    progress_bar = st.progress(0)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for ticker in tickers:
            market_trend = get_market_trend()
            sector_strength = get_sector_strength(ticker)
            rs_score = get_relative_strength(ticker)
            sentiment_score = get_news_sentiment(ticker)

            ai_score = calculate_ai_score(market_trend, sector_strength, rs_score, sentiment_score)
            trade_approved = ai_score >= 70  # ✅ Trade must score 70+ to be valid

            results.append((ticker, rs_score, sentiment_score, ai_score, trade_approved))

    # 🔹 Convert to Pandas DataFrame for Faster Processing
    df = pd.DataFrame(results, columns=["Stock", "RS Score", "Sentiment", "AI Score", "Trade Approved"])

    # 🔹 Display Final Trade Setups in Streamlit
    st.dataframe(df)
    st.success("✅ AI Analysis Completed & Ready for Deployment!")



