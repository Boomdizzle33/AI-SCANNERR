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
st.title("📊 AI Stock Scanner (Now Fixes Sentiment Scores 🚀)")
progress_bar = st.progress(0)
status_text = st.empty()  # 🔹 Live update text
stock_output = st.empty()  # 🔹 Live output table

# 🔹 Get Market Trend (Fetch Only Once)
def get_market_trend():
    """Fetches SPY & QQQ trend once instead of per stock."""
    try:
        spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apikey={POLYGON_API_KEY}"
        qqq_url = f"https://api.polygon.io/v2/aggs/ticker/QQQ/prev?apikey={POLYGON_API_KEY}"
        
        spy = requests.get(spy_url, timeout=5).json()["results"][0]["c"]
        qqq = requests.get(qqq_url, timeout=5).json()["results"][0]["c"]

        return 1.0 if spy > 50 and qqq > 50 else 0.5
    except:
        return 0.75  # Neutral Score

# 🔹 Fetch News Sentiment (Fixes Zero Score Issue)
def get_news_sentiment(symbol):
    """Fetches news sentiment for a stock and ensures correct OpenAI response parsing."""
    try:
        ddgs = DDGS()
        headlines = [result["title"] for result in ddgs.news(symbol, max_results=3)]
        
        if not headlines:
            st.write(f"⚠️ No news found for {symbol}, setting sentiment to neutral (0.0).")
            return 0.0  # No news = neutral sentiment
        
        news_text = "\n".join(headlines)
        prompt = f"""
        Analyze the sentiment of these stock news headlines for {symbol}.
        Assign a sentiment score between -1.0 (very negative) and 1.0 (very positive).
        Headlines:
        {news_text}
        Respond with only a single number.
        """

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 10}

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"].strip()

        # Ensure the response is a valid float number
        try:
            sentiment_score = float(result_text)
            return max(-1.0, min(1.0, sentiment_score))  # Ensure score is between -1.0 and 1.0
        except ValueError:
            st.write(f"⚠️ Invalid OpenAI response for {symbol}: {result_text}. Setting sentiment to neutral (0.0).")
            return 0.0  # Default to neutral if response is invalid
    except:
        return 0.0  # Default to neutral sentiment on error

# 🔹 AI Trade Score Calculation (Ensures Scores Change)
def calculate_ai_score(market_trend, sentiment_score):
    """Calculates AI trade score based on weighted factors (0-100 scale)."""
    return round((market_trend * 50) + ((sentiment_score + 1) * 25), 2)  # Adjusted weighting

# 🔹 Process Stocks Sequentially (Shows Live Progress)
def process_stocks(tickers):
    """Processes stocks one by one, updating Streamlit UI in real-time."""
    market_trend = get_market_trend()
    results = []

    for i, ticker in enumerate(tickers):
        status_text.write(f"🔍 Scanning Stock: **{ticker}** ({i+1}/{len(tickers)})...")

        sentiment_score = get_news_sentiment(ticker)
        ai_score = calculate_ai_score(market_trend, sentiment_score)
        trade_approved = ai_score >= 70  # ✅ Trade must score 70+ to be valid

        results.append([ticker, sentiment_score, ai_score, trade_approved])

        # 🔹 Update Progress Bar & UI
        progress_bar.progress((i + 1) / len(tickers))
        stock_output.write(pd.DataFrame(results, columns=["Stock", "Sentiment Score", "AI Score", "Trade Approved"]))

    return results

# 🔹 File Upload & Scanner Execution
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()

    st.write("🔍 **Scanning Stocks... Please Wait...**")
    final_results = process_stocks(tickers)

    st.success("✅ AI Analysis Completed!")
    st.dataframe(pd.DataFrame(final_results, columns=["Stock", "Sentiment Score", "AI Score", "Trade Approved"]))




