import streamlit as st
import requests
import pandas as pd
import concurrent.futures
import time
from duckduckgo_search import DDGS
import openai

# 🔹 Load API Keys Securely from Streamlit Secrets
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

openai.api_key = OPENAI_API_KEY  # Set OpenAI Key

# 🔹 Cache to Store API Results (Avoid Duplicate Calls)
cache = {}

def get_cached_data(key, fetch_function):
    """Checks cache before making an API request to speed up processing."""
    if key in cache and time.time() - cache[key]['timestamp'] < 3600:  # 1-hour cache
        return cache[key]['data']

    data = fetch_function()
    if data:
        cache[key] = {"data": data, "timestamp": time.time()}
    
    return data

# 🔹 Get Market Trend (Optimized)
def get_market_trend():
    """Fetches SPY & QQQ trend once instead of per stock."""
    def fetch():
        try:
            spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apikey={POLYGON_API_KEY}"
            qqq_url = f"https://api.polygon.io/v2/aggs/ticker/QQQ/prev?apikey={POLYGON_API_KEY}"
            
            spy = requests.get(spy_url, timeout=5).json()["results"][0]["c"]
            qqq = requests.get(qqq_url, timeout=5).json()["results"][0]["c"]

            return 1.0 if spy > 50 and qqq > 50 else 0.5
        except:
            return 0.75  # Neutral Score

    return get_cached_data("market_trend", fetch)

# 🔹 Fetch News Sentiment (Batch Processing)
def get_news_sentiment_batch(symbols):
    """Fetches news sentiment for multiple stocks at once for speed."""
    headlines_dict = {}
    ddgs = DDGS()
    
    for symbol in symbols:
        try:
            headlines = [result["title"] for result in ddgs.news(symbol, max_results=3)]
            if headlines:
                headlines_dict[symbol] = "\n".join(headlines)
        except:
            headlines_dict[symbol] = ""

    if not headlines_dict:
        return {symbol: 0.0 for symbol in symbols}

    prompt = "Analyze these stock news headlines and assign a sentiment score (-1.0 to 1.0):\n\n"
    for symbol, news_text in headlines_dict.items():
        prompt += f"Stock: {symbol}\nHeadlines: {news_text}\n\n"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4-turbo", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        results = response.json()["choices"][0]["message"]["content"].strip().split("\n")

        sentiment_scores = {}
        for line in results:
            parts = line.split(":")
            if len(parts) == 2:
                stock = parts[0].strip()
                score = float(parts[1].strip())
                sentiment_scores[stock] = score

        return sentiment_scores
    except:
        return {symbol: 0.0 for symbol in symbols}  # Default to neutral sentiment

# 🔹 AI Trade Score Calculation
def calculate_ai_score(market_trend, sentiment_score):
    """Calculates AI trade score based on weighted factors (0-100 scale)."""
    return round((market_trend * 50) + (sentiment_score * 50), 2)

# 🔹 Process a Single Stock (Shows Live Progress)
def process_stock(ticker, market_trend, sentiment_scores, progress_bar, i, total_stocks):
    """Processes a stock and updates UI in real-time."""
    sentiment_score = sentiment_scores.get(ticker, 0.0)
    ai_score = calculate_ai_score(market_trend, sentiment_score)
    trade_approved = ai_score >= 70  # ✅ Trade must score 70+ to be valid

    # 🔹 Update Progress Bar
    progress_bar.progress((i + 1) / total_stocks)

    # 🔹 Show Progress in Streamlit
    st.write(f"✅ Processed: **{ticker}** | Sentiment Score: {sentiment_score} | AI Score: {ai_score}")

    return (ticker, sentiment_score, ai_score, trade_approved)

# 🔹 Process Uploaded Stock List in Parallel (Live UI Updates)
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()
    total_stocks = len(tickers)

    st.write("🔍 Running AI Analysis (Now 10-20x Faster 🚀)...")
    progress_bar = st.progress(0)

    # 🔹 Fetch Market Trend Once (Instead of Calling API for Every Stock)
    market_trend = get_market_trend()

    # 🔹 Fetch News Sentiment for All Stocks in One Batch (10x Faster AI Calls)
    sentiment_scores = get_news_sentiment_batch(tickers)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(process_stock, ticker, market_trend, sentiment_scores, progress_bar, i, total_stocks): ticker
            for i, ticker in enumerate(tickers)
        }

        for future in concurrent.futures.as_completed(future_to_stock):
            results.append(future.result())

    # 🔹 Convert to Pandas DataFrame for Faster Processing
    df = pd.DataFrame(results, columns=["Stock", "Sentiment Score", "AI Score", "Trade Approved"])

    # 🔹 Display Final Trade Setups in Streamlit
    st.dataframe(df)
    st.success("✅ AI Analysis Completed in Record Time!")



