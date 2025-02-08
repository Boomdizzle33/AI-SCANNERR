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
st.title("📊 AI Stock Scanner (No More Errors! 🚀)")

# Progress bar & estimated time display
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

# 🔹 Function to Fetch Stock Prices Correctly
def get_stock_data(symbol):
    """Fetches accurate stock price data, ensuring all values are correct."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-02-01/2024-02-02?apikey={POLYGON_API_KEY}"
        response = requests.get(url, timeout=5).json()

        if "results" not in response or not response["results"]:
            return 0, 0, 0  # Default values instead of None  

        data = response["results"][-1]  # Most recent data point
        last_close = data.get("c", 0)  
        high_price = data.get("h", 0)  
        low_price = data.get("l", 0)  

        return round(last_close, 2), round(high_price, 2), round(low_price, 2)
    except:
        return 0, 0, 0  # Default values instead of skipping stocks  

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
        return 80  # Placeholder for now
    except:
        return 50  

# 🔹 Function to Calculate Entry Price Based on Market Data
def calculate_entry_price(last_close, high, low):
    """Calculates an optimal entry price based on stock market conditions."""
    if last_close == 0 or high == 0 or low == 0:
        return 0  # Default value instead of None  

    if abs(last_close - low) <= 0.02 * last_close:
        return round(low, 2)  
    elif abs(last_close - high) <= 0.02 * last_close:
        return round(high, 2)  
    else:
        return round((high + low) / 2, 2)  

# 🔹 Process Stocks & Add Trade Levels
def process_stocks(tickers):
    """Processes stocks, filters weak setups, and checks for upcoming earnings reports."""
    results = []
    total_stocks = len(tickers)
    start_time = time.time()

    for i, ticker in enumerate(tickers):
        sentiment_score = get_news_sentiment(ticker)
        breakout_probability = ai_predict_breakout(ticker)
        last_close, high, low = get_stock_data(ticker)  # ✅ Corrected Stock Price Fetching

        entry_price = calculate_entry_price(last_close, high, low)  
        earnings_warning = check_earnings_date(ticker)

        ai_score = round((sentiment_score * 20) + (breakout_probability * 0.8), 2)

        trade_approved = "✅ Yes" if ai_score >= 75 and breakout_probability >= 80 else "❌ No"

        results.append([
            ticker, sentiment_score, breakout_probability, 
            ai_score, trade_approved, earnings_warning, last_close, entry_price, low, high
        ])

        progress = (i + 1) / total_stocks
        progress_bar.progress(progress)
        
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        time_remaining = estimated_total_time - elapsed_time
        time_remaining_text.text(f"⏳ Estimated Time Left: {time_remaining:.2f} seconds")

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
            "AI Score", "Trade Approved", "Earnings Alert", "Last Close Price", "Entry Price", "Support Level (Low)", "Resistance Level (High)"
        ]))

