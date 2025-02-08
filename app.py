import streamlit as st
import requests
import pandas as pd
import time
import datetime
import logging
import numpy as np

# -----------------------------------------------------------------------------
# Patch for NumPy: Ensure np.NaN exists for libraries like pandas_ta
# -----------------------------------------------------------------------------
np.NaN = np.nan

import pandas_ta as ta  # For technical indicators (if needed)
from duckduckgo_search import DDGS
import openai

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load API Keys Securely from Streamlit Secrets
# -----------------------------------------------------------------------------
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

# -----------------------------------------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------------------------------------
st.title("📊 AI Stock Scanner (No More Errors! 🚀)")
progress_bar = st.progress(0)
time_remaining_text = st.empty()

# -----------------------------------------------------------------------------
# Function: Fetch News Sentiment
# -----------------------------------------------------------------------------
def get_news_sentiment(symbol):
    """
    Fetches news sentiment for a stock using OpenAI and DuckDuckGo Search.
    It pulls up to 5 news headlines, builds a prompt, and sends it to OpenAI's API.
    """
    try:
        ddgs = DDGS()
        # Fetch up to 5 news headlines
        headlines = [result["title"] for result in ddgs.news(symbol, max_results=5)]
        
        if not headlines:
            logger.info(f"No headlines found for {symbol}.")
            return 0.0  

        news_text = "\n".join(headlines)
        prompt = f"""
Analyze the sentiment of these stock news headlines for {symbol}.
Assign a sentiment score between -1.0 (very negative) and 1.0 (very positive).
Respond with only a single number.
Headlines:
{news_text}
"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        sentiment_value = float(result_text)
        # Ensure the sentiment is within [-1.0, 1.0]
        return max(-1.0, min(1.0, sentiment_value))
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error for {symbol} in get_news_sentiment: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# Function: Fetch Stock Data with Exponential Backoff
# -----------------------------------------------------------------------------
def get_stock_data(symbol):
    """
    Fetches accurate stock price data from Polygon.io for the given symbol.
    Uses exponential backoff to handle rate-limits or transient network issues.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-02-01/2024-02-02?apikey={POLYGON_API_KEY}"
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            # Handle rate-limit (HTTP 429) errors
            if response.status_code == 429:
                st.warning(f"Rate limit reached for {symbol}. Retrying (attempt {attempt + 1})...")
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            
            data = response.json()
            if "results" not in data or not data["results"]:
                logger.info(f"No results found for {symbol}.")
                return 0, 0, 0  # Default values

            # Use the most recent data point
            result = data["results"][-1]
            last_close = result.get("c", 0)
            high_price = result.get("h", 0)
            low_price = result.get("l", 0)

            return round(last_close, 2), round(high_price, 2), round(low_price, 2)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {symbol}: {e}")
            if attempt == 2:
                return 0, 0, 0
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return 0, 0, 0

# -----------------------------------------------------------------------------
# Function: Check for Upcoming Earnings
# -----------------------------------------------------------------------------
def check_earnings_date(symbol):
    """
    Checks if the stock has earnings in the next 7 days using Polygon.io.
    If the API returns a 404 error (no earnings data available for the ticker),
    the function returns a message indicating no earnings risk.
    """
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}/earnings?apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        # If the endpoint returns 404, treat it as no earnings data available.
        if response.status_code == 404:
            return "✅ No earnings risk"
        
        response.raise_for_status()
        data = response.json()
        earnings_list = data.get("results", [])
        
        if earnings_list:
            # Get the upcoming earnings report date (if available)
            upcoming_earnings = earnings_list[0].get("reportDate")
            if upcoming_earnings:
                days_until_earnings = (pd.to_datetime(upcoming_earnings) - pd.to_datetime("today")).days
                if days_until_earnings <= 7:
                    return f"⚠️ Earnings in {days_until_earnings} days ({upcoming_earnings})"
        return "✅ No earnings risk"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error checking earnings for {symbol}: {e}")
        return "❓ Earnings data unavailable"
    except Exception as e:
        logger.error(f"Error processing earnings data for {symbol}: {e}")
        return "❓ Earnings data unavailable"

# -----------------------------------------------------------------------------
# Function: AI Predicts Breakout Probability (Implemented)
# -----------------------------------------------------------------------------
def ai_predict_breakout(symbol):
    """
    Uses historical technical indicators to predict breakout probability.
    
    This function fetches the past 30 days of daily aggregated data for the given symbol,
    computes the 20-day SMA, identifies the highest closing price in that period, and calculates 
    the volatility (std. deviation of daily returns). It then combines these measures into a 
    breakout probability between 0 and 100.
    """
    try:
        # Define date range: last 30 days
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=30)
        end_date_str = end_date.strftime("%Y-%m-%d")
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Build the URL for Polygon.io's aggregates endpoint (daily data)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start_date_str}/{end_date_str}?apikey={POLYGON_API_KEY}"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data or not data["results"]:
            logger.info(f"No historical data available for {symbol}.")
            return 50  # Return a neutral probability
        
        results = data["results"]
        # Sort by timestamp ascending
        results.sort(key=lambda x: x["t"])
        
        # Extract closing prices (skip any zero values)
        closes = [r.get("c", 0) for r in results if r.get("c", 0) != 0]
        if not closes:
            return 50
        
        # Get the most recent closing price
        current_close = closes[-1]
        
        # Calculate the 20-day simple moving average (SMA)
        period = min(20, len(closes))
        sma20 = sum(closes[-period:]) / period
        
        # Determine the historical high over the period
        historical_max = max(closes)
        
        # Start with a base probability (neutral score)
        probability = 50
        
        # Factor 1: Trend bias (if current price is above the SMA, add a bonus)
        if current_close > sma20:
            probability += 15
        
        # Factor 2: Proximity to historical high (closer to the max suggests a breakout)
        if historical_max > 0:
            gap_percentage = (historical_max - current_close) / historical_max
            if gap_percentage <= 0.02:
                probability += 20
            elif gap_percentage <= 0.05:
                probability += 10
        
        # Factor 3: Volatility adjustment (standard deviation of daily returns)
        if len(closes) >= 2:
            returns = np.diff(np.array(closes)) / np.array(closes[:-1])
            volatility = np.std(returns)
            if volatility < 0.02:
                probability += 10
            elif volatility > 0.05:
                probability -= 10
        
        # Ensure the probability is between 0 and 100
        probability = max(0, min(100, probability))
        return round(probability, 2)
    
    except Exception as e:
        logger.error(f"Error predicting breakout for {symbol}: {e}")
        return 50  # Default neutral probability on error

# -----------------------------------------------------------------------------
# Function: Calculate Entry Price Based on Market Data
# -----------------------------------------------------------------------------
def calculate_entry_price(last_close, high, low):
    """
    Calculates an optimal entry price based on the stock market conditions.
    Uses the day's high and low to determine a fair entry price.
    """
    if last_close == 0 or high == 0 or low == 0:
        return 0  # Default value
    if abs(last_close - low) <= 0.02 * last_close:
        return round(low, 2)
    elif abs(last_close - high) <= 0.02 * last_close:
        return round(high, 2)
    else:
        return round((high + low) / 2, 2)

# -----------------------------------------------------------------------------
# Function: Process Stocks & Update UI
# -----------------------------------------------------------------------------
def process_stocks(tickers):
    """
    Processes a list of tickers by fetching sentiment, breakout probability,
    stock data, earnings info, and then computes an overall AI score.
    The progress bar and estimated time remaining are updated in the UI.
    """
    results = []
    total_stocks = len(tickers)
    start_time = time.time()

    for i, ticker in enumerate(tickers):
        # Fetch and compute data for each ticker
        sentiment_score = get_news_sentiment(ticker)
        breakout_probability = ai_predict_breakout(ticker)
        last_close, high, low = get_stock_data(ticker)
        entry_price = calculate_entry_price(last_close, high, low)
        earnings_warning = check_earnings_date(ticker)
        ai_score = round((sentiment_score * 20) + (breakout_probability * 0.8), 2)
        trade_approved = "✅ Yes" if ai_score >= 75 and breakout_probability >= 80 else "❌ No"

        results.append([
            ticker, sentiment_score, breakout_probability, 
            ai_score, trade_approved, earnings_warning,
            last_close, entry_price, low, high
        ])

        # Update progress bar and estimated time remaining
        progress = (i + 1) / total_stocks
        progress_bar.progress(progress)
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        time_remaining = estimated_total_time - elapsed_time
        time_remaining_text.text(f"⏳ Estimated Time Left: {time_remaining:.2f} seconds")

    return results

# -----------------------------------------------------------------------------
# File Upload & Scanner Execution
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    try:
        stock_list = pd.read_csv(uploaded_file)
        # Validate CSV format
        if "Ticker" not in stock_list.columns:
            st.error("CSV file must contain a 'Ticker' column.")
        else:
            tickers = stock_list["Ticker"].dropna().tolist()
            if st.button("Run AI Scanner"):
                st.write("🔍 **Scanning Stocks... Please Wait...**")
                final_results = process_stocks(tickers)
                st.success("✅ AI Analysis Completed!")
                results_df = pd.DataFrame(final_results, columns=[
                    "Stock", "Sentiment Score", "Breakout Probability",  
                    "AI Score", "Trade Approved", "Earnings Alert",
                    "Last Close Price", "Entry Price", "Support Level (Low)", "Resistance Level (High)"
                ])
                st.dataframe(results_df)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

