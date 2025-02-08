import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import datetime
import pandas_ta as ta  # For technical indicators
import logging
from duckduckgo_search import DDGS
import openai

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Load API Keys Securely from Streamlit Secrets
# ------------------------------------------------------------------------------
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

openai.api_key = OPENAI_API_KEY

# ------------------------------------------------------------------------------
# Global Risk Parameters
# ------------------------------------------------------------------------------
# Example account balance; in production, fetch this dynamically.
account_balance = 10000  
risk_per_trade = 0.02 * account_balance  # Risk 2% of account per trade

# ------------------------------------------------------------------------------
# Function: Get News Sentiment using DuckDuckGo & OpenAI
# ------------------------------------------------------------------------------
def get_news_sentiment(symbol):
    """
    Fetches news sentiment for a given symbol by gathering headlines and using
    GPT-4 to generate a sentiment score between -1.0 (very negative) and 1.0 (very positive).
    """
    try:
        ddgs = DDGS()
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
        # Clamp sentiment to the range [-1.0, 1.0]
        return max(-1.0, min(1.0, sentiment_value))
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        return 0.0

# ------------------------------------------------------------------------------
# Function: Placeholder AI Breakout Prediction
# ------------------------------------------------------------------------------
def ai_predict_breakout(symbol):
    """
    Placeholder for breakout prediction. In a real model, you would
    incorporate technical analysis signals here.
    """
    try:
        # Replace with more robust technical analysis as needed.
        return 80  # Example: fixed score (0-100)
    except Exception as e:
        logger.error(f"Error predicting breakout for {symbol}: {e}")
        return 50

# ------------------------------------------------------------------------------
# Function: Fetch Historical Price Data from Polygon.io
# ------------------------------------------------------------------------------
def get_historical_data(symbol, start, end):
    """
    Retrieves historical daily price data from Polygon.io for a given symbol.
    Expects the API to return JSON with a "results" list.
    """
    try:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start}/{end}?apikey={POLYGON_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" not in data or len(data["results"]) == 0:
            logger.info(f"No historical data for {symbol}.")
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        # Convert the epoch time (milliseconds) to datetime
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# Function: Calculate Technical Indicators using pandas_ta
# ------------------------------------------------------------------------------
def calculate_technical_indicators(df):
    """
    Calculates ATR and SMA50 as technical indicators.
    Assumes df has columns: 'h' (high), 'l' (low), 'c' (close).
    """
    try:
        df['ATR'] = ta.atr(high=df['h'], low=df['l'], close=df['c'], length=14)
        df['SMA50'] = ta.sma(close=df['c'], length=50)
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df

# ------------------------------------------------------------------------------
# Function: Compute Entry, Stop Loss, Profit Target, and Position Size
# ------------------------------------------------------------------------------
def compute_entry_stop_profit(symbol):
    """
    Using recent historical data and technical indicators, calculates:
      - Entry price (near support/breakout)
      - Stop loss (using ATR as a proxy for volatility)
      - Profit target (3:1 risk/reward ratio)
      - Position size (based on risking 2% of the account)
    """
    # Use a rolling 60-day window of data
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    df = get_historical_data(symbol, start_date, end_date)
    if df.empty:
        return None, None, None, None
    
    df = calculate_technical_indicators(df)
    latest = df.iloc[-1]
    current_price = latest['c']
    # Use ATR as a measure of volatility; if not available, default to 2% of price.
    atr = latest['ATR'] if not pd.isna(latest['ATR']) else current_price * 0.02
    # Use SMA50 as a proxy for support. If current price is above SMA50, we assume bullish support.
    sma50 = latest['SMA50'] if not pd.isna(latest['SMA50']) else current_price * 0.98

    # Determine entry: if current price is above support, use current price; otherwise, use support with a small buffer.
    entry_price = current_price if current_price > sma50 else sma50 * 1.005

    # Stop loss: 1 ATR below the entry price.
    stop_loss = entry_price - atr

    # Calculate risk per share (entry - stop loss)
    risk_amount = entry_price - stop_loss
    # Profit target: 3 times the risk
    profit_target = entry_price + (risk_amount * 3)

    # Position size: (2% of account) / (risk per share)
    position_size = risk_per_trade / risk_amount if risk_amount > 0 else 0

    return entry_price, stop_loss, profit_target, position_size

# ------------------------------------------------------------------------------
# Function: Analyze Stock and Return Trade Setup if Conditions are Met
# ------------------------------------------------------------------------------
def analyze_stock(symbol):
    """
    Combines sentiment analysis and breakout prediction with technical analysis.
    Returns a dictionary with trade setup details if the stock meets criteria.
    """
    sentiment_score = get_news_sentiment(symbol)
    breakout_signal = ai_predict_breakout(symbol)
    
    # Define criteria for an optimal setup.
    if sentiment_score > 0.3 and breakout_signal > 70:
        entry, stop, target, size = compute_entry_stop_profit(symbol)
        if entry and stop and target and size:
            return {
                "Symbol": symbol,
                "Entry": round(entry, 2),
                "Stop": round(stop, 2),
                "Target": round(target, 2),
                "Position Size": round(size, 2),
                "Sentiment": sentiment_score,
                "Breakout Signal": breakout_signal
            }
    return None

# ------------------------------------------------------------------------------
# Streamlit UI: Main Execution
# ------------------------------------------------------------------------------
st.title("📊 AI Stock Scanner with Entry, Stop, and Target")

uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])
if uploaded_file:
    try:
        stock_list = pd.read_csv(uploaded_file)
        if "Ticker" not in stock_list.columns:
            st.error("CSV file must contain a 'Ticker' column.")
        else:
            tickers = stock_list["Ticker"].dropna().tolist()
            results = []
            for ticker in tickers:
                st.write(f"Analyzing {ticker}...")
                analysis = analyze_stock(ticker)
                if analysis:
                    results.append(analysis)
                time.sleep(1)  # Small delay to avoid overwhelming APIs; adjust as needed.
            if results:
                st.success("Trade setups found!")
                st.dataframe(pd.DataFrame(results))
            else:
                st.write("No optimal setups found based on current criteria.")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")

