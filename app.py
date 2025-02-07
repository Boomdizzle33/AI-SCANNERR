import streamlit as st
import requests
import pandas as pd
import concurrent.futures

# 🔹 App Title
st.set_page_config(page_title="📈 AI Stock Scanner", layout="wide")
st.title("📊 AI-Powered Stock Trading WebApp")

st.write("Upload your stock list to analyze swing trade setups using AI.")

# 🔹 File Upload for Stock List
uploaded_file = st.file_uploader("Upload TradingView CSV File", type=["csv"])

# 🔹 API Keys (Will Be Set in Streamlit Secrets)
POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# 🔹 Function to Fetch Stock Data from Polygon.io
def get_stock_data(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "results" not in data or not data["results"]:
            return None
        stock_info = data["results"][0]
        return {
            "symbol": symbol,
            "close": stock_info["c"],
            "volume": stock_info["v"],
            "high": stock_info["h"],
            "low": stock_info["l"]
        }
    except requests.RequestException:
        return None  

# 🔹 Function to Call OpenAI for AI Stock Analysis
def get_ai_analysis(stock_data):
    prompt = f"""
    Analyze the following stock for swing trading:

    Stock Symbol: {stock_data['symbol']}
    Last Close Price: {stock_data['close']}
    Volume: {stock_data['volume']}
    Recent High: {stock_data['high']}
    Recent Low: {stock_data['low']}
    
    Based on trend analysis, institutional flow, sentiment, and market conditions, 
    estimate the probability of a successful swing trade (0-100%) and suggest 
    an entry price, stop-loss, and profit target.

    Format response as:
    Probability: XX%
    Entry: $XX.XX
    Stop-Loss: $XX.XX
    Target: $XX.XX
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException:
        return "Error retrieving AI analysis"

# 🔹 Process Uploaded Stock List
if uploaded_file:
    stock_list = pd.read_csv(uploaded_file)
    tickers = stock_list["Ticker"].tolist()

    st.write("Running AI Analysis on Stocks...")

    swing_trade_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for ticker in tickers:
            stock_data = get_stock_data(ticker)
            if stock_data:
                ai_analysis = get_ai_analysis(stock_data)
                swing_trade_results.append((ticker, ai_analysis))

    # Display Final Trade Setups
    df = pd.DataFrame(swing_trade_results, columns=["Stock", "AI Analysis"])
    st.dataframe(df)
