def fetch_stock_news(ticker, start_date, end_date, max_retries=3, delay=1):
    """
    Fetch news articles for a given ticker between start_date and end_date using DuckDuckGo.
    Includes basic retry logic to handle rate limits.
    """
    query = f"{ticker} news"
    retries = 0
    while retries < max_retries:
        try:
            with DDGS() as ddgs:
                results = ddgs.news(query, max_results=20)
            filtered_results = []
            for article in results:
                if "date" in article:
                    try:
                        article_date = datetime.datetime.strptime(article["date"], "%Y-%m-%d")
                        article_date = article_date.replace(tzinfo=datetime.timezone.utc)
                        if start_date <= article_date < end_date:
                            filtered_results.append(article)
                    except Exception:
                        filtered_results.append(article)
                else:
                    filtered_results.append(article)
            return filtered_results
        except Exception as e:
            if "403" in str(e):
                st.warning(f"Rate limited for {ticker}, retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
                delay *= 2  # exponential backoff
            else:
                st.error(f"Error fetching news for {ticker}: {e}")
                return []
    st.error(f"Max retries exceeded for {ticker}.")
    return []


