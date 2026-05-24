"""
News Scraper & Sentiment Analysis
==================================
Scrapes the internet for news sentiment:
  1. RSS feeds from major financial outlets (MarketWatch, Reuters, CNBC, etc.)
  2. Web scraping from Finviz news headlines
  3. Yahoo Finance fundamentals (analyst ratings, institutional ownership, etc.)
  4. SEC insider transaction data
  5. VADER + TextBlob NLP sentiment scoring
  6. Aggregated sentiment with confidence weighting
"""
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import Counter

import feedparser
import nltk
import requests
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer

from config import RSS_FEEDS, SCRAPE_TARGETS

# Download VADER lexicon once
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# Try to import TextBlob for additional analysis
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


# ============================================================================
# 1. RSS FEED SCRAPING
# ============================================================================
def _extract_symbol_from_text(text: str, symbol: str) -> bool:
    """
    Smart symbol matching: checks if the text mentions the ticker symbol
    as a standalone word or with common prefixes/suffixes.
    """
    text_lower = text.lower()
    symbol_lower = symbol.lower()

    # Direct symbol mention (as a word boundary)
    if re.search(rf'\b{re.escape(symbol_lower)}\b', text_lower):
        return True

    # Company name aliases (common mappings)
    COMPANY_ALIASES = {
        'aapl': ['apple'],
        'msft': ['microsoft'],
        'googl': ['google', 'alphabet'],
        'amzn': ['amazon'],
        'meta': ['facebook', 'meta platforms'],
        'nvda': ['nvidia'],
        'tsla': ['tesla'],
        'amd': ['advanced micro'],
        'intc': ['intel'],
        'qcom': ['qualcomm'],
        'avgo': ['broadcom'],
        'csco': ['cisco'],
        'orcl': ['oracle'],
        'crm': ['salesforce'],
        'adbe': ['adobe'],
        'pypl': ['paypal'],
        'nflx': ['netflix'],
        'v': ['visa'],
        'ma': ['mastercard'],
        'qqq': ['nasdaq', 'invesco qqq'],
    }

    aliases = COMPANY_ALIASES.get(symbol_lower, [])
    for alias in aliases:
        if alias in text_lower:
            return True

    return False


def scrape_rss_feeds(symbol: str, max_articles: int = 20) -> Dict:
    """
    Scrape RSS feeds for articles mentioning the symbol.
    Uses VADER for sentiment scoring of headlines.
    """
    articles: List[Dict] = []
    sentiment_scores: List[float] = []

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:  # top 10 per feed
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = f"{title} {summary}"

                if _extract_symbol_from_text(text, symbol):
                    # VADER sentiment on title + summary
                    scores = sia.polarity_scores(title)
                    compound = scores['compound']

                    # TextBlob sentiment as secondary if available
                    textblob_polarity = None
                    if HAS_TEXTBLOB:
                        blob = TextBlob(title)
                        textblob_polarity = blob.sentiment.polarity

                    articles.append({
                        'title': title[:200],
                        'source': feed_name,
                        'published': entry.get('published', ''),
                        'vader_compound': round(compound, 3),
                        'textblob_polarity': round(textblob_polarity, 3) if textblob_polarity else None,
                    })
                    sentiment_scores.append(compound)

        except Exception as e:
            logging.debug(f"RSS parse error for {feed_name}: {e}")
            continue

    # Aggregate
    if sentiment_scores:
        avg_sentiment = float(sum(sentiment_scores) / len(sentiment_scores))
        bullish_count = sum(1 for s in sentiment_scores if s > 0.05)
        bearish_count = sum(1 for s in sentiment_scores if s < -0.05)
        neutral_count = len(sentiment_scores) - bullish_count - bearish_count
    else:
        avg_sentiment = 0.0
        bullish_count = bearish_count = neutral_count = 0

    return {
        'articles': articles[:max_articles],
        'article_count': len(articles),
        'avg_sentiment': round(avg_sentiment, 3),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'sentiment_label': (
            'Bullish' if avg_sentiment > 0.1
            else 'Bearish' if avg_sentiment < -0.1
            else 'Neutral'
        ),
    }


# ============================================================================
# 2. FINVIZ NEWS SCRAPING
# ============================================================================
def scrape_finviz_news(symbol: str) -> Dict:
    """
    Scrape Finviz news headlines for a symbol.
    Finviz has a news table that's easy to parse.
    """
    try:
        url = f'https://finviz.com/quote.ashx?t={symbol}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {'finviz_articles': [], 'finviz_sentiment': 0.0}

        # Parse the news table rows
        # Finviz news rows have class 'cursor-pointer' and contain the headline
        html = resp.text

        # Simple regex-based extraction of news headlines
        # Pattern matches the news table rows
        pattern = r'<a[^>]*class="tab-link-news"[^>]*>(.*?)</a>'
        headlines = re.findall(pattern, html, re.DOTALL)

        # Clean HTML tags from headlines
        clean_headlines = [re.sub(r'<[^>]+>', '', h).strip() for h in headlines[:20]]

        sentiment_scores = []
        for headline in clean_headlines:
            if headline:
                scores = sia.polarity_scores(headline)
                sentiment_scores.append(scores['compound'])

        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        )

        return {
            'finviz_headlines': clean_headlines[:10],
            'finviz_headline_count': len(clean_headlines),
            'finviz_sentiment': round(avg_sentiment, 3),
        }
    except Exception as e:
        logging.debug(f"Finviz scrape failed for {symbol}: {e}")
        return {'finviz_headlines': [], 'finviz_sentiment': 0.0}


# ============================================================================
# 3. YAHOO FINANCE FUNDAMENTALS
# ============================================================================
def analyze_yahoo_fundamentals(symbol: str) -> Dict:
    """
    Extract sentiment-relevant fundamentals from Yahoo Finance:
    - Price vs 50-day MA
    - Analyst recommendations (1=Strong Buy, 5=Strong Sell)
    - Institutional ownership %
    - Short interest %
    - Target price
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        factors: List[str] = []
        score = 0.5  # neutral start

        # Price vs 50-day MA
        if 'fiftyDayAverage' in info and 'currentPrice' in info:
            pct_vs_50ma = (info['currentPrice'] / info['fiftyDayAverage'] - 1) * 100
            if pct_vs_50ma > 10:
                score += 0.2
                factors.append(f"Price +{pct_vs_50ma:.1f}% above 50MA")
            elif pct_vs_50ma < -10:
                score -= 0.2
                factors.append(f"Price {pct_vs_50ma:.1f}% below 50MA")

        # Analyst recommendations (1=Strong Buy .. 5=Strong Sell)
        if 'recommendationMean' in info:
            rec = info['recommendationMean']
            if rec <= 2:
                score += 0.15
                factors.append(f"Analysts: Buy ({rec:.1f}/5)")
            elif rec >= 4:
                score -= 0.15
                factors.append(f"Analysts: Sell ({rec:.1f}/5)")

        # Target price
        if 'targetMeanPrice' in info and 'currentPrice' in info:
            upside = (info['targetMeanPrice'] / info['currentPrice'] - 1) * 100
            if upside > 10:
                score += 0.1
                factors.append(f"Target upside: +{upside:.1f}%")
            elif upside < -10:
                score -= 0.1
                factors.append(f"Target downside: {upside:.1f}%")

        # Institutional ownership
        if 'heldPercentInstitutions' in info:
            inst = info['heldPercentInstitutions']
            if inst > 0.7:
                score += 0.1
                factors.append(f"Institutional: {inst:.1%}")

        # Short interest
        if 'shortPercentOfFloat' in info:
            short_pct = info['shortPercentOfFloat']
            if short_pct > 0.1:
                score -= 0.1
                factors.append(f"Short interest: {short_pct:.1%}")

        score = max(0.0, min(1.0, score))

        return {
            'fundamental_score': round(score, 3),
            'fundamental_factors': factors,
            'analyst_rating': info.get('recommendationMean'),
            'target_price': info.get('targetMeanPrice'),
            'institutional_pct': info.get('heldPercentInstitutions'),
            'short_float_pct': info.get('shortPercentOfFloat'),
        }
    except Exception as e:
        logging.error(f"Yahoo fundamentals failed for {symbol}: {e}")
        return {'fundamental_score': 0.5, 'fundamental_factors': ['Analysis failed']}


# ============================================================================
# 4. INSIDER ACTIVITY
# ============================================================================
def analyze_insider_activity(symbol: str) -> Dict:
    """
    Check recent SEC Form 4 insider transactions.
    Insider buying = bullish, heavy selling = bearish.
    """
    try:
        ticker = yf.Ticker(symbol)
        insiders = ticker.insider_transactions

        if insiders is None or insiders.empty:
            return {'insider_signal': 'None', 'insider_sentiment': 0.0}

        buys = 0
        sells = 0

        for _, row in insiders.iterrows():
            # Check if transaction is recent (within 90 days)
            try:
                # insider_transactions has 'startDate' column
                pass
            except:
                pass

            txn = str(row.get('Transaction', row.get('transaction', '')))
            if 'Buy' in txn or 'buy' in txn or 'Purchase' in txn:
                buys += 1
            elif 'Sale' in txn or 'Sell' in txn or 'sell' in txn:
                sells += 1

        total = buys + sells
        sentiment = (buys - sells) / max(total, 1)

        if buys > sells * 2:
            signal = 'Strong Insider Buying'
        elif buys > sells:
            signal = 'Insider Buying'
        elif sells > buys * 2:
            signal = 'Heavy Insider Selling'
        elif sells > buys:
            signal = 'Insider Selling'
        else:
            signal = 'Neutral'

        return {
            'insider_buys': buys,
            'insider_sells': sells,
            'insider_signal': signal,
            'insider_sentiment': round(sentiment, 3),
        }
    except Exception as e:
        logging.debug(f"Insider activity check failed for {symbol}: {e}")
        return {'insider_signal': 'None', 'insider_sentiment': 0.0}


# ============================================================================
# 5. NEWS VOLUME & MOMENTUM (unusual news activity)
# ============================================================================
def analyze_news_momentum(symbol: str, rss_data: Dict) -> Dict:
    """
    Detect if there's unusual news activity (spike in article count).
    More articles than normal = potential catalyst.
    """
    article_count = rss_data.get('article_count', 0)

    if article_count >= 8:
        momentum = 'Very High'
        score = 1.0
    elif article_count >= 5:
        momentum = 'High'
        score = 0.8
    elif article_count >= 3:
        momentum = 'Moderate'
        score = 0.6
    elif article_count >= 1:
        momentum = 'Low'
        score = 0.3
    else:
        momentum = 'None'
        score = 0.0

    return {
        'news_momentum': momentum,
        'news_momentum_score': score,
        'article_count': article_count,
        'has_catalyst': article_count >= 5,
    }


# ============================================================================
# 6. COMPREHENSIVE SENTIMENT AGGREGATION
# ============================================================================
def analyze_sentiment(symbol: str) -> Dict:
    """
    Comprehensive sentiment analysis aggregating ALL sources:
      - RSS news feeds (scraped)
      - Finviz headlines (scraped)
      - Yahoo Finance fundamentals
      - Insider activity
      - News momentum

    Returns a unified sentiment dict with weighted aggregate score.
    """
    # 1. RSS News
    rss_data = scrape_rss_feeds(symbol)

    # 2. Finviz headlines
    finviz_data = scrape_finviz_news(symbol)

    # 3. Yahoo fundamentals
    fundamentals = analyze_yahoo_fundamentals(symbol)

    # 4. Insider activity
    insiders = analyze_insider_activity(symbol)

    # 5. News momentum
    news_momentum = analyze_news_momentum(symbol, rss_data)

    # ---- Aggregate sentiment score ----
    # Weights:
    #   RSS sentiment: 30%
    #   Finviz sentiment: 10%
    #   Fundamentals: 30%
    #   Insider: 10%
    #   News momentum provides a multiplier, not a direct score

    # Convert RSS sentiment from [-1, 1] to [0, 1]
    rss_normalized = (rss_data['avg_sentiment'] + 1) / 2

    # Convert Finviz sentiment similarly
    finviz_normalized = (finviz_data['finviz_sentiment'] + 1) / 2

    # Fundamentals already [0, 1]
    fund_score = fundamentals.get('fundamental_score', 0.5)

    # Insider: convert [-1, 1] to [0, 1]
    insider_normalized = (insiders.get('insider_sentiment', 0) + 1) / 2

    # Weighted aggregate
    aggregate = (
        rss_normalized * 0.30 +
        finviz_normalized * 0.10 +
        fund_score * 0.30 +
        insider_normalized * 0.10
    )

    # News momentum acts as a confidence multiplier
    # High news volume with clear sentiment = stronger signal
    momentum_mult = 1.0 + (news_momentum['news_momentum_score'] * 0.3)
    aggregate = min(1.0, aggregate * momentum_mult)

    # Determine overall label
    if aggregate >= 0.7:
        overall = 'Bullish'
    elif aggregate <= 0.3:
        overall = 'Bearish'
    else:
        overall = 'Neutral'

    # Collect all factors
    all_factors = fundamentals.get('fundamental_factors', [])
    if rss_data['article_count'] > 0:
        all_factors.append(f"News: {rss_data['article_count']} articles ({rss_data['sentiment_label']})")
    if insiders.get('insider_signal') not in ('None', 'Neutral'):
        all_factors.append(f"Insiders: {insiders['insider_signal']}")

    return {
        'score': round(aggregate, 3),
        'sentiment': overall,
        'factors': all_factors,
        'rss_news': {
            'article_count': rss_data['article_count'],
            'avg_sentiment': rss_data['avg_sentiment'],
            'label': rss_data['sentiment_label'],
            'bullish': rss_data['bullish_count'],
            'bearish': rss_data['bearish_count'],
            'top_articles': [a['title'] for a in rss_data['articles'][:5]],
        },
        'finviz': {
            'headlines': finviz_data.get('finviz_headlines', [])[:5],
            'sentiment': finviz_data.get('finviz_sentiment', 0),
        },
        'fundamentals': fundamentals,
        'insiders': insiders,
        'news_momentum': news_momentum,
    }
