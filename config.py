"""
Swing Trading Bot v3 — Configuration
=====================================
All tunable parameters, secrets, watchlists, and thresholds.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== SECRETS ====================
TELEGRAM_TOKEN      = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID    = os.getenv('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# ==================== WATCHLIST ====================
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
    'QCOM', 'AVGO', 'CSCO', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'NFLX',
    'V', 'MA', 'QQQ'
]

# ==================== TIMEFRAMES ====================
# For supply/demand and confluence — swing-relevant timeframes
TIMEFRAMES = {
    '4hour':  '4h',
    'daily':  '1d',
    'weekly': '1wk',
}

# For price action / intraday precision (used in volume & candle analysis)
INTRADAY_TF = '5m'

# ==================== ALERT CONFIG ====================
ALERT_CONFIG = {
    # Price & volume thresholds
    'price_change_threshold': 3.0,          # % move from session open to trigger
    'volume_spike_threshold': 1.5,          # RVOL multiplier to confirm
    'volume_lookback_days': 10,

    # RSI thresholds (applied to daily bars)
    'rsi_overbought': 70,
    'rsi_oversold':  30,

    # Sentiment gates
    'sentiment_threshold': 0.7,
    'volatility_threshold': 0.02,

    # Confidence gate for BUY/SELL recommendation
    'recommendation_confidence': 0.70,

    # Cooldown — suppress re-alerts on same ticker
    'alert_cooldown_days': 3,

    # Swing-specific: only alert in closing window (12:30–1:00 PM PT)
    'require_closing_window': True,

    # QQQ relative-strength filter
    'qqq_outperformance_min_pct': 2.0,

    # Weekly EMA for trend filter
    'weekly_ema_period': 10,

    # ATR expansion — today's range must exceed N * avg ATR
    'atr_expansion_factor': 1.1,

    # Supply/Demand zone proximity threshold (% distance from current price)
    'sd_zone_proximity_pct': 2.0,

    # Pivot confirmation — min timeframes agreeing for strong confluence
    'confluence_strong_threshold': 0.6,
}

# ==================== NEWS / SENTIMENT SOURCES ====================
RSS_FEEDS = {
    'marketwatch':       'https://feeds.marketwatch.com/marketwatch/topstories/',
    'reuters_business':  'https://feeds.reuters.com/reuters/businessNews',
    'seeking_alpha':     'https://feeds.seekingalpha.com/feed/market-news',
    'yahoo_finance':     'https://finance.yahoo.com/news/rssindex',
    'cnbc':              'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
    'bloomberg':         'https://feeds.bloomberg.com/markets/news.rss',
}

# Web scraping targets for additional sentiment
SCRAPE_TARGETS = {
    'finviz_news':  'https://finviz.com/news.ashx',
    'stocktwits':   'https://stocktwits.com/symbol/{symbol}',
}

# ==================== SUPPLY/DEMAND PIVOTS ====================
# Types of pivot calculations to run
PIVOT_TYPES = ['classic', 'camarilla', 'fibonacci']

# Lookback windows for S&D zone detection
SD_LOOKBACK = {
    '4hour':  50,
    'daily':  50,
    'weekly': 30,
}

# Minimum % distance between S&D levels to prevent clustering
SD_MIN_DISTANCE_PCT = 1.5

# ==================== PRICE ACTION ====================
# Candlestick patterns to detect
CANDLE_PATTERNS = [
    'doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
    'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
    'piercing_line', 'dark_cloud_cover', 'marubozu',
]

# Market structure swing points
SWING_LOOKBACK = 20  # candles to look back for swing highs/lows

# ==================== VOLUME ANALYSIS ====================
VWAP_RESET = 'daily'         # 'daily' or 'weekly'
VOLUME_PROFILE_BINS = 30     # number of price bins for volume profile
VOLUME_CLIMAX_MULTIPLIER = 2.5  # volume must be this × average to be a climax
ACCUM_DIST_LOOKBACK = 20     # candles for accumulation/distribution

# ==================== WEIGHTS (for final recommendation) ====================
RECOMMENDATION_WEIGHTS = {
    'technical':      0.20,   # price change + RSI
    'sentiment':      0.15,   # news + analyst sentiment
    'supply_demand':  0.25,   # pivot levels & S&D zones
    'volume':         0.15,   # volume profile, RVOL, VWAP
    'confluence':     0.15,   # multi-timeframe agreement
    'risk_inverse':   0.10,   # inverse of risk score
}
