"""
Data Fetcher — yfinance wrapper with rate-limiting & caching.
Fetches market data once per symbol/interval and shares across all analysis modules.
"""
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from config import TIMEFRAMES, INTRADAY_TF, ALERT_CONFIG

# ---------------------------------------------------------------------------
# Rate limiter — max N calls per 60 seconds to yfinance
# ---------------------------------------------------------------------------
_last_call_times: list[float] = []


def _rate_limit(calls_per_minute: int = 60):
    """Simple sliding-window rate limiter."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove timestamps older than 60 seconds
            while _last_call_times and now - _last_call_times[0] > 60:
                _last_call_times.pop(0)
            if len(_last_call_times) >= calls_per_minute:
                wait = 60 - (now - _last_call_times[0]) + 1
                logging.debug(f"Rate limit: waiting {wait:.1f}s")
                time.sleep(wait)
            _last_call_times.append(time.time())
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
@_rate_limit(calls_per_minute=60)
def fetch_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            logging.warning(f"No data for {symbol} ({interval}/{period})")
        return df
    except Exception as e:
        logging.error(f"Fetch failed for {symbol} {interval}: {e}")
        return pd.DataFrame()


def get_market_data(symbol: str, interval: str = "1d", period: str = "60d") -> pd.DataFrame:
    """
    Fetch market data with pre-computed columns (volume MA, price change %).
    """
    df = fetch_history(symbol, interval, period)
    if df.empty:
        return df

    lookback = ALERT_CONFIG['volume_lookback_days']
    df['volume_ma'] = df['Volume'].rolling(lookback, min_periods=1).mean()
    df['price_change_pct'] = df['Close'].pct_change() * 100
    df['range'] = df['High'] - df['Low']
    return df


# ---------------------------------------------------------------------------
# Multi-timeframe cache — fetched once per scan, shared across modules
# ---------------------------------------------------------------------------
def get_multi_timeframe_data(symbol: str, period_map: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch all configured timeframes at once.
    Returns { '4hour': df, 'daily': df, 'weekly': df }
    """
    if period_map is None:
        period_map = {
            '4h': '1mo',
            '1d': '3mo',
            '1wk': '1y',
        }

    data: Dict[str, pd.DataFrame] = {}
    for tf_name, tf_interval in TIMEFRAMES.items():
        period = period_map.get(tf_interval, '3mo')
        data[tf_name] = get_market_data(symbol, interval=tf_interval, period=period)
    return data


def get_intraday_data(symbol: str) -> pd.DataFrame:
    """Fetch intraday data for volume/VWAP/order-flow analysis."""
    return get_market_data(symbol, interval=INTRADAY_TF, period='5d')


# ---------------------------------------------------------------------------
# Convenience: session-based price change
# ---------------------------------------------------------------------------
def calculate_session_change(df: pd.DataFrame) -> float:
    """
    Calculate % move from today's market open to current price.
    Works with intraday data that has timezone-aware index.
    """
    if df.empty or 'Close' not in df.columns:
        return 0.0

    try:
        sdf = df.dropna(subset=['Close']).copy()
        if sdf.empty:
            return 0.0

        # Group by NY session date
        if getattr(sdf.index, 'tz', None) is not None:
            dates = pd.Series(sdf.index.tz_convert('America/New_York').date, index=sdf.index)
        else:
            dates = pd.Series(sdf.index.date, index=sdf.index)

        latest_session = dates.iloc[-1]
        today = sdf[dates == latest_session]
        if today.empty:
            return 0.0

        session_open = today['Open'].iloc[0] if 'Open' in today.columns else today['Close'].iloc[0]
        current_price = today['Close'].iloc[-1]

        if session_open == 0:
            return 0.0
        return float(((current_price - session_open) / session_open) * 100)
    except Exception as e:
        logging.debug(f"Session change calc failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# QQQ benchmark
# ---------------------------------------------------------------------------
_qqq_return_cache: Optional[float] = None


def get_qqq_10d_return() -> float:
    """Return QQQ's 10-day return, cached per scan cycle."""
    global _qqq_return_cache
    if _qqq_return_cache is not None:
        return _qqq_return_cache
    try:
        df = yf.Ticker('QQQ').history(interval='1d', period='15d')
        if len(df) < 10:
            _qqq_return_cache = 0.0
            return 0.0
        _qqq_return_cache = float((df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100)
        return _qqq_return_cache
    except Exception as e:
        logging.error(f"QQQ return fetch failed: {e}")
        return 0.0


def reset_qqq_cache() -> None:
    """Reset QQQ cache between scan cycles."""
    global _qqq_return_cache
    _qqq_return_cache = None
