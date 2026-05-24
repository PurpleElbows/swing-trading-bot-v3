"""
Technical Indicators — RSI, MACD, ATR, MAs, Trend Detection
============================================================
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

from config import ALERT_CONFIG, TIMEFRAMES


# ============================================================================
# 1. RSI — Wilder's Smoothing (matches TradingView / charting platforms)
# ============================================================================
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing method."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_daily_rsi(prices: pd.Series, period: int = 14) -> float:
    """Return the latest daily RSI value, or 50.0 on failure."""
    try:
        rsi_series = calculate_rsi(prices, period)
        val = rsi_series.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0
    except Exception as e:
        logging.debug(f"Daily RSI failed: {e}")
        return 50.0


# ============================================================================
# 2. MACD
# ============================================================================
def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict:
    """MACD with trend classification."""
    if len(prices) < slow:
        return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'Neutral'}

    fast_ema = prices.ewm(span=fast).mean()
    slow_ema = prices.ewm(span=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line

    m = float(macd_line.iloc[-1])
    s = float(signal_line.iloc[-1])
    h = float(histogram.iloc[-1])

    if m > s and h > 0:
        trend = 'Bullish'
    elif m < s and h < 0:
        trend = 'Bearish'
    else:
        trend = 'Neutral'

    return {
        'macd': round(m, 4),
        'signal': round(s, 4),
        'histogram': round(h, 4),
        'trend': trend,
    }


# ============================================================================
# 3. ATR — Average True Range
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def get_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value."""
    try:
        atr_series = calculate_atr(df, period)
        return float(atr_series.iloc[-1])
    except Exception:
        return 0.0


# ============================================================================
# 4. TREND DETECTION (MA Cross)
# ============================================================================
def detect_trend(df: pd.DataFrame, short: int = 20, long: int = 50) -> str:
    """Detect trend using short/long MA cross."""
    if df.empty or len(df) < long:
        return 'Insufficient data'

    short_ma = df['Close'].rolling(short).mean()
    long_ma = df['Close'].rolling(long).mean()

    current_above = short_ma.iloc[-1] > long_ma.iloc[-1]
    prev_above = short_ma.iloc[-2] > long_ma.iloc[-2]

    if current_above and prev_above:
        return 'Uptrend'
    elif not current_above and not prev_above:
        return 'Downtrend'
    else:
        return 'Sideways'


# ============================================================================
# 5. BOLLINGER BANDS
# ============================================================================
def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Dict:
    """Bollinger Bands with %B and bandwidth."""
    if len(prices) < period:
        return {}

    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    # %B = (price - lower) / (upper - lower)
    band_range = upper - lower
    pct_b = (prices - lower) / band_range.replace(0, np.nan)

    current_price = float(prices.iloc[-1])
    bb_upper = float(upper.iloc[-1])
    bb_lower = float(lower.iloc[-1])
    bb_mid = float(sma.iloc[-1])

    return {
        'upper': round(bb_upper, 2),
        'middle': round(bb_mid, 2),
        'lower': round(bb_lower, 2),
        'pct_b': round(float(pct_b.iloc[-1]), 3),
        'bandwidth_pct': round((bb_upper - bb_lower) / bb_mid * 100, 2),
        'price_vs_bands': (
            'Above upper' if current_price > bb_upper
            else 'Below lower' if current_price < bb_lower
            else 'Inside bands'
        ),
    }


# ============================================================================
# 6. SWING FILTERS — Pre-signal gates
# ============================================================================
def passes_weekly_trend_filter(symbol: str, df_weekly: pd.DataFrame) -> bool:
    """
    Trend filter: weekly close must be above the N-week EMA.
    Keeps longs aligned with the larger move.
    """
    if df_weekly.empty:
        return True
    period = ALERT_CONFIG['weekly_ema_period']
    if len(df_weekly) < period:
        return True

    ema = df_weekly['Close'].ewm(span=period).mean()
    above = float(df_weekly['Close'].iloc[-1]) > float(ema.iloc[-1])
    if not above:
        logging.info(f"Skipping {symbol}: weekly close below {period}-week EMA")
    return above


def passes_atr_expansion_filter(df_daily: pd.DataFrame) -> bool:
    """
    Today's range must be expanding relative to average ATR.
    Avoids low-energy consolidation days.
    """
    if df_daily.empty or len(df_daily) < 12:
        return True

    try:
        high_low = df_daily['High'] - df_daily['Low']
        high_close = (df_daily['High'] - df_daily['Close'].shift()).abs()
        low_close = (df_daily['Low'] - df_daily['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr_avg = tr.iloc[-12:-1].mean()
        current_range = high_low.iloc[-1]
        factor = ALERT_CONFIG['atr_expansion_factor']
        expanding = current_range >= atr_avg * factor

        if not expanding:
            logging.debug(f"ATR filter: range {current_range:.2f} < {atr_avg * factor:.2f} avg")
        return expanding
    except Exception as e:
        logging.debug(f"ATR expansion filter failed: {e}")
        return True


def passes_qqq_filter(symbol: str, qqq_return: float, stock_return: float) -> bool:
    """
    Stock must outperform QQQ by the configured margin.
    """
    if symbol == 'QQQ':
        return True
    outperformance = stock_return - qqq_return
    if outperformance < ALERT_CONFIG['qqq_outperformance_min_pct']:
        logging.info(
            f"Skipping {symbol}: outperforms QQQ by {outperformance:.2f}% "
            f"(need {ALERT_CONFIG['qqq_outperformance_min_pct']}%)"
        )
        return False
    return True


# ============================================================================
# 7. VOLATILITY REGIME
# ============================================================================
def classify_volatility_regime(df_daily: pd.DataFrame) -> Dict:
    """Classify current volatility regime and compute ratio."""
    if df_daily.empty:
        return {'daily_vol': 0, 'historical_vol': 0, 'vol_ratio': 1, 'regime': 'Unknown'}

    returns = df_daily['Close'].pct_change().dropna()
    if len(returns) < 5:
        return {'daily_vol': 0, 'historical_vol': 0, 'vol_ratio': 1, 'regime': 'Unknown'}

    daily_vol = float(returns.std() * np.sqrt(252))
    historical_vol = float(returns.std() * np.sqrt(len(returns)))
    vol_ratio = daily_vol / historical_vol if historical_vol > 0 else 1.0

    if daily_vol > 0.25:
        regime = 'High Volatility'
    elif daily_vol > 0.15:
        regime = 'Moderate Volatility'
    else:
        regime = 'Low Volatility'

    return {
        'daily_volatility': round(daily_vol, 4),
        'historical_volatility': round(historical_vol, 4),
        'volatility_ratio': round(vol_ratio, 3),
        'regime': regime,
    }


# ============================================================================
# 8. COMPREHENSIVE TECHNICAL SUMMARY
# ============================================================================
def analyze_technical(df_daily: pd.DataFrame) -> Dict:
    """Run all technical indicators on daily data."""
    if df_daily.empty:
        return {}

    prices = df_daily['Close']

    rsi = get_daily_rsi(prices)
    macd = calculate_macd(prices)
    atr = get_atr(df_daily)
    trend = detect_trend(df_daily)
    bbands = calculate_bollinger_bands(prices)
    vol = classify_volatility_regime(df_daily)

    # Simple composite score
    tech_score = 0.5
    if rsi < ALERT_CONFIG['rsi_oversold']:
        tech_score += 0.2   # oversold = potential buy
    elif rsi > ALERT_CONFIG['rsi_overbought']:
        tech_score -= 0.2   # overbought = caution

    if macd.get('trend') == 'Bullish':
        tech_score += 0.1
    elif macd.get('trend') == 'Bearish':
        tech_score -= 0.1

    if trend == 'Uptrend':
        tech_score += 0.1
    elif trend == 'Downtrend':
        tech_score -= 0.1

    tech_score = max(0.0, min(1.0, tech_score))

    return {
        'rsi': round(rsi, 1),
        'macd': macd,
        'atr': round(atr, 2),
        'trend': trend,
        'bollinger_bands': bbands,
        'volatility': vol,
        'technical_score': round(tech_score, 3),
    }
