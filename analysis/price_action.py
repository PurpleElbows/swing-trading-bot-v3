"""
Price Action — Candlestick Patterns & Market Structure
======================================================
Detects:
  - 10+ candlestick reversal/continuation patterns
  - Market structure: swing highs/lows, HH/HL, LH/LL
  - Fair value gaps (FVGs)
  - Order flow imbalance (buyer/seller pressure)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import SWING_LOOKBACK

# ============================================================================
# 1. CANDLESTICK PATTERN DETECTION
# ============================================================================

def _candle_properties(df: pd.DataFrame, offset: int = 0) -> Dict:
    """Extract properties of a single candle at the given offset from the end."""
    i = -1 - offset
    open_ = float(df['Open'].iloc[i])
    close_ = float(df['Close'].iloc[i])
    high_ = float(df['High'].iloc[i])
    low_ = float(df['Low'].iloc[i])

    body = close_ - open_                  # positive = bullish, negative = bearish
    body_abs = abs(body)
    upper_wick = high_ - max(open_, close_)
    lower_wick = min(open_, close_) - low_
    total_range = high_ - low_

    return {
        'open': open_, 'close': close_, 'high': high_, 'low': low_,
        'body': body, 'body_abs': body_abs,
        'upper_wick': upper_wick, 'lower_wick': lower_wick,
        'total_range': total_range,
        'is_bullish': body > 0,
        'is_bearish': body < 0,
    }


def detect_candle_patterns(df: pd.DataFrame) -> Dict:
    """
    Detect candlestick patterns from recent price action.
    Returns dict with pattern name(s), direction, and strength.
    """
    if df.empty or len(df) < 4:
        return {'patterns': [], 'primary': 'None', 'direction': 'Neutral', 'strength': 0}

    try:
        c0 = _candle_properties(df, 0)  # current (latest complete)
        c1 = _candle_properties(df, 1)  # previous
        c2 = _candle_properties(df, 2)  # 2 back
        c3 = _candle_properties(df, 3)  # 3 back

        patterns: List[Dict] = []

        # --- Doji (indecision) ---
        if c0['body_abs'] < c0['total_range'] * 0.1:
            patterns.append({'name': 'Doji', 'direction': 'Neutral', 'strength': 0.3})

        # --- Hammer (bullish reversal at bottom) ---
        if (c0['is_bullish'] and
            c0['lower_wick'] > c0['body_abs'] * 2 and
            c0['upper_wick'] < c0['body_abs'] * 0.5):
            patterns.append({'name': 'Hammer', 'direction': 'Bullish', 'strength': 0.6})

        # --- Shooting Star (bearish reversal at top) ---
        if (c0['is_bearish'] and
            c0['upper_wick'] > c0['body_abs'] * 2 and
            c0['lower_wick'] < c0['body_abs'] * 0.5):
            patterns.append({'name': 'Shooting Star', 'direction': 'Bearish', 'strength': 0.6})

        # --- Bullish Engulfing ---
        if (c1['is_bearish'] and c0['is_bullish'] and
            c0['close'] > c1['open'] and c0['open'] < c1['close']):
            patterns.append({'name': 'Bullish Engulfing', 'direction': 'Bullish', 'strength': 0.7})

        # --- Bearish Engulfing ---
        if (c1['is_bullish'] and c0['is_bearish'] and
            c0['close'] < c1['open'] and c0['open'] > c1['close']):
            patterns.append({'name': 'Bearish Engulfing', 'direction': 'Bearish', 'strength': 0.7})

        # --- Morning Star (3-candle bullish reversal) ---
        if (c2['is_bearish'] and
            abs(c1['body_abs']) < abs(c2['body_abs']) * 0.5 and
            c0['is_bullish'] and c0['close'] > (c2['open'] + c2['close']) / 2):
            patterns.append({'name': 'Morning Star', 'direction': 'Bullish', 'strength': 0.8})

        # --- Evening Star (3-candle bearish reversal) ---
        if (c2['is_bullish'] and
            abs(c1['body_abs']) < abs(c2['body_abs']) * 0.5 and
            c0['is_bearish'] and c0['close'] < (c2['open'] + c2['close']) / 2):
            patterns.append({'name': 'Evening Star', 'direction': 'Bearish', 'strength': 0.8})

        # --- Three White Soldiers (strong bullish continuation) ---
        if (c2['is_bullish'] and c1['is_bullish'] and c0['is_bullish'] and
            c0['close'] > c1['close'] > c2['close'] and
            c0['open'] < c0['close'] and c1['open'] < c1['close'] and c2['open'] < c2['close']):
            patterns.append({'name': 'Three White Soldiers', 'direction': 'Bullish', 'strength': 0.85})

        # --- Three Black Crows (strong bearish continuation) ---
        if (c2['is_bearish'] and c1['is_bearish'] and c0['is_bearish'] and
            c0['close'] < c1['close'] < c2['close']):
            patterns.append({'name': 'Three Black Crows', 'direction': 'Bearish', 'strength': 0.85})

        # --- Piercing Line (bullish reversal) ---
        if (c1['is_bearish'] and c0['is_bullish'] and
            c0['open'] < c1['close'] and
            c0['close'] > (c1['open'] + c1['close']) / 2 and
            c0['close'] < c1['open']):
            patterns.append({'name': 'Piercing Line', 'direction': 'Bullish', 'strength': 0.65})

        # --- Dark Cloud Cover (bearish reversal) ---
        if (c1['is_bullish'] and c0['is_bearish'] and
            c0['open'] > c1['close'] and
            c0['close'] < (c1['open'] + c1['close']) / 2 and
            c0['close'] > c1['open']):
            patterns.append({'name': 'Dark Cloud Cover', 'direction': 'Bearish', 'strength': 0.65})

        # --- Marubozu (strong momentum, no wicks) ---
        if c0['body_abs'] > c0['total_range'] * 0.9:
            direction = 'Bullish' if c0['is_bullish'] else 'Bearish'
            patterns.append({'name': f"{direction} Marubozu", 'direction': direction, 'strength': 0.75})

        # Determine primary pattern (highest strength)
        if patterns:
            primary = max(patterns, key=lambda p: p['strength'])
        else:
            primary = {'name': 'None', 'direction': 'Neutral', 'strength': 0}

        return {
            'patterns': [p['name'] for p in patterns],
            'primary': primary['name'],
            'direction': primary['direction'],
            'strength': primary['strength'],
            'pattern_count': len(patterns),
        }

    except Exception as e:
        logging.debug(f"Candle pattern detection failed: {e}")
        return {'patterns': [], 'primary': 'None', 'direction': 'Neutral', 'strength': 0}


# ============================================================================
# 2. MARKET STRUCTURE — SWING HIGHS & LOWS
# ============================================================================
def detect_swing_points(prices: np.ndarray, lookback: int = SWING_LOOKBACK) -> Dict:
    """
    Detect swing highs and swing lows in a price series.
    Uses a lookback window: a swing high is the highest point in its neighborhood.
    """
    n = len(prices)
    if n < lookback * 2 + 1:
        return {'swing_highs': [], 'swing_lows': [], 'structure': 'Neutral'}

    half = lookback // 2
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(half, n - half):
        window = prices[i - half:i + half + 1]
        if prices[i] == window.max():
            swing_highs.append((i, float(prices[i])))
        if prices[i] == window.min():
            swing_lows.append((i, float(prices[i])))

    # Market structure: Higher Highs / Higher Lows = Uptrend, etc.
    structure = 'Neutral'
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]

        higher_highs = last_two_highs[1][1] > last_two_highs[0][1]
        higher_lows = last_two_lows[1][1] > last_two_lows[0][1]

        if higher_highs and higher_lows:
            structure = 'Bullish (HH/HL)'
        elif not higher_highs and not higher_lows:
            structure = 'Bearish (LH/LL)'
        elif higher_highs and not higher_lows:
            structure = 'Weakening (HH/LL)'
        else:
            structure = 'Recovering (LH/HL)'

    return {
        'swing_highs': [(i, v) for i, v in swing_highs[-3:]],
        'swing_lows': [(i, v) for i, v in swing_lows[-3:]],
        'structure': structure,
        'is_bullish_structure': 'HH/HL' in structure,
        'is_bearish_structure': 'LH/LL' in structure,
    }


def analyze_market_structure(df: pd.DataFrame) -> Dict:
    """
    Full market structure analysis on a dataframe.
    """
    if df.empty or len(df) < SWING_LOOKBACK * 2:
        return {'swing_highs': [], 'swing_lows': [], 'structure': 'Insufficient data'}

    prices = df['Close'].values
    return detect_swing_points(prices)


# ============================================================================
# 3. FAIR VALUE GAPS (FVGs)
# ============================================================================
def detect_fair_value_gaps(df: pd.DataFrame, max_gaps: int = 3) -> List[Dict]:
    """
    Detect Fair Value Gaps — areas where price jumped, leaving a gap
    between candle wicks. These often act as support/resistance magnets.
    """
    if df.empty or len(df) < 4:
        return []

    gaps: List[Dict] = []

    for i in range(-4, -1):
        prev_high = float(df['High'].iloc[i])
        prev_low = float(df['Low'].iloc[i])
        curr_high = float(df['High'].iloc[i + 1])
        curr_low = float(df['Low'].iloc[i + 1])

        # Bullish FVG: current low > previous high (gap up)
        if curr_low > prev_high:
            gap_size = (curr_low - prev_high) / prev_high * 100
            gaps.append({
                'type': 'Bullish FVG',
                'top': curr_low,
                'bottom': prev_high,
                'size_pct': round(gap_size, 2),
                'index': int(i + len(df)),
            })

        # Bearish FVG: current high < previous low (gap down)
        elif curr_high < prev_low:
            gap_size = (prev_low - curr_high) / prev_low * 100
            gaps.append({
                'type': 'Bearish FVG',
                'top': prev_low,
                'bottom': curr_high,
                'size_pct': round(gap_size, 2),
                'index': int(i + len(df)),
            })

    return gaps[:max_gaps]


# ============================================================================
# 4. ORDER FLOW IMBALANCE
# ============================================================================
def analyze_order_flow(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Analyze buyer vs seller pressure over the last N candles.
    Returns imbalance score (-1 to +1) and direction.
    """
    if df.empty or len(df) < lookback:
        return {'imbalance': 0.0, 'direction': 'Neutral', 'up_candles': 0, 'down_candles': 0}

    try:
        recent = df.iloc[-lookback:]
        up_candles = int((recent['Close'] > recent['Open']).sum())
        down_candles = int((recent['Close'] < recent['Open']).sum())

        total = up_candles + down_candles
        imbalance = (up_candles - down_candles) / total if total > 0 else 0.0

        if imbalance > 0.3:
            direction = 'Bullish'
        elif imbalance < -0.3:
            direction = 'Bearish'
        else:
            direction = 'Neutral'

        return {
            'imbalance': round(imbalance, 3),
            'direction': direction,
            'up_candles': up_candles,
            'down_candles': down_candles,
        }
    except Exception as e:
        logging.debug(f"Order flow analysis failed: {e}")
        return {'imbalance': 0.0, 'direction': 'Neutral', 'up_candles': 0, 'down_candles': 0}


# ============================================================================
# 5. COMPREHENSIVE PRICE ACTION SUMMARY
# ============================================================================
def analyze_price_action(df_daily: pd.DataFrame) -> Dict:
    """
    Run all price action analyses and return a combined summary.
    """
    candles = detect_candle_patterns(df_daily)
    structure = analyze_market_structure(df_daily)
    fvgs = detect_fair_value_gaps(df_daily)
    order_flow = analyze_order_flow(df_daily)

    # Composite direction score (-1 to +1)
    direction_score = 0.0
    if candles['direction'] == 'Bullish':
        direction_score += candles['strength'] * 0.5
    elif candles['direction'] == 'Bearish':
        direction_score -= candles['strength'] * 0.5

    if structure.get('is_bullish_structure'):
        direction_score += 0.3
    elif structure.get('is_bearish_structure'):
        direction_score -= 0.3

    if order_flow['direction'] == 'Bullish' and order_flow['imbalance'] > 0.3:
        direction_score += 0.2
    elif order_flow['direction'] == 'Bearish' and order_flow['imbalance'] < -0.3:
        direction_score -= 0.2

    direction_score = max(-1.0, min(1.0, direction_score))

    if direction_score > 0.3:
        overall = 'Bullish'
    elif direction_score < -0.3:
        overall = 'Bearish'
    else:
        overall = 'Neutral'

    return {
        'candles': candles,
        'structure': structure,
        'fair_value_gaps': fvgs,
        'order_flow': order_flow,
        'direction_score': round(direction_score, 3),
        'overall_direction': overall,
    }
