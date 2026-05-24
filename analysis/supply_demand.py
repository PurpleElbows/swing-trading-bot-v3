"""
Supply & Demand — Pivot Levels Detection
========================================
Identifies pivot levels using:
  1. Classic Pivot Points (PP, R1-R3, S1-S3)
  2. Camarilla Pivots (tighter levels for intraday/swing)
  3. Fibonacci Pivots (fib-based retracement levels)
  4. Supply/Demand zones via local extrema detection (multi-timeframe)
  5. Multi-timeframe confluence scoring
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    TIMEFRAMES,
    PIVOT_TYPES,
    SD_LOOKBACK,
    SD_MIN_DISTANCE_PCT,
    ALERT_CONFIG,
)

# ============================================================================
# 1. CLASSIC PIVOT POINTS
# ============================================================================
def classic_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Standard pivot points: PP = (H+L+C)/3
    Supports: S1-S3, Resistances: R1-R3
    """
    pp = (high + low + close) / 3

    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)

    return {
        'pp': round(pp, 2),
        'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2),
        's1': round(s1, 2), 's2': round(s2, 2), 's3': round(s3, 2),
    }


# ============================================================================
# 2. CAMARILLA PIVOTS
# ============================================================================
def camarilla_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Camarilla pivots — tighter levels popular with swing traders.
    Uses H-L range multiplied by coefficients.
    """
    rng = high - low
    if rng == 0:
        rng = close * 0.001  # fallback

    h4 = close + rng * 1.1 / 2
    h3 = close + rng * 1.1 / 4
    l3 = close - rng * 1.1 / 4
    l4 = close - rng * 1.1 / 2

    return {
        'h4': round(h4, 2), 'h3': round(h3, 2),
        'l3': round(l3, 2), 'l4': round(l4, 2),
    }


# ============================================================================
# 3. FIBONACCI PIVOTS
# ============================================================================
def fibonacci_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Fibonacci pivot levels based on prior period's range.
    PP = (H+L+C)/3, then fib extensions/retracements.
    """
    pp = (high + low + close) / 3
    rng = high - low

    r1 = pp + 0.382 * rng
    r2 = pp + 0.618 * rng
    r3 = pp + 1.000 * rng
    s1 = pp - 0.382 * rng
    s2 = pp - 0.618 * rng
    s3 = pp - 1.000 * rng

    return {
        'pp': round(pp, 2),
        'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2),
        's1': round(s1, 2), 's2': round(s2, 2), 's3': round(s3, 2),
    }


# ============================================================================
# 4. COMPUTE ALL PIVOT TYPES FOR A DATAFRAME
# ============================================================================
def compute_all_pivots(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute all configured pivot types from the last complete candle.
    Uses the previous candle (index -2) for pivot calculations,
    since the current candle is still forming.
    """
    if df.empty or len(df) < 2:
        return {}

    # Use previous complete candle for pivot levels
    prev = df.iloc[-2]
    high, low, close = float(prev['High']), float(prev['Low']), float(prev['Close'])

    pivots: Dict[str, Dict[str, float]] = {}
    if 'classic' in PIVOT_TYPES:
        pivots['classic'] = classic_pivots(high, low, close)
    if 'camarilla' in PIVOT_TYPES:
        pivots['camarilla'] = camarilla_pivots(high, low, close)
    if 'fibonacci' in PIVOT_TYPES:
        pivots['fibonacci'] = fibonacci_pivots(high, low, close)

    return pivots


# ============================================================================
# 5. SUPPLY/DEMAND ZONES via LOCAL EXTREMA
# ============================================================================
def _find_local_extrema(prices: np.ndarray, min_spacing_pct: float = 1.5) -> Tuple[List[float], List[float]]:
    """
    Find local minima (support) and maxima (resistance) with minimum spacing
    to prevent level clustering.
    """
    support: List[float] = []
    resistance: List[float] = []

    # Use window of 3 on each side to identify swing points
    window = 3
    n = len(prices)

    for i in range(window, n - window):
        left_slice = prices[i - window:i]
        right_slice = prices[i + 1:i + window + 1]

        # Local minimum (support)
        if np.all(prices[i] < left_slice) and np.all(prices[i] < right_slice):
            if not support or abs(prices[i] - support[-1]) / support[-1] > min_spacing_pct / 100:
                support.append(float(prices[i]))

        # Local maximum (resistance)
        if np.all(prices[i] > left_slice) and np.all(prices[i] > right_slice):
            if not resistance or abs(prices[i] - resistance[-1]) / resistance[-1] > min_spacing_pct / 100:
                resistance.append(float(prices[i]))

    # Return most recent 3 levels
    return support[-3:], resistance[-3:]


def analyze_supply_demand(df: pd.DataFrame) -> Dict:
    """
    Analyze supply/demand zones for a single timeframe.
    Uses both local extrema (price action) and the highs/lows for zone boundaries.
    """
    if df.empty or len(df) < 10:
        return {}

    # Use closing prices for extrema detection
    prices = df['Close'].values
    support_levels, resistance_levels = _find_local_extrema(
        prices,
        min_spacing_pct=SD_MIN_DISTANCE_PCT,
    )

    current_price = float(df['Close'].iloc[-1])

    # Nearest support (below price) and resistance (above price)
    supports_below = [s for s in support_levels if s < current_price]
    resistances_above = [r for r in resistance_levels if r > current_price]

    nearest_support = max(supports_below) if supports_below else None
    nearest_resistance = min(resistances_above) if resistances_above else None

    # Distance calculations
    support_dist = ((current_price - nearest_support) / current_price * 100) if nearest_support else None
    resistance_dist = ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None

    zone_proximity_threshold = ALERT_CONFIG['sd_zone_proximity_pct']

    return {
        'current_price': current_price,
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'support_distance_pct': support_dist,
        'resistance_distance_pct': resistance_dist,
        'in_support_zone': support_dist is not None and support_dist < zone_proximity_threshold,
        'in_resistance_zone': resistance_dist is not None and resistance_dist < zone_proximity_threshold,
    }


# ============================================================================
# 6. MULTI-TIMEFRAME S&D + CONFLUENCE + PIVOTS
# ============================================================================
def analyze_sd_confluence_pivots(
    symbol: str,
    multi_tf_data: Dict[str, pd.DataFrame],
) -> Dict:
    """
    Combines:
      - Supply/demand zone analysis per timeframe
      - Multi-timeframe confluence scoring
      - Pivot point levels (classic, camarilla, fibonacci)

    All data is passed in via multi_tf_data (already fetched elsewhere).
    Returns a rich dict with all S&D, pivot, and confluence info.
    """
    tf_results: Dict[str, Dict] = {}
    support_values: List[float] = []
    resistance_values: List[float] = []
    support_confirmations = 0
    resistance_confirmations = 0
    current_price: Optional[float] = None
    all_pivots: Dict[str, Dict[str, float]] = {}

    for tf_name, df in multi_tf_data.items():
        try:
            if df.empty:
                tf_results[tf_name] = {}
                continue

            # S&D analysis
            sd = analyze_supply_demand(df)
            tf_results[tf_name] = sd

            if current_price is None:
                current_price = sd.get('current_price')

            if sd.get('nearest_support') is not None:
                support_values.append(sd['nearest_support'])
            if sd.get('nearest_resistance') is not None:
                resistance_values.append(sd['nearest_resistance'])
            if sd.get('in_support_zone'):
                support_confirmations += 1
            if sd.get('in_resistance_zone'):
                resistance_confirmations += 1

            # Pivots — use the daily timeframe
            if tf_name == 'daily' and not df.empty:
                all_pivots = compute_all_pivots(df)

        except Exception as e:
            logging.error(f"S&D analysis failed for {symbol} {tf_name}: {e}")
            tf_results[tf_name] = {}

    # Aggregate across timeframes
    n = len([t for t in multi_tf_data if not multi_tf_data[t].empty])
    agg_support = float(np.median(support_values)) if support_values else None
    agg_resistance = float(np.median(resistance_values)) if resistance_values else None
    confluence_score = (
        (support_confirmations + resistance_confirmations) / (n * 2) if n > 0 else 0
    )

    return {
        'current_price': current_price,
        'nearest_support': agg_support,
        'nearest_resistance': agg_resistance,
        'support_confirmations': support_confirmations,
        'resistance_confirmations': resistance_confirmations,
        'in_support_zone': support_confirmations > 0,
        'in_resistance_zone': resistance_confirmations > 0,
        'timeframe_details': tf_results,
        'confluence_score': confluence_score,
        'total_timeframes': n,
        'confirmed_timeframes': support_confirmations + resistance_confirmations,
        'strong_confluence': confluence_score >= ALERT_CONFIG['confluence_strong_threshold'],
        'pivots': all_pivots,
    }


# ============================================================================
# 7. Pivot-level proximity scoring
# ============================================================================
def pivot_proximity_score(current_price: float, pivots: Dict[str, Dict[str, float]]) -> Dict:
    """
    Check how close current price is to each pivot level.
    Returns a score and the nearest pivot info.
    Useful for: "price is approaching R2 from classic pivots" alerts.
    """
    if not pivots or current_price is None:
        return {'nearest_pivot': None, 'nearest_type': None, 'distance_pct': None}

    all_levels: List[Tuple[str, str, float]] = []
    for pivot_type, levels in pivots.items():
        for level_name, level_value in levels.items():
            all_levels.append((pivot_type, level_name, level_value))

    if not all_levels:
        return {'nearest_pivot': None, 'nearest_type': None, 'distance_pct': None}

    # Find closest pivot level (by absolute distance %)
    closest = min(all_levels, key=lambda x: abs(x[2] - current_price) / current_price)
    pivot_type, level_name, level_value = closest
    distance_pct = (level_value - current_price) / current_price * 100

    return {
        'nearest_pivot': f"{pivot_type}.{level_name}",
        'nearest_pivot_price': level_value,
        'nearest_pivot_type': pivot_type,
        'distance_pct': round(distance_pct, 2),
        'is_near_pivot': abs(distance_pct) < 1.0,  # within 1%
        'direction': 'resistance' if distance_pct > 0 else 'support',
    }
