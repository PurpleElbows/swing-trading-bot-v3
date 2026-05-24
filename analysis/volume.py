"""
Volume Analysis — VWAP, Volume Profile, Climax Detection
=========================================================
Provides:
  - VWAP (Volume-Weighted Average Price) with standard deviation bands
  - Volume Profile (price-level volume concentration)
  - Volume Climax detection (exhaustion / capitulation signals)
  - Accumulation / Distribution (A/D) line
  - RVOL (Relative Volume) — volume relative to average
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    VOLUME_PROFILE_BINS,
    VOLUME_CLIMAX_MULTIPLIER,
    ACCUM_DIST_LOOKBACK,
    VWAP_RESET,
)

# ============================================================================
# 1. VWAP with Standard Deviation Bands
# ============================================================================
def calculate_vwap(df: pd.DataFrame, reset: str = 'daily') -> Dict:
    """
    Calculate VWAP with ±1σ and ±2σ bands.
    VWAP = Σ(Price × Volume) / Σ(Volume) within the reset period.

    reset: 'daily' resets each session, 'weekly' resets each week.
    """
    if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        return {'vwap': None, 'upper_1': None, 'lower_1': None, 'upper_2': None, 'lower_2': None}

    try:
        sdf = df.copy()

        # Determine session grouping
        if getattr(sdf.index, 'tz', None) is not None:
            sdf['session'] = pd.Series(
                sdf.index.tz_convert('America/New_York').date, index=sdf.index
            )
        else:
            sdf['session'] = pd.Series(sdf.index.date, index=sdf.index)

        if reset == 'weekly':
            if getattr(sdf.index, 'tz', None) is not None:
                sdf['session'] = pd.Series(
                    sdf.index.tz_convert('America/New_York').isocalendar().week, index=sdf.index
                )
            else:
                sdf['session'] = pd.Series(sdf.index.isocalendar().week, index=sdf.index)

        # Typical price
        sdf['typical_price'] = (sdf['High'] + sdf['Low'] + sdf['Close']) / 3
        sdf['tp_x_vol'] = sdf['typical_price'] * sdf['Volume']

        # Compute VWAP per session
        sdf['cum_tp_vol'] = sdf.groupby('session')['tp_x_vol'].cumsum()
        sdf['cum_vol'] = sdf.groupby('session')['Volume'].cumsum()
        sdf['vwap'] = sdf['cum_tp_vol'] / sdf['cum_vol']

        # Standard deviation of price around VWAP
        sdf['price_dev'] = sdf['typical_price'] - sdf['vwap']
        sdf['var'] = (sdf['price_dev'] ** 2 * sdf['Volume']).groupby(sdf['session']).cumsum() / sdf['cum_vol']
        sdf['vwap_std'] = np.sqrt(sdf['var'].clip(lower=0))

        latest = sdf.iloc[-1]
        vwap_val = float(latest['vwap'])
        std_val = float(latest['vwap_std'])

        return {
            'vwap': round(vwap_val, 2),
            'upper_1': round(vwap_val + std_val, 2),
            'lower_1': round(vwap_val - std_val, 2),
            'upper_2': round(vwap_val + 2 * std_val, 2),
            'lower_2': round(vwap_val - 2 * std_val, 2),
            'std': round(std_val, 2),
            'price_vs_vwap_pct': round((float(latest['Close']) - vwap_val) / vwap_val * 100, 2),
            'above_vwap': float(latest['Close']) > vwap_val,
        }
    except Exception as e:
        logging.debug(f"VWAP calculation failed: {e}")
        return {'vwap': None, 'upper_1': None, 'lower_1': None, 'upper_2': None, 'lower_2': None}


# ============================================================================
# 2. VOLUME PROFILE
# ============================================================================
def analyze_volume_profile(df: pd.DataFrame, bins: int = VOLUME_PROFILE_BINS) -> Dict:
    """
    Build a volume profile histogram showing where volume concentrated by price.
    Identifies high-volume nodes (HVNs) and low-volume nodes (LVNs).
    """
    if df.empty or len(df) < 10:
        return {}

    try:
        price_min = float(df['Low'].min())
        price_max = float(df['High'].max())
        price_bins = np.linspace(price_min, price_max, bins)

        df_copy = df.copy()
        df_copy['price_bin'] = pd.cut(df_copy['Close'], price_bins)
        vol_profile = df_copy.groupby('price_bin', observed=False)['Volume'].sum()

        # High Volume Nodes (top 20% by volume)
        threshold_80 = vol_profile.quantile(0.8)
        high_vol_bins = vol_profile[vol_profile > threshold_80]

        # Point of Control (POC) — price level with most volume
        if not vol_profile.empty:
            poc_bin = vol_profile.idxmax()
            poc = float(poc_bin.mid) if hasattr(poc_bin, 'mid') else float(vol_profile.index[0].mid)
        else:
            poc = None

        current_price = float(df['Close'].iloc[-1])
        vol_at_price = float(vol_profile.sum()) if not vol_profile.empty else 0.0

        return {
            'poc': round(poc, 2) if poc else None,
            'high_volume_nodes': len(high_vol_bins),
            'total_volume': vol_at_price,
            'avg_volume': float(df['Volume'].mean()),
            'profile_bins': bins,
            'price_near_hvn': poc is not None and abs(current_price - poc) / current_price < 0.02,
        }
    except Exception as e:
        logging.debug(f"Volume profile analysis failed: {e}")
        return {}


# ============================================================================
# 3. VOLUME CLIMAX DETECTION
# ============================================================================
def detect_volume_climax(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect volume climaxes — extreme volume spikes that often signal
    exhaustion (end of move) or capitulation.
    """
    if df.empty or len(df) < lookback:
        return {'is_climax': False, 'type': 'None', 'volume_ratio': 1.0}

    try:
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-lookback:-1].mean())
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        is_climax = vol_ratio >= VOLUME_CLIMAX_MULTIPLIER

        # Determine type: buying climax (price up) or selling climax (price down)
        price_change = float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2])
        climax_type = 'None'
        if is_climax:
            if price_change > 0:
                climax_type = 'Buying Climax'     # potential exhaustion top
            else:
                climax_type = 'Selling Climax'    # potential capitulation bottom

        return {
            'is_climax': is_climax,
            'type': climax_type,
            'volume_ratio': round(vol_ratio, 2),
            'current_volume': int(current_vol),
            'average_volume': round(avg_vol, 1),
        }
    except Exception as e:
        logging.debug(f"Volume climax detection failed: {e}")
        return {'is_climax': False, 'type': 'None', 'volume_ratio': 1.0}


# ============================================================================
# 4. ACCUMULATION / DISTRIBUTION LINE
# ============================================================================
def calculate_ad_line(df: pd.DataFrame) -> Dict:
    """
    Calculate the Accumulation/Distribution line and its trend.
    A/D rising = accumulation (bullish), falling = distribution (bearish).
    """
    if df.empty or len(df) < ACCUM_DIST_LOOKBACK:
        return {'ad_trend': 'Neutral', 'ad_momentum': 0.0}

    try:
        # Money Flow Multiplier
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # handle zero-range candles

        # Money Flow Volume
        mfv = mfm * volume

        # A/D Line (cumulative)
        ad_line = mfv.cumsum()

        # Trend: compare recent A/D to earlier A/D
        recent_ad = ad_line.iloc[-ACCUM_DIST_LOOKBACK:].mean()
        earlier_ad = ad_line.iloc[-ACCUM_DIST_LOOKBACK * 2:-ACCUM_DIST_LOOKBACK].mean()

        if earlier_ad > 0:
            momentum = (recent_ad - earlier_ad) / earlier_ad
        else:
            momentum = 0.0

        if momentum > 0.1:
            trend = 'Accumulation'
        elif momentum < -0.1:
            trend = 'Distribution'
        else:
            trend = 'Neutral'

        return {
            'ad_trend': trend,
            'ad_momentum': round(float(momentum), 3),
            'ad_value': round(float(ad_line.iloc[-1]), 1),
        }
    except Exception as e:
        logging.debug(f"A/D line calculation failed: {e}")
        return {'ad_trend': 'Neutral', 'ad_momentum': 0.0}


# ============================================================================
# 5. RELATIVE VOLUME (RVOL)
# ============================================================================
def calculate_rvol(df: pd.DataFrame, lookback: int = 10) -> float:
    """
    Relative Volume = current volume / N-day average volume.
    """
    if df.empty or len(df) < lookback:
        return 1.0
    try:
        current = float(df['Volume'].iloc[-1])
        avg = float(df['Volume'].iloc[-lookback:-1].mean())
        return round(current / avg, 2) if avg > 0 else 1.0
    except Exception:
        return 1.0


# ============================================================================
# 6. VOLUME BREAKOUT CONFIRMATION
# ============================================================================
def is_volume_breakout(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Confirm that a volume spike is a real breakout, not noise.
    Requires: volume ≥ 1.5x average + directional price agreement + price move > 0.5%.
    """
    if df.empty or len(df) < lookback + 3:
        return {'is_breakout': False, 'confidence': 0.0}

    try:
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-lookback:].mean())
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        price_dir = float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2])
        confirmations = sum(
            1 for i in range(1, 4)
            if float(df['Close'].iloc[-i]) > float(df['Close'].iloc[-i - 1])
        )
        direction_up = price_dir  # True if last candle closed higher

        price_move = abs(
            (float(df['Close'].iloc[-1]) - float(df['Close'].iloc[-2]))
            / float(df['Close'].iloc[-2])
        )

        is_breakout = (
            vol_ratio >= 1.5 and
            confirmations >= (2 if direction_up else 0) and
            price_move > 0.005
        )

        return {
            'is_breakout': is_breakout,
            'volume_ratio': round(vol_ratio, 2),
            'direction_confirmations': confirmations,
            'price_move_pct': round(price_move * 100, 2),
        }
    except Exception as e:
        logging.debug(f"Volume breakout check failed: {e}")
        return {'is_breakout': False, 'confidence': 0.0}


# ============================================================================
# 7. COMPREHENSIVE VOLUME SUMMARY
# ============================================================================
def analyze_volume(df_intraday: pd.DataFrame, df_daily: pd.DataFrame) -> Dict:
    """
    Run all volume analyses and return a combined summary.
    df_intraday: 5-min bars for VWAP, climax detection
    df_daily: daily bars for volume profile, A/D, RVOL
    """
    vwap = calculate_vwap(df_intraday, reset=VWAP_RESET) if not df_intraday.empty else {}
    vol_profile = analyze_volume_profile(df_daily) if not df_daily.empty else {}
    climax = detect_volume_climax(df_intraday) if not df_intraday.empty else {}
    ad = calculate_ad_line(df_daily) if not df_daily.empty else {}
    rvol = calculate_rvol(df_daily) if not df_daily.empty else 1.0
    breakout = is_volume_breakout(df_intraday) if not df_intraday.empty else {}

    # Composite volume score (0 to 1, higher = more bullish volume)
    vol_score = 0.5  # neutral start
    if vwap.get('above_vwap'):
        vol_score += 0.15
    if vol_profile.get('price_near_hvn'):
        vol_score += 0.1
    if ad.get('ad_trend') == 'Accumulation':
        vol_score += 0.15
    elif ad.get('ad_trend') == 'Distribution':
        vol_score -= 0.15
    if climax.get('type') == 'Selling Climax':
        vol_score += 0.1   # capitulation can be bullish reversal
    elif climax.get('type') == 'Buying Climax':
        vol_score -= 0.1   # exhaustion can be bearish reversal
    if breakout.get('is_breakout'):
        vol_score += 0.1

    vol_score = max(0.0, min(1.0, vol_score))

    return {
        'vwap': vwap,
        'volume_profile': vol_profile,
        'climax': climax,
        'accumulation_distribution': ad,
        'rvol': rvol,
        'breakout': breakout,
        'volume_score': round(vol_score, 3),
        'volume_signal': 'Bullish' if vol_score > 0.6 else 'Bearish' if vol_score < 0.4 else 'Neutral',
    }
