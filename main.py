#!/usr/bin/env python3
"""
Swing Trading Bot v3 — Main Entry Point
========================================
Modular swing trading bot with:
  • Supply/demand pivot levels (Classic, Camarilla, Fibonacci)
  • Price action analysis (candlestick patterns, market structure, FVGs)
  • Volume analysis (VWAP, volume profile, climax detection, A/D)
  • News sentiment from scraped RSS feeds + Finviz + Yahoo fundamentals
  • Multi-timeframe confluence scoring
  • Telegram & Discord alerts

Usage:
  python main.py              # Normal mode: runs scheduler
  python main.py --test       # Run a single scan immediately
  python main.py --test-alerts # Test alert delivery only
"""
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import pytz
import schedule
import yfinance as yf

from config import (
    STOCKS,
    ALERT_CONFIG,
    RECOMMENDATION_WEIGHTS,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    DISCORD_WEBHOOK_URL,
)
from data.fetcher import (
    get_market_data,
    get_multi_timeframe_data,
    get_intraday_data,
    calculate_session_change,
    get_qqq_10d_return,
    reset_qqq_cache,
)
from analysis.supply_demand import (
    analyze_sd_confluence_pivots,
    pivot_proximity_score,
)
from analysis.price_action import analyze_price_action
from analysis.volume import analyze_volume
from analysis.technical import (
    analyze_technical,
    passes_weekly_trend_filter,
    passes_atr_expansion_filter,
    passes_qqq_filter,
)
from analysis.sentiment import analyze_sentiment
from alerts.messenger import send_alert, format_alert_message

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# Cooldown tracking
# ============================================================================
_last_alert: dict[str, datetime] = {}


def _is_on_cooldown(symbol: str) -> bool:
    """Return True if this ticker was alerted within the cooldown window."""
    if symbol not in _last_alert:
        return False
    cooldown = timedelta(days=ALERT_CONFIG['alert_cooldown_days'])
    return datetime.now(timezone.utc) - _last_alert[symbol] < cooldown


def _mark_alerted(symbol: str) -> None:
    """Record alert time for cooldown tracking."""
    _last_alert[symbol] = datetime.now(timezone.utc)


# ============================================================================
# Market timing
# ============================================================================
def _is_market_open() -> bool:
    """Check if US stock market is open (6:30 AM – 1:00 PM PT)."""
    utc = datetime.now(timezone.utc)
    la = pytz.timezone('America/Los_Angeles')
    pt_time = utc.astimezone(la)
    if pt_time.weekday() >= 5:
        return False
    hour, minute = pt_time.hour, pt_time.minute
    if hour < 6 or (hour == 6 and minute < 30):
        return False
    if hour >= 13:
        return False
    return True


def _is_closing_window() -> bool:
    """Return True if current time is in the final 30 min (12:30–1:00 PM PT)."""
    utc = datetime.now(timezone.utc)
    la = pytz.timezone('America/Los_Angeles')
    pt_time = utc.astimezone(la)
    if pt_time.weekday() >= 5:
        return False
    return pt_time.hour == 12 and pt_time.minute >= 30


def _is_friday_close() -> bool:
    """Return True if it's Friday and within the closing window."""
    utc = datetime.now(timezone.utc)
    la = pytz.timezone('America/Los_Angeles')
    pt_time = utc.astimezone(la)
    return pt_time.weekday() == 4 and _is_closing_window()


# ============================================================================
# Risk assessment
# ============================================================================
def _assess_risk(analysis: dict) -> dict:
    """Compute a risk score from the assembled analysis."""
    risk_score = 0.0
    risk_factors: list[str] = []

    vol = analysis.get('technical', {}).get('volatility', {})
    if vol.get('volatility_ratio', 1) > 1.5:
        risk_score += 0.3
        risk_factors.append('High volatility')

    sent = analysis.get('sentiment', {})
    if sent.get('score', 0.5) < 0.3:
        risk_score += 0.2
        risk_factors.append('Bearish sentiment')

    sd = analysis.get('supply_demand', {})
    if sd.get('in_resistance_zone'):
        risk_score += 0.2
        risk_factors.append('Near resistance')
    if sd.get('in_support_zone'):
        risk_score -= 0.1
        risk_factors.append('Near support')

    rsi = analysis.get('technical', {}).get('rsi', 50)
    if rsi > 70:
        risk_score += 0.15
        risk_factors.append('RSI overbought')
    elif rsi < 30:
        risk_score -= 0.1
        risk_factors.append('RSI oversold')

    risk_score = max(0.0, min(1.0, risk_score))
    risk_level = 'Low' if risk_score < 0.3 else 'Medium' if risk_score < 0.6 else 'High'

    return {
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
    }


# ============================================================================
# Recommendation engine
# ============================================================================
def _generate_recommendation(analysis: dict) -> dict:
    """
    Weighted recommendation based on all analysis components.
    """
    w = RECOMMENDATION_WEIGHTS

    # Technical score
    tech_score = analysis.get('technical', {}).get('technical_score', 0.5)

    # Sentiment score
    sent_score = analysis.get('sentiment', {}).get('score', 0.5)

    # Supply/demand score
    sd = analysis.get('supply_demand', {})
    sd_score = 0.5
    if sd.get('in_support_zone'):
        sd_score += 0.25
    if sd.get('in_resistance_zone'):
        sd_score -= 0.25
    sd_score = max(0.0, min(1.0, sd_score))

    # Volume score
    vol_score = analysis.get('volume', {}).get('volume_score', 0.5)

    # Confluence score
    confluence_score = analysis.get('sd_confluence', {}).get('confluence_score', 0)

    # Risk inverse
    risk_score = analysis.get('risk', {}).get('risk_score', 0.5)

    # Weighted aggregate
    overall = (
        tech_score * w['technical'] +
        sent_score * w['sentiment'] +
        sd_score * w['supply_demand'] +
        vol_score * w['volume'] +
        confluence_score * w['confluence'] +
        (1 - min(risk_score, 1)) * w['risk_inverse']
    )

    if overall >= 0.7:
        action = 'BUY'
        confidence = overall
    elif overall <= 0.3:
        action = 'SELL'
        confidence = 1 - overall
    else:
        action = 'HOLD'
        confidence = 0.5

    # Trend filter: reduce confidence for counter-trend trades
    trend = analysis.get('technical', {}).get('trend', 'Neutral')
    if action == 'BUY' and trend == 'Downtrend':
        confidence *= 0.7
        logging.debug(f"BUY in Downtrend → confidence reduced to {confidence:.2f}")
    elif action == 'SELL' and trend == 'Uptrend':
        confidence *= 0.7
        logging.debug(f"SELL in Uptrend → confidence reduced to {confidence:.2f}")

    return {'action': action, 'confidence': round(confidence, 3)}


# ============================================================================
# Main scan loop — per-symbol analysis
# ============================================================================
def scan_stock(symbol: str, qqq_return: float) -> dict | None:
    """
    Full analysis pipeline for a single stock.
    Returns the analysis dict if a tradable signal is found, else None.
    """
    try:
        # --- Cooldown check ---
        if _is_on_cooldown(symbol):
            logging.info(f"Skipping {symbol}: cooldown active")
            return None

        # --- Fetch all data ---
        multi_tf_data = get_multi_timeframe_data(symbol)
        df_daily = multi_tf_data.get('daily', None)
        df_weekly = multi_tf_data.get('weekly', None)
        df_intraday = get_intraday_data(symbol)

        if df_daily is None or df_daily.empty:
            logging.warning(f"No daily data for {symbol}")
            return None

        # --- Pre-signal swing filters ---
        if not passes_weekly_trend_filter(symbol, df_weekly or df_daily):
            return None

        # Stock 10-day return for QQQ filter
        try:
            stock_return = float(
                (df_daily['Close'].iloc[-1] / df_daily['Close'].iloc[-10] - 1) * 100
                if len(df_daily) >= 10 else 0.0
            )
        except Exception:
            stock_return = 0.0

        if not passes_qqq_filter(symbol, qqq_return, stock_return):
            return None

        if not passes_atr_expansion_filter(df_daily):
            return None

        # --- Price & volume thresholds ---
        price_change = calculate_session_change(df_intraday) if not df_intraday.empty else 0.0
        vol_ratio = (
            float(df_intraday['Volume'].iloc[-1] / df_intraday['volume_ma'].iloc[-1])
            if not df_intraday.empty and df_intraday['volume_ma'].iloc[-1] > 0
            else 1.0
        )

        if (abs(price_change) < ALERT_CONFIG['price_change_threshold']
                and vol_ratio < ALERT_CONFIG['volume_spike_threshold']):
            logging.info(
                f"Skipping {symbol}: price {price_change:.2f}% and RVOL {vol_ratio:.2f} below thresholds"
            )
            return None

        # === Build comprehensive analysis ===
        analysis: dict = {
            'price_change_pct': round(price_change, 2),
            'volume_ratio': round(vol_ratio, 2),
            'current_volume': int(df_intraday['Volume'].iloc[-1]) if not df_intraday.empty else 0,
        }

        # 1. Technical indicators
        analysis['technical'] = analyze_technical(df_daily)

        # 2. Supply/Demand + Pivots + Confluence
        sd_full = analyze_sd_confluence_pivots(symbol, multi_tf_data)
        analysis['supply_demand'] = sd_full
        analysis['sd_confluence'] = {
            'confluence_score': sd_full['confluence_score'],
            'total_timeframes': sd_full['total_timeframes'],
            'confirmed_timeframes': sd_full['confirmed_timeframes'],
            'strong_confluence': sd_full['strong_confluence'],
        }

        # 3. Pivot proximity
        analysis['pivot_proximity'] = pivot_proximity_score(
            sd_full.get('current_price'),
            sd_full.get('pivots', {}),
        )

        # 4. Price action
        analysis['price_action'] = analyze_price_action(df_daily)

        # 5. Volume analysis (intraday + daily)
        analysis['volume'] = analyze_volume(df_intraday, df_daily)

        # 6. Sentiment (news scraping)
        analysis['sentiment'] = analyze_sentiment(symbol)

        # 7. Risk assessment
        analysis['risk'] = _assess_risk(analysis)

        # 8. Recommendation
        analysis['recommendation'] = _generate_recommendation(analysis)

        # --- Confidence gate ---
        confidence = analysis['recommendation']['confidence']
        if confidence < ALERT_CONFIG['recommendation_confidence']:
            logging.info(
                f"Skipping {symbol}: confidence {confidence:.1%} below threshold "
                f"({ALERT_CONFIG['recommendation_confidence']:.0%})"
            )
            return None

        return analysis

    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}", exc_info=True)
        return None


# ============================================================================
# Scan all stocks
# ============================================================================
def scan_all_stocks(scan_type: str = 'daily') -> None:
    """Run the full scan across all watchlist stocks."""
    logging.info(f"Starting {scan_type} swing scan across {len(STOCKS)} stocks...")

    reset_qqq_cache()
    qqq_return = get_qqq_10d_return()
    logging.info(f"QQQ 10-day return: {qqq_return:.2f}%")

    alerts_sent = 0
    for symbol in STOCKS:
        try:
            result = scan_stock(symbol, qqq_return)
            if result is None:
                continue

            # Build and send alert
            message = format_alert_message(symbol, result)
            status = send_alert(message)

            if status['telegram'] or status['discord']:
                _mark_alerted(symbol)
                alerts_sent += 1
                action = result['recommendation']['action']
                confidence = result['recommendation']['confidence']
                logging.info(
                    f"✓ Alert sent: {symbol} [{action} {confidence:.1%}] "
                    f"(TG: {status['telegram']}, DC: {status['discord']})"
                )
            else:
                logging.error(f"Failed to deliver alert for {symbol} to any channel")

        except Exception as e:
            logging.error(f"Scan error for {symbol}: {e}", exc_info=True)

    logging.info(f"{scan_type.title()} scan complete — {alerts_sent} alerts sent")


# ============================================================================
# Scheduler
# ============================================================================
def main() -> None:
    """Main scheduler loop."""
    logging.info("=" * 50)
    logging.info("Swing Trading Bot v3 Starting")
    logging.info("=" * 50)
    logging.info(f"Watchlist: {len(STOCKS)} stocks")
    logging.info(f"Telegram: {'Configured' if TELEGRAM_TOKEN else 'Disabled'}")
    logging.info(f"Discord: {'Configured' if DISCORD_WEBHOOK_URL else 'Disabled'}")
    logging.info(f"Timeframes: 4h, 1d, 1wk")
    logging.info(f"Pivot types: Classic, Camarilla, Fibonacci")
    logging.info(f"News sources: RSS feeds + Finviz scraping + Yahoo fundamentals")
    logging.info(f"Confidence threshold: {ALERT_CONFIG['recommendation_confidence']:.0%}")
    logging.info(f"Cooldown: {ALERT_CONFIG['alert_cooldown_days']} days")

    def daily_close_scan():
        """Fires at 12:30 PM PT Mon-Fri."""
        if not _is_market_open():
            return
        if ALERT_CONFIG['require_closing_window'] and not _is_closing_window():
            logging.info("Not in closing window — skipping daily scan")
            return
        scan_all_stocks('daily')

    def weekly_close_scan():
        """Fires at 12:45 PM PT Friday."""
        if not _is_friday_close():
            return
        scan_all_stocks('weekly')

    # Schedule scans
    schedule.every().monday.at("12:30").do(daily_close_scan)
    schedule.every().tuesday.at("12:30").do(daily_close_scan)
    schedule.every().wednesday.at("12:30").do(daily_close_scan)
    schedule.every().thursday.at("12:30").do(daily_close_scan)
    schedule.every().friday.at("12:30").do(daily_close_scan)
    schedule.every().friday.at("12:45").do(weekly_close_scan)

    logging.info("Scheduler active: daily scans at 12:30 PT, weekly at 12:45 PT (Fri)")

    while True:
        schedule.run_pending()
        time.sleep(30)


# ============================================================================
# CLI entry
# ============================================================================
if __name__ == '__main__':
    if '--test' in sys.argv:
        logging.info("🧪 TEST MODE: Running single scan now...")
        scan_all_stocks('test')
        logging.info("✓ Test scan completed")
        sys.exit(0)

    elif '--test-alerts' in sys.argv:
        logging.info("🧪 TEST MODE: Testing alert delivery...")
        test_msg = (
            f"<b>🤖 Swing Bot v3 — Test Alert</b>\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"If you see this, alerts are working!"
        )
        status = send_alert(test_msg)
        logging.info(f"Telegram: {'✓' if status['telegram'] else '✗ Failed'}")
        logging.info(f"Discord: {'✓' if status['discord'] else '✗ Failed'}")
        sys.exit(0)

    else:
        main()
