"""
Alert Delivery — Telegram & Discord
====================================
"""
import logging
from typing import Optional

import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL


def send_telegram(message: str, parse_mode: str = 'HTML') -> bool:
    """Send an alert to Telegram. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.debug("Telegram not configured — skipping")
        return False

    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        resp = requests.post(
            url,
            json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logging.error(f"Telegram send failed: HTTP {resp.status_code} {resp.text[:300]}")
            return False
        return bool(resp.json().get('ok'))
    except Exception as e:
        logging.error(f"Telegram send exception: {e}")
        return False


def send_discord(message: str) -> bool:
    """Send an alert to Discord via webhook. Returns True on success."""
    if not DISCORD_WEBHOOK_URL:
        logging.debug("Discord not configured — skipping")
        return False

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={'content': message},
            timeout=15,
        )
        if resp.status_code not in (200, 204):
            logging.error(f"Discord send failed: HTTP {resp.status_code} {resp.text[:300]}")
            return False
        return True
    except Exception as e:
        logging.error(f"Discord send exception: {e}")
        return False


def send_alert(message: str) -> Dict[str, bool]:
    """Send an alert to all configured channels. Returns success per channel."""
    tg = send_telegram(message)
    dc = send_discord(message)
    return {'telegram': tg, 'discord': dc}


def format_alert_message(symbol: str, analysis: dict) -> str:
    """
    Build a rich HTML-formatted alert message from the analysis dict.
    Compatible with Telegram (HTML parse mode) and Discord.
    """
    try:
        rec = analysis.get('recommendation', {})
        sd = analysis.get('supply_demand', {})
        pivots = sd.get('pivots', {})
        pivot_prox = analysis.get('pivot_proximity', {})
        sent = analysis.get('sentiment', {})
        tech = analysis.get('technical', {})
        pa = analysis.get('price_action', {})
        vol = analysis.get('volume', {})
        risk = analysis.get('risk', {})
        conf = analysis.get('sd_confluence', {})

        action = rec.get('action', 'N/A')
        confidence = rec.get('confidence', 0)

        # Emoji for action
        emoji = '🟢' if action == 'BUY' else '🔴' if action == 'SELL' else '🟡'

        # Format pivot levels nicely
        pivot_lines = ''
        if pivots:
            for ptype, levels in pivots.items():
                if ptype == 'classic':
                    pivot_lines += (
                        f"  <b>Classic:</b> PP={levels.get('pp','-')} | "
                        f"R1={levels.get('r1','-')} S1={levels.get('s1','-')} | "
                        f"R2={levels.get('r2','-')} S2={levels.get('s2','-')}\n"
                    )
                elif ptype == 'camarilla':
                    pivot_lines += (
                        f"  <b>Camarilla:</b> H3={levels.get('h3','-')} H4={levels.get('h4','-')} | "
                        f"L3={levels.get('l3','-')} L4={levels.get('l4','-')}\n"
                    )
                elif ptype == 'fibonacci':
                    pivot_lines += (
                        f"  <b>Fibonacci:</b> PP={levels.get('pp','-')} | "
                        f"R1={levels.get('r1','-')} S1={levels.get('s1','-')}\n"
                    )

        # Support/resistance
        support_str = f"${sd.get('nearest_support', 0):.2f}" if sd.get('nearest_support') else 'N/A'
        resistance_str = f"${sd.get('nearest_resistance', 0):.2f}" if sd.get('nearest_resistance') else 'N/A'
        current_price_str = f"${sd.get('current_price', 0):.2f}" if sd.get('current_price') else 'N/A'

        # Price action
        candles = pa.get('candles', {})
        structure = pa.get('structure', {})
        order_flow = pa.get('order_flow', {})
        fvgs = pa.get('fair_value_gaps', [])

        # Volume
        vwap = vol.get('vwap', {})
        climax = vol.get('climax', {})
        ad = vol.get('accumulation_distribution', {})

        # News sentiment
        rss = sent.get('rss_news', {})
        finviz = sent.get('finviz', {})
        insiders = sent.get('insiders', {})
        news_mom = sent.get('news_momentum', {})

        # Build message
        message = f"""{emoji} <b>{symbol} SWING ALERT</b> {emoji}

<b>━━━ RECOMMENDATION ━━━</b>
  Action: <b>{action}</b> | Confidence: <b>{confidence:.1%}</b>

<b>━━━ PRICE & VOLUME ━━━</b>
  Price: {current_price_str} | Day Change: {analysis.get('price_change_pct', 0):+.2f}%
  RVOL: {analysis.get('volume_ratio', 0):.2f}x | Vol Climax: {climax.get('type', 'None')}
  VWAP: ${vwap.get('vwap', 'N/A')} | {'Above' if vwap.get('above_vwap') else 'Below'} VWAP ({vwap.get('price_vs_vwap_pct', 0):+.2f}%)

<b>━━━ TECHNICALS ━━━</b>
  RSI: {tech.get('rsi', 50):.1f} | MACD: {tech.get('macd', {}).get('trend', 'N/A')} | Trend: {tech.get('trend', 'N/A')}
  ATR: ${tech.get('atr', 0):.2f} | Vol Regime: {tech.get('volatility', {}).get('regime', 'N/A')}
  BB: {tech.get('bollinger_bands', {}).get('price_vs_bands', 'N/A')} (%B: {tech.get('bollinger_bands', {}).get('pct_b', 'N/A')})

<b>━━━ SUPPLY & DEMAND ━━━</b>
  Support: {support_str} | Resistance: {resistance_str}
  In Zone: {'Support' if sd.get('in_support_zone') else 'Resistance' if sd.get('in_resistance_zone') else 'None'}
  Confluence: {conf.get('confluence_score', 0):.1%} ({conf.get('confirmed_timeframes', 0)}/{conf.get('total_timeframes', 0)} TFs) | Strong: {'Yes' if conf.get('strong_confluence') else 'No'}
{pivot_lines}"""

        # Add pivot proximity if available
        if pivot_prox.get('nearest_pivot'):
            message += f"  Nearest Pivot: <b>{pivot_prox['nearest_pivot']}</b> ({pivot_prox['distance_pct']:+.2f}%)\n"

        message += f"""
<b>━━━ PRICE ACTION ━━━</b>
  Pattern: {candles.get('primary', 'None')} ({pa.get('overall_direction', 'Neutral')})
  Structure: {structure.get('structure', 'N/A')}
  Order Flow: {order_flow.get('direction', 'Neutral')} ({order_flow.get('up_candles', 0)}↑ {order_flow.get('down_candles', 0)}↓)"""

        if fvgs:
            for fvg in fvgs[:2]:
                message += f"\n  FVG: {fvg['type']} ({fvg['size_pct']}%)"

        message += f"""

<b>━━━ SENTIMENT & NEWS ━━━</b>
  Overall: <b>{sent.get('sentiment', 'Neutral')}</b> ({sent.get('score', 0.5):.2f})
  News: {rss.get('article_count', 0)} articles ({rss.get('label', 'N/A')} | {rss.get('bullish', 0)}↑ {rss.get('bearish', 0)}↓)
  News Momentum: {news_mom.get('news_momentum', 'None')} | Catalyst: {'Yes' if news_mom.get('has_catalyst') else 'No'}
  Insiders: {insiders.get('insider_signal', 'N/A')}"""

        # Top news headlines
        top_articles = rss.get('top_articles', [])
        if top_articles:
            message += "\n  <b>Recent headlines:</b>"
            for i, article in enumerate(top_articles[:3]):
                message += f"\n    {i+1}. {article[:120]}"

        # Finviz headlines
        finviz_heads = finviz.get('headlines', [])
        if finviz_heads:
            message += "\n  <b>Finviz:</b>"
            for i, h in enumerate(finviz_heads[:2]):
                message += f"\n    • {h[:100]}"

        message += f"""

<b>━━━ VOLUME ANALYSIS ━━━</b>
  A/D Line: {ad.get('ad_trend', 'N/A')} | RVOL: {vol.get('rvol', 1):.2f}x
  Breakout: {'Yes' if vol.get('breakout', {}).get('is_breakout') else 'No'} | POC: ${vol.get('volume_profile', {}).get('poc', 'N/A')}

<b>━━━ RISK ━━━</b>
  Level: {risk.get('risk_level', 'N/A')} | Score: {risk.get('risk_score', 0):.2f}
  Factors: {', '.join(risk.get('risk_factors', [])[:4])}
"""

        return message.strip()

    except Exception as e:
        logging.error(f"Alert message formatting failed: {e}", exc_info=True)
        return f"🔔 ALERT: {symbol} — Error formatting alert"
