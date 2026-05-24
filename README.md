# Swing Trading Bot v3

A modular, feature-rich swing trading bot that identifies pivot levels using supply/demand analysis, price action, volume, and scraped news sentiment.

## Features

### 📊 Supply & Demand Pivot Levels
- **Classic Pivot Points** (PP, R1-R3, S1-S3)
- **Camarilla Pivots** (H3, H4, L3, L4) — tighter levels for swing trading
- **Fibonacci Pivots** (fib-based retracement/extensions)
- **S&D Zone Detection** via local extrema across 4h/daily/weekly timeframes
- **Multi-timeframe Confluence Scoring** — confirms levels across timeframes
- **Pivot Proximity Alerts** — warns when price approaches key levels

### 🕯️ Price Action Analysis
- **12 Candlestick Patterns**: Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Three White Soldiers, Three Black Crows, Piercing Line, Dark Cloud Cover, Marubozu
- **Market Structure**: Swing highs/lows, HH/HL & LH/LL trend identification
- **Fair Value Gaps (FVGs)** — gap detection that acts as support/resistance magnets
- **Order Flow Imbalance** — buyer vs seller pressure over 20 candles

### 📈 Volume Analysis
- **VWAP** with ±1σ and ±2σ standard deviation bands
- **Volume Profile** — Point of Control (POC) & High Volume Nodes (HVN)
- **Volume Climax Detection** — buying/selling climax signals (exhaustion/capitulation)
- **Accumulation/Distribution Line** — institutional flow tracking
- **Relative Volume (RVOL)** — volume relative to average
- **Volume Breakout Confirmation** — validates spike is real, not noise

### 📰 News Sentiment (Scraped from Internet)
- **RSS Feeds**: MarketWatch, Reuters, Seeking Alpha, Yahoo Finance, CNBC, Bloomberg
- **Web Scraping**: Finviz news headlines
- **Yahoo Finance Fundamentals**: Analyst ratings, institutional ownership, short interest, target prices
- **SEC Insider Transactions**: Form 4 buy/sell tracking
- **NLP Sentiment**: VADER + TextBlob dual-engine scoring
- **News Momentum**: Detects unusual article volume (potential catalysts)

### 🔔 Alert Delivery
- **Telegram** (HTML-formatted rich alerts)
- **Discord Webhooks**
- Detailed alerts with all analysis components

## Project Structure

```
swing-trading-bot-v3/
├── main.py                       # Entry point + scheduler
├── config.py                     # All configuration & thresholds
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── __init__.py
│   └── fetcher.py                # yfinance data fetching + caching
├── analysis/
│   ├── __init__.py
│   ├── supply_demand.py          # Pivot levels + S&D zones + confluence
│   ├── price_action.py           # Candlestick patterns + structure + FVGs
│   ├── volume.py                 # VWAP, volume profile, climax, A/D
│   ├── technical.py              # RSI, MACD, ATR, MAs, trend filters
│   └── sentiment.py              # News scraping + sentiment aggregation
└── alerts/
    ├── __init__.py
    └── messenger.py              # Telegram & Discord delivery
```

## Setup

### 1. Install Python 3.10+

### 2. Clone & Install Dependencies

```bash
git clone <repo-url>
cd swing-trading-bot-v3
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 4. Run the Bot

```bash
# Normal mode (scheduled scans)
python main.py

# Test mode (run one scan immediately)
python main.py --test

# Test alert delivery only
python main.py --test-alerts
```

## Strategy Configuration

All thresholds are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_change_threshold` | 3.0% | Min % move to trigger scan |
| `volume_spike_threshold` | 1.5x | RVOL multiplier to confirm |
| `rsi_overbought` | 70 | Overbought threshold |
| `rsi_oversold` | 30 | Oversold threshold |
| `recommendation_confidence` | 0.70 | Min confidence for alert |
| `alert_cooldown_days` | 3 | Days before re-alerting same symbol |
| `qqq_outperformance_min_pct` | 2.0% | Min outperformance vs QQQ |
| `atr_expansion_factor` | 1.1x | Range must exceed N × avg ATR |
| `sd_zone_proximity_pct` | 2.0% | Zone proximity threshold |

## Watchlist

`AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, QCOM, AVGO, CSCO, ORCL, CRM, ADBE, PYPL, NFLX, V, MA, QQQ`

## Scheduling

- **Daily scan**: Mon-Fri at 12:30 PM PT (final 30 min of session)
- **Weekly scan**: Friday at 12:45 PM PT (after daily scan)
- **Closing window gate**: Alerts only fire in the last 30 min of market hours

## Recommendation Weights

| Component | Weight |
|-----------|--------|
| Supply/Demand | 25% |
| Technical (RSI, MACD) | 20% |
| Sentiment (News + Fundamentals) | 15% |
| Volume (VWAP, Profile, A/D) | 15% |
| Multi-TF Confluence | 15% |
| Risk Inverse | 10% |

## Disclaimer

This is an educational/research tool, NOT financial advice. Always validate signals independently and use proper risk management. Past performance does not guarantee future results.
