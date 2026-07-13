# ============================================================
#  AlphaStrike Engine — V2 STRUCTURAL (backtest-validated)
#
#  Replaces the old score-stack engine after a 365-day backtest
#  (8 symbols, 804 trades) showed:
#    - OLD engine: PF 0.93 after fees = LOSING
#    - V2 + these exits: PF 1.20 taker / 1.31 maker
#
#  Strategy: "Structural Pullback in Trend"
#  A signal only exists when ALL gates pass (binary, not scored):
#    A. 4H regime is BEAR (short only) or BULL (long only).
#       RANGING = no trades. Chop kills pullback systems.
#    B. 4H confirmation actually read from 4H data:
#       close vs EMA21 vs EMA50 alignment + MACD histogram side.
#    C. A real impulse leg (>= 3 ATR) exists in the last 40 1H bars.
#    D. Price has retraced 30-65% of that leg (value zone).
#    E. Momentum is turning back WITH the trend on the current bar.
#    F. 1H RSI in a sane band (no chasing, no knife-catching).
#
#  Stops & targets (config C — best risk-adjusted in exit sweep):
#    SL  = beyond the pullback extreme + 0.25 ATR (structure, not blind ATR)
#    TP1 = 1.5R  -> close 50%, move SL to breakeven
#    TP2 = 3.0R  -> close remaining 50%
#    TP3 = structural swing target (informational runner level)
#
#  Entries are LIMIT orders at signal price. Maker fees are a
#  third of the edge — market orders give most of it back.
#
#  Expected signal rate: ~2 per symbol per week. Far fewer than
#  the old engine. That is the point: rare + structural.
#
#  Honest expectations (from backtest, ceiling not floor):
#    Win rate ~47% | avg +0.12-0.17R/trade | max DD ~18R
#    Validate 4+ weeks on the free public track record before
#    treating this as a sellable edge.
# ============================================================

import asyncio
import json
import logging
import os
import re
import time

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

from ta.momentum   import RSIIndicator
from ta.trend      import EMAIndicator, ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume     import OnBalanceVolumeIndicator

from market_intel import MarketContext, build_market_context

logger = logging.getLogger(__name__)

# ─── Settings ─────────────────────────────────────────────
MANUAL_THRESHOLD    = 60   # informational: all valid V2 signals score >= 60
AUTO_THRESHOLD      = 60
MAX_SIGNALS         = 5
MAX_LONGS           = 3
MAX_SHORTS          = 3
MIN_24H_VOLUME_USDT = 10_000_000

# V2 structural parameters (backtested — change only with new backtest)
LOOKBACK        = 40      # 1H bars searched for the impulse leg
MIN_IMPULSE_ATR = 3.0
RETRACE_MIN     = 0.30
RETRACE_MAX     = 0.65
STOP_PAD_ATR    = 0.25
MAX_RISK_ATR    = 3.0     # reject if stop distance wider than this
TP1_R           = 1.5
TP2_R           = 3.0
COOLDOWN_HOURS  = 12      # no re-signal on same symbol within this window

BAN_FILE      = "data/ban_until.txt"
COOLDOWN_FILE = "data/last_signal.json"


# ─── Ban Handling (unchanged) ─────────────────────────────

def is_banned():
    if not os.path.exists(BAN_FILE): return False
    try:
        with open(BAN_FILE) as f: ban_ts = float(f.read().strip())
        if time.time() < ban_ts: return True
        os.remove(BAN_FILE); return False
    except Exception: return False

def get_ban_remaining_mins():
    try:
        with open(BAN_FILE) as f: ban_ts = float(f.read().strip())
        return max(0, int((ban_ts - time.time()) / 60))
    except Exception: return 0

def save_ban(ms):
    try:
        os.makedirs("data", exist_ok=True)
        ts = ms / 1000
        with open(BAN_FILE, "w") as f: f.write(str(ts))
        logger.error(f"Binance ban — {int((ts-time.time())/60)}min")
    except Exception as e: logger.error(f"save_ban: {e}")


# ─── Per-symbol Cooldown ──────────────────────────────────

def load_cooldowns():
    try:
        with open(COOLDOWN_FILE) as f: return json.load(f)
    except Exception: return {}

def save_cooldowns(d):
    try:
        os.makedirs("data", exist_ok=True)
        with open(COOLDOWN_FILE, "w") as f: json.dump(d, f)
    except Exception as e: logger.warning(f"save_cooldowns: {e}")

def on_cooldown(symbol, cooldowns):
    ts = cooldowns.get(symbol, 0)
    return (time.time() - ts) < COOLDOWN_HOURS * 3600


# ─── OHLCV (unchanged) ────────────────────────────────────

async def get_candles(exchange, symbol, tf, limit=200):
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not raw or len(raw) < 50: return None
        df = pd.DataFrame(raw, columns=["t","o","h","l","c","v"]).dropna()
        df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug(f"{symbol} {tf}: {e}")
        return None


# ─── Indicators (unchanged) ───────────────────────────────

def calc_indicators(df):
    c, h, l, v = df["c"], df["h"], df["l"], df["v"]

    df["ema8"]   = EMAIndicator(close=c, window=8).ema_indicator()
    df["ema21"]  = EMAIndicator(close=c, window=21).ema_indicator()
    df["ema50"]  = EMAIndicator(close=c, window=50).ema_indicator()
    df["ema200"] = EMAIndicator(close=c, window=200).ema_indicator()

    df["rsi"]    = RSIIndicator(close=c, window=14).rsi()
    df["rsi7"]   = RSIIndicator(close=c, window=7).rsi()

    _m           = MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]   = _m.macd()
    df["macd_s"] = _m.macd_signal()
    df["macd_h"] = _m.macd_diff()

    df["atr"]    = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    df["adx"]    = ADXIndicator(high=h, low=l, close=c, window=14).adx()

    _bb          = BollingerBands(close=c, window=20, window_dev=2)
    df["bb_up"]  = _bb.bollinger_hband()
    df["bb_mid"] = _bb.bollinger_mavg()
    df["bb_lo"]  = _bb.bollinger_lband()
    df["bb_pct"] = _bb.bollinger_pband()
    df["bb_w"]   = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"].replace(0, np.nan)

    df["obv"]    = OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    df["obv_e"]  = EMAIndicator(close=df["obv"], window=21).ema_indicator()
    df["vol_ma"] = v.rolling(20).mean()

    return df.dropna()


# ─── Market Regime (unchanged — validated in backtest) ────

def get_regime(df4h):
    last  = df4h.iloc[-1]
    prev  = df4h.iloc[-2]
    close = last["c"]
    e21   = last["ema21"]
    e50   = last["ema50"]
    e200  = last["ema200"]

    bear_signals = 0
    if close < e21:   bear_signals += 2
    if close < e50:   bear_signals += 2
    if close < e200:  bear_signals += 1
    if e21 < e50:     bear_signals += 2
    if last["macd_h"] < 0: bear_signals += 1
    if e50 < prev["ema50"]: bear_signals += 1

    bull_signals = 0
    if close > e21:   bull_signals += 2
    if close > e50:   bull_signals += 2
    if close > e200:  bull_signals += 1
    if e21 > e50:     bull_signals += 2
    if last["macd_h"] > 0: bull_signals += 1
    if e50 > prev["ema50"]: bull_signals += 1

    if bear_signals >= 5 and bear_signals > bull_signals + 2:
        return "BEAR", bear_signals
    if bull_signals >= 5 and bull_signals > bear_signals + 2:
        return "BULL", bull_signals
    return "RANGING", max(bear_signals, bull_signals)


# ─── 4H Confirmation (actually reads the 4H data) ─────────

def confirm_4h_bear(df4h):
    last = df4h.iloc[-1]
    return (last["c"] < last["ema21"] and
            last["ema21"] < last["ema50"] and
            last["macd_h"] < 0)

def confirm_4h_bull(df4h):
    last = df4h.iloc[-1]
    return (last["c"] > last["ema21"] and
            last["ema21"] > last["ema50"] and
            last["macd_h"] > 0)


# ─── Impulse Structure ────────────────────────────────────

def find_impulse_down(window):
    highs = window["h"].values
    lows  = window["l"].values
    hi_idx = int(np.argmax(highs))
    if hi_idx >= len(window) - 3:
        return None
    lo_idx = hi_idx + 1 + int(np.argmin(lows[hi_idx + 1:]))
    swing_high, swing_low = highs[hi_idx], lows[lo_idx]
    if swing_low >= swing_high:
        return None
    return swing_high, swing_low, hi_idx, lo_idx

def find_impulse_up(window):
    highs = window["h"].values
    lows  = window["l"].values
    lo_idx = int(np.argmin(lows))
    if lo_idx >= len(window) - 3:
        return None
    hi_idx = lo_idx + 1 + int(np.argmax(highs[lo_idx + 1:]))
    swing_low, swing_high = lows[lo_idx], highs[hi_idx]
    if swing_high <= swing_low:
        return None
    return swing_low, swing_high, lo_idx, hi_idx


# ─── V2 Detectors ─────────────────────────────────────────

def detect_short_v2(df1h, df4h):
    if len(df1h) < LOOKBACK + 2 or len(df4h) < 3:
        return None
    last, prev = df1h.iloc[-1], df1h.iloc[-2]
    atr = last["atr"]
    if pd.isna(atr) or atr <= 0:
        return None
    if not confirm_4h_bear(df4h):
        return None

    window = df1h.iloc[-LOOKBACK:]
    imp = find_impulse_down(window)
    if imp is None:
        return None
    swing_high, swing_low, hi_idx, lo_idx = imp
    leg = swing_high - swing_low
    if leg < MIN_IMPULSE_ATR * atr:
        return None

    close = last["c"]
    retrace = (close - swing_low) / leg
    if not (RETRACE_MIN <= retrace <= RETRACE_MAX):
        return None

    turn = (last["c"] < last["o"] and
            pd.notna(last["macd_h"]) and pd.notna(prev["macd_h"]) and
            last["macd_h"] < prev["macd_h"] and
            pd.notna(last["rsi7"]) and pd.notna(prev["rsi7"]) and
            last["rsi7"] < prev["rsi7"])
    if not turn:
        return None

    rsi = last["rsi"]
    if pd.isna(rsi) or not (38 <= rsi <= 68):
        return None

    pullback_high = window["h"].values[lo_idx:].max()
    sl = pullback_high + STOP_PAD_ATR * atr
    if sl <= close:
        return None
    risk = sl - close
    if risk > MAX_RISK_ATR * atr:
        return None

    entry = close
    tp1 = entry - risk * TP1_R
    tp2 = entry - risk * TP2_R
    tp3 = min(swing_low, entry - risk * 4.0)

    score = 60
    score += int(10 * (0.65 - abs(retrace - 0.5) * 2))
    if last["adx"] > 25: score += 10
    if pd.notna(last["vol_ma"]) and last["v"] > last["vol_ma"]: score += 5
    score = min(100, score)

    return {"direction": "SHORT", "entry": entry, "sl": sl, "tp1": tp1,
            "tp2": tp2, "tp3": tp3, "score": score, "risk": risk,
            "retrace": retrace}


def detect_long_v2(df1h, df4h):
    if len(df1h) < LOOKBACK + 2 or len(df4h) < 3:
        return None
    last, prev = df1h.iloc[-1], df1h.iloc[-2]
    atr = last["atr"]
    if pd.isna(atr) or atr <= 0:
        return None
    if not confirm_4h_bull(df4h):
        return None

    window = df1h.iloc[-LOOKBACK:]
    imp = find_impulse_up(window)
    if imp is None:
        return None
    swing_low, swing_high, lo_idx, hi_idx = imp
    leg = swing_high - swing_low
    if leg < MIN_IMPULSE_ATR * atr:
        return None

    close = last["c"]
    retrace = (swing_high - close) / leg
    if not (RETRACE_MIN <= retrace <= RETRACE_MAX):
        return None

    turn = (last["c"] > last["o"] and
            pd.notna(last["macd_h"]) and pd.notna(prev["macd_h"]) and
            last["macd_h"] > prev["macd_h"] and
            pd.notna(last["rsi7"]) and pd.notna(prev["rsi7"]) and
            last["rsi7"] > prev["rsi7"])
    if not turn:
        return None

    rsi = last["rsi"]
    if pd.isna(rsi) or not (32 <= rsi <= 62):
        return None

    pullback_low = window["l"].values[hi_idx:].min()
    sl = pullback_low - STOP_PAD_ATR * atr
    if sl >= close:
        return None
    risk = close - sl
    if risk > MAX_RISK_ATR * atr:
        return None

    entry = close
    tp1 = entry + risk * TP1_R
    tp2 = entry + risk * TP2_R
    tp3 = max(swing_high, entry + risk * 4.0)

    score = 60
    score += int(10 * (0.65 - abs(retrace - 0.5) * 2))
    if last["adx"] > 25: score += 10
    if pd.notna(last["vol_ma"]) and last["v"] > last["vol_ma"]: score += 5
    score = min(100, score)

    return {"direction": "LONG", "entry": entry, "sl": sl, "tp1": tp1,
            "tp2": tp2, "tp3": tp3, "score": score, "risk": risk,
            "retrace": retrace}


# ─── Main Analysis ────────────────────────────────────────

async def analyze_symbol(exchange, symbol, ticker, fr, ctx):
    vol = ticker.get("quoteVolume") or 0
    if vol < MIN_24H_VOLUME_USDT: return None
    if fr is not None and abs(fr) > 0.003: return None

    df1h, df4h = await asyncio.gather(
        get_candles(exchange, symbol, "1h", 300),
        get_candles(exchange, symbol, "4h", 300),
    )
    if df1h is None or df4h is None: return None

    df1h = calc_indicators(df1h)
    df4h = calc_indicators(df4h)
    if len(df1h) < LOOKBACK + 10 or len(df4h) < 10: return None

    regime, regime_score = get_regime(df4h)

    # Regime gate: BEAR -> shorts only, BULL -> longs only, RANGING -> nothing
    sig = None
    if regime == "BEAR":
        sig = detect_short_v2(df1h, df4h)
    elif regime == "BULL":
        sig = detect_long_v2(df1h, df4h)
    if sig is None:
        return None

    coin      = symbol.replace("/USDT:USDT","").replace("/USDT","")
    direction = sig["direction"]
    entry     = sig["entry"]
    sl        = sig["sl"]
    icon      = "🟢" if direction == "LONG" else "🔴"
    liq       = entry * (0.92 if direction == "LONG" else 1.08)

    sl_pct = abs(entry - sl) / entry
    if sl_pct == 0: return None
    lev = min(20, max(1, round(0.02 / sl_pct)))
    rr  = round(abs(sig["tp2"] - entry) / abs(sl - entry), 2)   # = 3.0 by design

    reasons = (f"{regime}({regime_score}) | 4H-confirmed | "
               f"Impulse≥3ATR | Retrace {sig['retrace']*100:.0f}% | "
               f"Momentum-turn | LIMIT entry")

    return {
        "symbol"       : symbol.replace(":USDT",""),
        "score"        : sig["score"],
        "dir"          : f"{icon} {direction}",
        "entry"        : entry,
        "tp1"          : round(sig["tp1"], 8),
        "tp2"          : round(sig["tp2"], 8),
        "tp3"          : round(sig["tp3"], 8),
        "sl"           : round(sl, 8),
        "lev"          : lev,
        "rsi"          : round(df1h.iloc[-1]["rsi"], 1),
        "adx"          : round(df1h.iloc[-1]["adx"], 1),
        "rr"           : rr,
        "atr"          : df1h.iloc[-1]["atr"],
        "funding_rate" : round(fr * 100, 4) if fr is not None else None,
        "vol_24h_m"    : round(vol / 1_000_000, 1),
        "news"         : ctx.news_sentiment.get(coin, "NEUTRAL"),
        "news_headline": ctx.news_headlines.get(coin, ""),
        "liq_est"      : round(liq, 6),
        "sl_pct"       : round(sl_pct * 100, 2),
        "reasons"      : reasons,
    }


# ─── Dedupe (unchanged) ───────────────────────────────────

def dedupe(signals):
    final, longs, shorts = [], 0, 0
    for s in sorted(signals, key=lambda x: x["score"], reverse=True):
        il = "LONG" in s["dir"]
        if il and longs >= MAX_LONGS: continue
        if not il and shorts >= MAX_SHORTS: continue
        longs  += il
        shorts += not il
        final.append(s)
        if len(final) >= MAX_SIGNALS: break
    return final


# ─── Main Scanner ─────────────────────────────────────────

async def get_top_signals():
    if is_banned():
        logger.warning(f"Ban active — {get_ban_remaining_mins()}min")
        return [], MarketContext()

    exchange = ccxt.binance({"options":{"defaultType":"future"},"enableRateLimit":True})

    try:
        markets = await exchange.load_markets()
        futures = [s for s in markets if s.endswith("/USDT:USDT")]

        try:
            all_tickers = await exchange.fetch_tickers()
            tickers = {k: v for k, v in all_tickers.items() if k in futures}
        except Exception as e:
            if "418" in str(e):
                m = re.search(r"banned until (\d+)", str(e))
                save_ban(int(m.group(1)) if m else int((time.time()+3600)*1000))
            tickers = {}

        liquid = sorted(
            [s for s in tickers if (tickers[s].get("quoteVolume") or 0) >= MIN_24H_VOLUME_USDT],
            key=lambda s: tickers[s].get("quoteVolume") or 0,
            reverse=True
        )[:35]

        logger.info(f"Scanning {len(liquid)} pairs (V2 structural engine)")

        fr_map = {}
        try:
            fd = await exchange.fetch_funding_rates(liquid)
            for sym, d in fd.items(): fr_map[sym] = d.get("fundingRate")
        except Exception as e: logger.warning(f"FR: {e}")

        ctx = await build_market_context(exchange, liquid, os.getenv("CRYPTOPANIC_TOKEN",""))
        logger.info(f"BTC ${ctx.btc_price:,.0f} | 4H:{ctx.btc_trend_4h} | F&G:{ctx.fear_greed} | L/S:{ctx.ls_ratio:.2f}")

        cooldowns = load_cooldowns()
        raw = []
        for sym in liquid:
            try:
                if on_cooldown(sym, cooldowns):
                    continue
                r = await analyze_symbol(exchange, sym, tickers.get(sym,{}), fr_map.get(sym), ctx)
                if r:
                    raw.append(r)
                    cooldowns[sym] = time.time()
                    logger.info(f"✅ {r['symbol']:12s} {r['dir']} {r['score']}pts | {r['reasons'][:70]}")
            except Exception as e:
                logger.debug(f"❌ {sym}: {e}")
            await asyncio.sleep(0.8)

        if raw:
            save_cooldowns(cooldowns)

        final = dedupe(raw)
        logger.info(f"Scan complete. Passed:{len(raw)} | Final:{len(final)}")
        return final, ctx

    except Exception as e:
        if "418" in str(e):
            m = re.search(r"banned until (\d+)", str(e))
            save_ban(int(m.group(1)) if m else int((time.time()+7200)*1000))
        else:
            logger.error(f"get_top_signals: {e}", exc_info=True)
        return [], MarketContext()
    finally:
        await exchange.close()
