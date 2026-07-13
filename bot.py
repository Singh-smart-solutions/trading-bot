import os
import json
import time
import logging
import asyncio
from datetime import datetime, timezone

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, ContextTypes
)

from engine import (
    get_top_signals,
    MANUAL_THRESHOLD,
    MAX_SIGNALS,
    is_banned,
    get_ban_remaining_mins,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

TOKEN            = os.getenv("BOT_TOKEN")
SUBSCRIBERS_FILE = "data/subscribers.json"
BALANCES_FILE    = "data/balances.json"
TRADES_FILE      = "data/trades.json"
COOLDOWN_SECS    = 120
CTX_CACHE_SECS   = 1800


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"save_json {path}: {e}")

subscribers    = set(load_json(SUBSCRIBERS_FILE, []))
balances       = load_json(BALANCES_FILE, {})
open_trades    = load_json(TRADES_FILE, {})
last_scan_time = {}
last_ctx       = None
last_ctx_time  = 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt(p):
    if not p: return "0"
    if p >= 1000:  return f"{p:,.2f}"
    if p >= 1:     return f"{p:.4f}"
    if p >= 0.01:  return f"{p:.5f}"
    return f"{p:.8f}"

def calc_pos(bal, sl_pct, lev):
    if sl_pct <= 0 or bal <= 0: return 0, 0, 0
    risk   = bal * 0.02
    pos    = risk / (sl_pct / 100)
    margin = pos / lev
    return round(risk, 2), round(pos, 2), round(margin, 2)


# ─── Market Context Builder ───────────────────────────────────────────────────

async def build_context_standalone():
    global last_ctx, last_ctx_time
    try:
        import ccxt.async_support as ccxt_lib
        from market_intel import build_market_context
        exchange = ccxt_lib.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
        try:
            await exchange.load_markets()
            ctx = await build_market_context(exchange, [], os.getenv("CRYPTOPANIC_TOKEN",""))
            if ctx:
                last_ctx      = ctx
                last_ctx_time = time.time()
        finally:
            try: await exchange.close()
            except Exception: pass
    except Exception as e:
        logger.error(f"build_context_standalone: {e}")


# ─── BTC Trend Watcher (lightweight) ─────────────────────────────────────────

async def fetch_btc_trend_lightweight():
    try:
        import aiohttp
        import pandas as pd
        from ta.trend import EMAIndicator
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get("https://fapi.binance.com/fapi/v1/klines",
                             params={"symbol":"BTCUSDT","interval":"4h","limit":60}) as r:
                if r.status != 200: return None, 0.0
                candles = await r.json()
        if not candles or len(candles) < 50: return None, 0.0
        close = pd.Series([float(c[4]) for c in candles])
        e20   = EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
        e50   = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
        price = float(close.iloc[-1])
        trend = "BULL" if e20 > e50*1.001 else "BEAR" if e20 < e50*0.999 else "NEUTRAL"
        return trend, price
    except Exception as e:
        logger.warning(f"BTC lightweight: {e}")
        return None, 0.0


# ─── Signal Formatters ────────────────────────────────────────────────────────

def fmt_signal(res, rank, balance=None):
    medal   = {0:"🥇",1:"🥈",2:"🥉"}.get(rank,"💎")
    score   = res["score"]
    fr_str  = f"{res['funding_rate']:+.4f}%" if res.get("funding_rate") is not None else "N/A"
    quality = ("🔥 ELITE" if score>=90 else "⚡ STRONG" if score>=75
               else "✅ GOOD" if score>=65 else "📊 VALID")
    sym  = res["symbol"].replace("/USDT","")
    dir_ = "LONG" if "LONG" in res["dir"] else "SHORT"

    out = (
        f"{medal} *{res['symbol']}* {res['dir']}\n"
        f"Quality : {quality} `({score}pts)`\n"
        f"Entry   : `{fmt(res['entry'])}`\n"
        f"TP1     : `{fmt(res['tp1'])}` _(close 50% → SL to entry)_\n"
        f"TP2     : `{fmt(res['tp2'])}` _(close rest)_\n"
        f"TP3     : `{fmt(res['tp3'])}` _(structure target)_\n"
        f"SL      : `{fmt(res['sl'])}` _({res['sl_pct']}% away)_\n"
        f"Leverage: `{res['lev']}x` | R:R `1:{res['rr']}`\n"
        f"RSI     : `{res['rsi']}` | ADX: `{res['adx']}` | FR: `{fr_str}`\n"
        f"Liq Zone: `~{fmt(res['liq_est'])}`\n"
    )
    if balance and balance > 0:
        risk, pos, margin = calc_pos(balance, res["sl_pct"], res["lev"])
        out += (
            f"\n💰 *Position (2% rule):*\n"
            f"Put in  : `${margin:.2f}` USDT\n"
            f"Max loss: `${risk:.2f}` if SL hits\n"
            f"If TP2  : `+${risk * res['rr']:.2f}` profit\n"
        )
    if res.get("news_headline"):
        out += f"News    : _{res['news_headline'][:65]}_\n"
    out += f"Reason  : _{res['reasons'][:90]}_\n"
    out += f"\n_Monitor: /addtrade {sym} {dir_} {fmt(res['entry'])} {fmt(res['tp2'])} {fmt(res['sl'])}_"
    out += "\n" + "─"*30 + "\n\n"
    return out


def fmt_guide(res):
    side = "BUY / LONG" if "LONG" in res["dir"] else "SELL / SHORT"
    icon = "🟢" if "LONG" in res["dir"] else "🔴"
    sym  = res["symbol"].replace("/USDT","")
    dir_ = "LONG" if "LONG" in res["dir"] else "SHORT"
    return (
        f"📖 *HOW TO ENTER: {res['symbol']}*\n\n"
        f"1. Binance → Futures → USDT-M → `{sym}`\n"
        f"2. Leverage: `{res['lev']}x` max\n"
        f"3. {icon} `{side}` Limit at `{fmt(res['entry'])}`\n"
        f"4. SL: `{fmt(res['sl'])}` — set immediately ⚠️\n"
        f"5. TP1: `{fmt(res['tp1'])}` → close 50%\n"
        f"   TP2: `{fmt(res['tp2'])}` → close remaining 50%\n"
        f"   TP3: `{fmt(res['tp3'])}` → structure ref (optional runner)\n"
        f"6. After TP1: move SL to `{fmt(res['entry'])}` → risk-free\n\n"
        f"`/addtrade {sym} {dir_} {fmt(res['entry'])} {fmt(res['tp2'])} {fmt(res['sl'])}`"
    )


# ─── Scan & Send ──────────────────────────────────────────────────────────────

async def scan_and_send(bot, chat_id, threshold):
    global last_ctx, last_ctx_time
    try:
        results, ctx = await get_top_signals()

        if is_banned():
            mins = get_ban_remaining_mins()
            await bot.send_message(
                chat_id=chat_id,
                text=(f"⏸ *Binance API rate-limited*\n\n"
                      f"Auto-resumes in `{mins} minutes`.\n"
                      f"No action needed."),
                parse_mode="Markdown"
            )
            return

        if ctx and ctx.btc_price > 0:
            last_ctx      = ctx
            last_ctx_time = time.time()

        filtered = [s for s in results if s["score"] >= threshold]

        if not filtered:
            btc_s = ctx.btc_trend_4h if ctx else "unknown"
            macro = f"\n🚨 Macro: *{ctx.macro_event_name}*" if ctx and ctx.macro_event_today else ""
            await bot.send_message(
                chat_id=chat_id,
                text=(f"📭 *No signals passed criteria.*\n\n"
                      f"BTC is `{btc_s}`{macro}\n\n"
                      f"Scanned 25 pairs through condition checks.\n"
                      f"Nothing qualified at {threshold}%.\n\n"
                      f"_Use /top5 to see what's closest._"),
                parse_mode="Markdown"
            )
            return

        balance = balances.get(str(chat_id), 0)
        ts      = datetime.now(timezone.utc).strftime("%H:%M UTC")
        btc_ic  = "🟢" if ctx.btc_is_bullish() else ("🔴" if ctx.btc_is_bearish() else "⚪")
        header  = (f"🚀 *SIGNALS — {ts}*\n"
                   f"BTC: {btc_ic} `{ctx.btc_trend_4h}` | F&G:`{ctx.fear_greed}` | L/S:`{ctx.ls_ratio:.2f}`\n\n"
                   f"*{len(filtered)} signal(s) found*\n\n")

        try: await bot.send_message(chat_id=chat_id, text=header, parse_mode="Markdown")
        except Exception: await bot.send_message(chat_id=chat_id, text=header.replace("*","").replace("`","").replace("_",""))

        for i, res in enumerate(filtered):
            for text in [fmt_signal(res, i, balance if balance > 0 else None), fmt_guide(res)]:
                try: await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                except Exception: await bot.send_message(chat_id=chat_id, text=text.replace("*","").replace("`","").replace("_",""))
                await asyncio.sleep(0.3)

        footer = "_TP1 → close 40%, move SL to entry (risk-free). TP2 → close 40%. TP3 → let 20% run._\n_Never risk more than 2% per trade._"
        try: await bot.send_message(chat_id=chat_id, text=footer, parse_mode="Markdown")
        except Exception: await bot.send_message(chat_id=chat_id, text=footer.replace("*","").replace("`","").replace("_",""))

    except Exception as e:
        logger.error(f"scan_and_send [{chat_id}]: {e}", exc_info=True)
        await bot.send_message(chat_id=chat_id, text="⚠️ Scan error. Please try again.")


# ─── Trade Monitor ────────────────────────────────────────────────────────────

def get_trades(chat_id): return open_trades.get(str(chat_id), [])
def save_trades(chat_id, trades):
    open_trades[str(chat_id)] = trades
    save_json(TRADES_FILE, open_trades)

async def fetch_price(symbol):
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get("https://fapi.binance.com/fapi/v1/ticker/price",
                             params={"symbol": f"{symbol}USDT"}) as r:
                if r.status == 200:
                    return float((await r.json())["price"])
    except Exception: pass
    return None

async def monitor_trade(bot, chat_id, trade):
    sym   = trade["symbol"]
    dir_  = trade["direction"]
    entry = trade["entry"]
    tp1   = trade["tp1"]
    tp2   = trade["tp2"]
    sl    = trade["sl"]

    price = await fetch_price(sym)
    if not price: return trade

    alerts, updated = [], False

    if dir_ == "LONG":
        pnl = ((price - entry) / entry) * 100
        if not trade.get("tp1_hit") and price >= tp1:
            alerts.append(f"🎉 *TP1 HIT — {sym} LONG*\nPrice `{fmt(price)}` | Profit: `+{pnl:.2f}%`\n\n1. Close 40% now\n2. Move SL to `{fmt(entry)}`")
            trade["tp1_hit"] = True; updated = True
        elif trade.get("tp1_hit") and not trade.get("tp2_hit") and price >= tp2:
            alerts.append(f"🎉 *TP2 HIT — {sym} LONG*\nPrice `{fmt(price)}` | Profit: `+{pnl:.2f}%`\n\n1. Close 40% more\n2. Let 20% run to TP3")
            trade["tp2_hit"] = True; updated = True
        if price <= sl:
            alerts.append(f"🚨 *SL HIT — {sym} LONG*\nPrice `{fmt(price)}` | Loss: `{pnl:.2f}%`\nClose now if not done.")
            trade["sl_hit"] = True; updated = True
        elif ((price - sl) / entry) * 100 < 1.0 and price > sl:
            alerts.append(f"⚠️ *SL WARNING — {sym} LONG*\nPrice `{fmt(price)}` within 1% of SL `{fmt(sl)}`")
        if last_ctx and last_ctx.btc_is_bearish() and not trade.get("btc_warn"):
            alerts.append(f"⚠️ *BTC TURNED BEARISH — {sym} LONG at risk*\nP&L: `{pnl:+.2f}%` — consider tightening SL")
            trade["btc_warn"] = True; updated = True

    elif dir_ == "SHORT":
        pnl = ((entry - price) / entry) * 100
        if not trade.get("tp1_hit") and price <= tp1:
            alerts.append(f"🎉 *TP1 HIT — {sym} SHORT*\nPrice `{fmt(price)}` | Profit: `+{pnl:.2f}%`\n\n1. Close 40% now\n2. Move SL to `{fmt(entry)}`")
            trade["tp1_hit"] = True; updated = True
        elif trade.get("tp1_hit") and not trade.get("tp2_hit") and price <= tp2:
            alerts.append(f"🎉 *TP2 HIT — {sym} SHORT*\nPrice `{fmt(price)}` | Profit: `+{pnl:.2f}%`\n\n1. Close 40% more\n2. Let 20% run")
            trade["tp2_hit"] = True; updated = True
        if price >= sl:
            alerts.append(f"🚨 *SL HIT — {sym} SHORT*\nPrice `{fmt(price)}` | Loss: `{pnl:.2f}%`\nClose now if not done.")
            trade["sl_hit"] = True; updated = True
        elif ((sl - price) / entry) * 100 < 1.0 and price < sl:
            alerts.append(f"⚠️ *SL WARNING — {sym} SHORT*\nPrice `{fmt(price)}` within 1% of SL `{fmt(sl)}`")
        if last_ctx and last_ctx.btc_is_bullish() and not trade.get("btc_warn"):
            alerts.append(f"⚠️ *BTC TURNED BULLISH — {sym} SHORT at risk*\nP&L: `{pnl:+.2f}%` — consider tightening SL")
            trade["btc_warn"] = True; updated = True

    for alert in alerts:
        try: await bot.send_message(chat_id=int(chat_id), text=alert, parse_mode="Markdown")
        except Exception: await bot.send_message(chat_id=int(chat_id), text=alert.replace("*","").replace("`","").replace("_",""))
        await asyncio.sleep(0.3)

    if updated: save_trades(chat_id, get_trades(chat_id))
    return trade


# ─── Handlers ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.add(update.effective_chat.id)
    save_json(SUBSCRIBERS_FILE, list(subscribers))
    await update.message.reply_text(
        "🚀 *AlphaStrike Signal Bot*\n\n"
        "*Commands:*\n"
        "/signals — scan for signals now\n"
        "/top5 — see top 5 closest to qualifying\n"
        "/briefing — market overview\n"
        "/addtrade — register trade to monitor\n"
        "/mytrades — see open trades\n"
        "/closetrade — remove trade\n"
        "/setbalance 500 — set your balance\n"
        "/learn — trading education\n"
        "/status — bot info\n"
        "/stop — unsubscribe\n\n"
        "⚠️ _Educational only. Not financial advice._",
        parse_mode="Markdown"
    )

async def setbalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    try:
        if not context.args:
            await update.message.reply_text("Usage: /setbalance 500")
            return
        amount = float(context.args[0])
        if amount <= 0:
            await update.message.reply_text("Must be greater than 0.")
            return
        balances[chat_id] = amount
        save_json(BALANCES_FILE, balances)
        await update.message.reply_text(
            f"✅ *Balance: ${amount:,.2f} USDT*\n\nMax risk per trade: `${amount*0.02:.2f}` (2% rule)",
            parse_mode="Markdown"
        )
    except ValueError:
        await update.message.reply_text("Invalid. Example: /setbalance 500")

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    now     = asyncio.get_event_loop().time()
    if chat_id in last_scan_time and (now - last_scan_time[chat_id]) < COOLDOWN_SECS:
        wait = int(COOLDOWN_SECS - (now - last_scan_time[chat_id]))
        await update.message.reply_text(f"⏳ Wait {wait}s before scanning again.")
        return
    last_scan_time[chat_id] = now
    msg = await update.message.reply_text("🔎 Scanning top 25 pairs...")
    await scan_and_send(context.bot, chat_id, MANUAL_THRESHOLD)
    try: await msg.delete()
    except Exception: pass

async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🏆 Finding best signal...")
    try:
        results, ctx = await get_top_signals()
        if is_banned():
            await update.message.reply_text(f"⏸ Rate-limited — {get_ban_remaining_mins()}min remaining")
            return
        global last_ctx, last_ctx_time
        if ctx and ctx.btc_price > 0:
            last_ctx = ctx; last_ctx_time = time.time()
        if not results:
            await update.message.reply_text("📭 No signal qualified.\n\n_Use /top5 to see what's closest._", parse_mode="Markdown")
        else:
            balance = balances.get(str(update.effective_chat.id), 0)
            for text in ["🏆 *BEST SIGNAL*\n\n" + fmt_signal(results[0], 0, balance if balance>0 else None), fmt_guide(results[0])]:
                try: await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
                except Exception: await context.bot.send_message(chat_id=update.effective_chat.id, text=text.replace("*","").replace("`","").replace("_",""))
                await asyncio.sleep(0.3)
    except Exception as e:
        logger.error(f"/top: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error. Try again.")
    finally:
        try: await msg.delete()
        except Exception: pass

async def top5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Finding top 5 closest coins...")
    try:
        results, ctx = await get_top_signals()
        if is_banned():
            await update.message.reply_text(f"⏸ Rate-limited — {get_ban_remaining_mins()}min remaining")
            return
        global last_ctx, last_ctx_time
        if ctx and ctx.btc_price > 0:
            last_ctx = ctx; last_ctx_time = time.time()
        if not results:
            await update.message.reply_text(
                "📭 *No coins passed minimum conditions.*\n\n"
                "This means on ALL 25 pairs:\n"
                "• MACD is still bullish (can't short)\n"
                "• OR RSI is already oversold (too risky to short)\n"
                "• OR 4H has no bearish bias\n\n"
                "_Market needs to make a clearer move._",
                parse_mode="Markdown"
            )
            return

        btc_p  = f"${ctx.btc_price:,.0f}" if ctx and ctx.btc_price > 0 else "N/A"
        header = (f"🔍 *TOP {min(5,len(results))} COINS RIGHT NOW*\n"
                  f"BTC: `{btc_p}` | Threshold: `{MANUAL_THRESHOLD}%`\n\n")
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text=header, parse_mode="Markdown")
        except Exception: await context.bot.send_message(chat_id=update.effective_chat.id, text=header.replace("*","").replace("`","").replace("_",""))

        for i, res in enumerate(results[:5]):
            score = res["score"]
            gap   = MANUAL_THRESHOLD - score
            color = "✅" if gap <= 0 else "🟡" if gap <= 8 else "🟠" if gap <= 15 else "🔴"
            status = "QUALIFIES" if gap <= 0 else f"needs +{gap}pts"
            text = (
                f"{color} *{res['symbol']}* {res['dir']}\n"
                f"Score: `{score}pts` — {status}\n"
                f"RSI: `{res['rsi']}` | ADX: `{res['adx']}`\n"
                f"Reason: _{res['reasons'][:85]}_\n"
            )
            try: await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
            except Exception: await context.bot.send_message(chat_id=update.effective_chat.id, text=text.replace("*","").replace("`","").replace("_",""))
            await asyncio.sleep(0.2)

        avg_score = sum(r["score"] for r in results[:5]) / min(5, len(results))
        summary   = (f"📊 *Summary:*\nAvg score: `{avg_score:.0f}pts` | Need: `{MANUAL_THRESHOLD}pts`\n\n"
                     f"_When BTC makes a clear move, scores jump 10-15pts and signals fire._")
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text=summary, parse_mode="Markdown")
        except Exception: await context.bot.send_message(chat_id=update.effective_chat.id, text=summary.replace("*","").replace("`","").replace("_",""))

    except Exception as e:
        logger.error(f"/top5: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error. Try again.")
    finally:
        try: await msg.delete()
        except Exception: pass

async def briefing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_ctx, last_ctx_time
    msg = await update.message.reply_text("📊 Fetching market data...")
    try:
        ctx_age = time.time() - last_ctx_time
        if last_ctx is None or last_ctx.btc_price == 0 or ctx_age > CTX_CACHE_SECS:
            await build_context_standalone()
        if last_ctx is None:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                text="⚠️ Could not fetch data. Try /signals first.")
            return

        ctx  = last_ctx
        bic  = "🟢" if ctx.btc_is_bullish() else ("🔴" if ctx.btc_is_bearish() else "⚪")
        dic  = "🟢" if ctx.btc_trend_daily=="BULL" else ("🔴" if ctx.btc_trend_daily=="BEAR" else "⚪")
        fic  = "😱" if ctx.is_extreme_fear() else ("🤑" if ctx.is_extreme_greed() else "😐")
        p    = f"${ctx.btc_price:,.0f}" if ctx.btc_price > 0 else "N/A"
        chg  = ctx.btc_change_24h or 0
        bar  = "█"*(ctx.fear_greed//10) + "░"*(10-ctx.fear_greed//10)
        oi   = ctx.oi_change_pct or 0
        age  = f" _(cached {int(ctx_age/60)}min ago)_" if ctx_age > 60 else ""

        if ctx.btc_is_bearish():   verdict = "🔴 BTC bearish — SHORT signals active, LONGs blocked"
        elif ctx.btc_is_bullish() and ctx.fear_greed < 50: verdict = "🟢 BTC bullish + Fear = strong LONG zone"
        elif ctx.is_extreme_fear(): verdict = "💡 Extreme Fear — historically best LONG accumulation zone"
        elif ctx.is_extreme_greed(): verdict = "⚠️ Extreme Greed — reduce position sizes"
        else: verdict = "⚪ Neutral — follow signal conditions"

        report = (
            f"📊 *MARKET BRIEFING*\n_{ctx.fetched_at}_{age}\n\n"
            f"₿ BTC: `{p}` ({'+' if chg>=0 else ''}{chg:.1f}% 24h)\n"
            f"4H: {bic} `{ctx.btc_trend_4h}` | Daily: {dic} `{ctx.btc_trend_daily}`\n\n"
            f"F&G: `{ctx.fear_greed}/100` {fic} {ctx.fear_greed_label}\n"
            f"`{bar}`\n"
            f"BTC Dom: `{ctx.btc_dominance:.1f}%` | L/S: `{ctx.ls_ratio:.2f}`\n"
            f"OI: `{'+' if oi>=0 else ''}{oi:.1f}%`\n\n"
        )
        if ctx.macro_event_today:
            report += f"🚨 *MACRO: {ctx.macro_event_name}* ({ctx.macro_event_impact})\nReduce size today.\n\n"
        else:
            report += "✅ No major macro events today\n\n"
        report += f"*Verdict:* {verdict}"

        try: await context.bot.send_message(chat_id=update.effective_chat.id, text=report, parse_mode="Markdown")
        except Exception: await context.bot.send_message(chat_id=update.effective_chat.id, text=report.replace("*","").replace("`","").replace("_",""))

    except Exception as e:
        logger.error(f"/briefing: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ Briefing failed. Try /signals first.")
    finally:
        try: await msg.delete()
        except Exception: pass

async def addtrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if not context.args or len(context.args) < 5:
        await update.message.reply_text("Usage:\n`/addtrade ETHUSDT SHORT 1824 1742 1862`\nFields: symbol direction entry tp sl", parse_mode="Markdown")
        return
    try:
        sym   = context.args[0].upper().replace("USDT","")
        dir_  = context.args[1].upper()
        entry = float(context.args[2])
        tp    = float(context.args[3])
        sl    = float(context.args[4])
        if dir_ not in ("LONG","SHORT"):
            await update.message.reply_text("Direction must be LONG or SHORT.")
            return
        tp1 = entry + (tp-entry)*0.5 if dir_=="LONG" else entry - (entry-tp)*0.5
        trade = {"symbol":sym,"direction":dir_,"entry":entry,"tp1":tp1,"tp2":tp,"sl":sl,
                 "tp1_hit":False,"tp2_hit":False,"sl_hit":False,"btc_warn":False,
                 "added_at":datetime.now(timezone.utc).isoformat()}
        trades = get_trades(chat_id)
        if any(t["symbol"]==sym and not t.get("sl_hit") for t in trades):
            await update.message.reply_text(f"Already monitoring {sym}. Use /closetrade {sym} first.")
            return
        trades.append(trade)
        save_trades(chat_id, trades)
        sl_pct = abs(entry-sl)/entry*100
        await update.message.reply_text(
            f"✅ *{sym}/USDT {dir_} — Monitoring 24/7*\n\nEntry: `{fmt(entry)}` | SL: `{fmt(sl)}` ({sl_pct:.2f}%)\nTP2: `{fmt(tp)}`\n\n_Alerts on TP hits, SL danger, BTC trend flip._",
            parse_mode="Markdown"
        )
    except (ValueError, IndexError):
        await update.message.reply_text("Invalid format. Example:\n`/addtrade ETHUSDT SHORT 1824 1742 1862`", parse_mode="Markdown")

async def mytrades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    trades  = [t for t in get_trades(chat_id) if not t.get("sl_hit")]
    if not trades:
        await update.message.reply_text("📭 No open trades.\n\nUse /addtrade to register one.", parse_mode="Markdown")
        return
    msg = "📊 *Open Trades:*\n\n"
    for t in trades:
        icon  = "🟢" if t["direction"]=="LONG" else "🔴"
        price = await fetch_price(t["symbol"])
        if price:
            pnl = ((price-t["entry"])/t["entry"]*100) if t["direction"]=="LONG" else ((t["entry"]-price)/t["entry"]*100)
            p_str, pnl_str = f"`{fmt(price)}`", f"`{pnl:+.2f}%`"
        else:
            p_str, pnl_str = "N/A", "N/A"
        msg += (f"{icon} *{t['symbol']}/USDT {t['direction']}*\n"
                f"Entry: `{fmt(t['entry'])}` | Now: {p_str} | P&L: {pnl_str}\n"
                f"SL: `{fmt(t['sl'])}` | TP2: `{fmt(t['tp2'])}`\n"
                f"TP1: {'✅' if t.get('tp1_hit') else '⏳'} | TP2: {'✅' if t.get('tp2_hit') else '⏳'}\n"
                f"_/closetrade {t['symbol']} to remove_\n\n")
    try: await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception: await update.message.reply_text(msg.replace("*","").replace("`","").replace("_",""))

async def closetrade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("Usage: /closetrade ETHUSDT\nOr /closetrade ALL")
        return
    sym    = context.args[0].upper().replace("USDT","")
    trades = get_trades(chat_id)
    if sym == "ALL":
        save_trades(chat_id, [])
        await update.message.reply_text("✅ All trades removed.")
        return
    before = len(trades)
    trades = [t for t in trades if t["symbol"] != sym]
    if before == len(trades):
        await update.message.reply_text(f"No trade found for {sym}.")
        return
    save_trades(chat_id, trades)
    await update.message.reply_text(f"✅ *{sym}/USDT removed from monitoring.*", parse_mode="Markdown")

async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).lower() if context.args else ""
    lessons = {
        "leverage": "📚 *Leverage*\n\n3x on $100 = control $300.\nAlso 3x the losses. Bot recommends 1-5x max.",
        "sl":       "📚 *Stop Loss*\n\nSet it when you open. NEVER remove it.\nKeeps you in the game long-term.",
        "futures":  "📚 *Futures*\n\nLONG = price goes UP.\nSHORT = price goes DOWN.\nAlways use USDT-Margined on Binance.",
        "tp":       "📚 *Take Profit*\n\nTP1 = close 40%, TP2 = close 40%, TP3 = let 20% run.\nAfter TP1: move SL to entry = risk-free.",
        "position": "📚 *Position Sizing*\n\nNever risk more than 2% per trade.\n$500 = $10 max risk.\n/setbalance 500",
        "rsi":      "📚 *RSI*\n\nAbove 68 = bot rejects LONG.\nBelow 35 = bot rejects SHORT.\nBest LONG zone: RSI 32-52.",
        "monitor":  "📚 *Trade Monitor*\n\n`/addtrade ETHUSDT SHORT 1824 1742 1862`\n\nAlerts on TP hits, SL danger, BTC flip.",
    }
    if topic and topic in lessons:
        await update.message.reply_text(lessons[topic], parse_mode="Markdown")
    else:
        kb = [
            [InlineKeyboardButton("📊 Leverage", callback_data="learn_leverage"),
             InlineKeyboardButton("🛡 Stop Loss", callback_data="learn_sl")],
            [InlineKeyboardButton("📈 Futures", callback_data="learn_futures"),
             InlineKeyboardButton("🎯 Take Profit", callback_data="learn_tp")],
            [InlineKeyboardButton("💰 Position Size", callback_data="learn_position"),
             InlineKeyboardButton("📉 RSI", callback_data="learn_rsi")],
            [InlineKeyboardButton("👁 Trade Monitor", callback_data="learn_monitor")],
        ]
        await update.message.reply_text("📚 *Trading Education:*", parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(kb))

async def learn_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    topic = query.data.replace("learn_","")
    lessons = {
        "leverage": "📚 *Leverage*\n\nMultiplies position and losses equally. Bot recommends 1-5x max.",
        "sl":       "📚 *Stop Loss*\n\nSet it when you open. NEVER remove it.",
        "futures":  "📚 *Futures*\n\nLONG = UP. SHORT = DOWN. Always USDT-Margined on Binance.",
        "tp":       "📚 *Take Profit*\n\nTP1=40%, TP2=40%, TP3=20%. After TP1: move SL to entry.",
        "position": "📚 *Position Sizing*\n\nNever risk more than 2% per trade. /setbalance 500",
        "rsi":      "📚 *RSI*\n\nAbove 68 = bot rejects LONG. Below 35 = bot rejects SHORT.",
        "monitor":  "📚 *Trade Monitor*\n\n`/addtrade ETHUSDT SHORT 1824 1742 1862`\n\nAlerts on TP/SL.",
    }
    text = lessons.get(topic, "Try /learn")
    try: await context.bot.send_message(chat_id=query.message.chat_id, text=text, parse_mode="Markdown")
    except Exception: await context.bot.send_message(chat_id=query.message.chat_id, text=text.replace("*","").replace("`",""))

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers.discard(update.effective_chat.id)
    save_json(SUBSCRIBERS_FILE, list(subscribers))
    await update.message.reply_text("🔕 Unsubscribed. /start to resubscribe.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = str(update.effective_chat.id)
    bal_str   = f"${balances[chat_id]:,.2f}" if chat_id in balances else "Not set — /setbalance"
    trades    = [t for t in get_trades(chat_id) if not t.get("sl_hit")]
    btc_str   = f"${last_ctx.btc_price:,.0f} ({last_ctx.btc_trend_4h})" if last_ctx and last_ctx.btc_price > 0 else "Send /briefing"
    ban_str   = f"⏸ {get_ban_remaining_mins()}min remaining" if is_banned() else "✅ Clear"
    cache_age = int((time.time()-last_ctx_time)/60) if last_ctx_time > 0 else 0
    await update.message.reply_text(
        f"✅ *AlphaStrike Bot — Online*\n\n"
        f"Balance      : `{bal_str}`\n"
        f"Open trades  : `{len(trades)}`\n"
        f"Subscribers  : `{len(subscribers)}`\n"
        f"Engine       : `v7.0 Condition-Based`\n"
        f"Threshold    : `{MANUAL_THRESHOLD}%`\n"
        f"Auto scan    : `Disabled`\n"
        f"Trade monitor: `Every 5 min`\n"
        f"BTC watcher  : `Every 30 min`\n"
        f"Binance API  : `{ban_str}`\n"
        f"Context cache: `{cache_age}min old`\n"
        f"BTC          : `{btc_str}`",
        parse_mode="Markdown"
    )


# ─── Background Jobs ──────────────────────────────────────────────────────────

async def trade_monitor_job(context: ContextTypes.DEFAULT_TYPE):
    for chat_id, trades in list(open_trades.items()):
        active = [t for t in trades if not t.get("sl_hit") and not t.get("tp2_hit")]
        if not active: continue
        updated = []
        for trade in trades:
            if not trade.get("sl_hit") and not trade.get("tp2_hit"):
                updated.append(await monitor_trade(context.bot, chat_id, trade))
            else:
                updated.append(trade)
        open_trades[chat_id] = updated
        save_json(TRADES_FILE, open_trades)
        await asyncio.sleep(0.5)

async def btc_watcher_job(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers: return
    trend, price = await fetch_btc_trend_lightweight()
    if trend is None: return
    prev = context.job.data or "BEAR"
    logger.info(f"BTC watcher: {prev}→{trend} ${price:,.0f}")

    if trend == "BULL" and prev != "BULL":
        for cid in list(subscribers):
            try:
                await context.bot.send_message(
                    chat_id=cid,
                    text=f"🚨 *BTC 4H FLIPPED BULL*\n\nPrice: `${price:,.0f}`\n\nRun /signals NOW — LONG setups may be ready!",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"BTC alert {cid}: {e}")
            await asyncio.sleep(0.5)
    elif trend == "BEAR" and prev == "BULL":
        for cid in list(subscribers):
            try:
                await context.bot.send_message(
                    chat_id=cid,
                    text=f"⚠️ *BTC 4H FLIPPED BEAR*\n\nPrice: `${price:,.0f}`\n\nLONGs blocked. Check open positions.",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"BTC bear alert {cid}: {e}")
            await asyncio.sleep(0.5)

    context.job.data = trend


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not TOKEN:
        logger.error("BOT_TOKEN missing!")
        return

    app = ApplicationBuilder().token(TOKEN).read_timeout(30).write_timeout(30).build()

    app.add_handler(CommandHandler("start",      start))
    app.add_handler(CommandHandler("signals",    signals))
    app.add_handler(CommandHandler("top",        top))
    app.add_handler(CommandHandler("top5",       top5))
    app.add_handler(CommandHandler("briefing",   briefing))
    app.add_handler(CommandHandler("setbalance", setbalance))
    app.add_handler(CommandHandler("addtrade",   addtrade))
    app.add_handler(CommandHandler("mytrades",   mytrades))
    app.add_handler(CommandHandler("closetrade", closetrade))
    app.add_handler(CommandHandler("learn",      learn))
    app.add_handler(CommandHandler("stop",       stop_cmd))
    app.add_handler(CommandHandler("status",     status))
    app.add_handler(CallbackQueryHandler(learn_cb, pattern="^learn_"))

    app.job_queue.run_repeating(trade_monitor_job, interval=300,  first=30)
    app.job_queue.run_repeating(btc_watcher_job,   interval=1800, first=60, data="BEAR")

    logger.info(f"AlphaStrike Bot v7.0 | {len(subscribers)} subscribers | Engine: Condition-Based")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
