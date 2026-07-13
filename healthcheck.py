"""
AlphaStrike Bot Health Check
=============================
Self-diagnostic for the trading bot. Run this on the SAME machine/IP
where the bot runs (Render shell), because Binance bans are IP-based —
running it on your laptop tests your laptop's IP, not Render's.

WHAT IT CHECKS
--------------
1. Binance Futures public API reachability + latency
2. IP ban / rate-limit status (418 = banned, 429 = throttled)
3. Current rate-limit usage headers (how close you are to a ban)
4. Local ban file (data/ban_until.txt) status
5. All external data feeds the bot depends on:
   Fear & Greed, BTC dominance (CoinGecko), Long/Short ratio,
   Open Interest, macro calendar
6. Telegram BOT_TOKEN validity (if set in environment)

NOTE: This bot uses only PUBLIC Binance endpoints — no API keys exist
or are needed. Blocks come from IP rate-limiting, not key issues.

USAGE
-----
    python healthcheck.py

On Render: open the Shell tab for your service and run it there.
"""
import asyncio
import os
import time
from datetime import datetime, timezone

import aiohttp

TIMEOUT = aiohttp.ClientTimeout(total=10)
OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def show(status, name, detail=""):
    print(f"  {status} {name:<34} {detail}")


async def check_binance_futures():
    """Ping + exchangeInfo + rate limit headers + ban detection."""
    print("\n[1] Binance Futures API")
    url_ping = "https://fapi.binance.com/fapi/v1/ping"
    url_price = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
            t0 = time.time()
            async with s.get(url_ping) as r:
                latency = (time.time() - t0) * 1000
                if r.status == 200:
                    show(OK, "Ping", f"{latency:.0f}ms")
                elif r.status == 418:
                    body = await r.text()
                    show(FAIL, "Ping", f"HTTP 418 — IP IS BANNED. {body[:120]}")
                    return False
                elif r.status == 429:
                    show(WARN, "Ping", "HTTP 429 — rate limited (pre-ban warning). SLOW DOWN requests.")
                else:
                    show(FAIL, "Ping", f"HTTP {r.status}")
                    return False

            async with s.get(url_price) as r:
                used = r.headers.get("X-MBX-USED-WEIGHT-1M", "?")
                if r.status == 200:
                    d = await r.json()
                    show(OK, "BTC price fetch", f"${float(d['price']):,.0f}")
                    # futures limit is 2400 weight/min; warn at 50%
                    try:
                        u = int(used)
                        pct = u / 2400 * 100
                        status = OK if pct < 50 else WARN if pct < 80 else FAIL
                        show(status, "Rate-limit usage (1m weight)", f"{u}/2400 ({pct:.0f}%)")
                    except ValueError:
                        show(WARN, "Rate-limit usage", f"header: {used}")
                elif r.status == 418:
                    retry = r.headers.get("Retry-After", "?")
                    show(FAIL, "BTC price fetch", f"HTTP 418 BANNED — Retry-After: {retry}s")
                    return False
                elif r.status == 429:
                    show(WARN, "BTC price fetch", "HTTP 429 rate limited")
                else:
                    show(FAIL, "BTC price fetch", f"HTTP {r.status}")
        return True
    except asyncio.TimeoutError:
        show(FAIL, "Binance Futures", "timeout — network issue or regional block")
        return False
    except Exception as e:
        show(FAIL, "Binance Futures", f"{type(e).__name__}: {e}")
        return False


def check_ban_file():
    print("\n[2] Local ban file")
    for path in ("data/ban_until.txt", "/data/ban_until.txt"):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    ban_ts = float(f.read().strip())
                remaining = ban_ts - time.time()
                if remaining > 0:
                    until = datetime.fromtimestamp(ban_ts, tz=timezone.utc).strftime("%H:%M UTC")
                    show(WARN, path, f"BAN ACTIVE — {int(remaining/60)} min left (until {until}). "
                                     f"Bot will refuse to scan until then.")
                else:
                    show(WARN, path, "expired ban file present — bot will delete it on next scan")
            except Exception as e:
                show(WARN, path, f"unreadable ({e}) — consider deleting it")
            return
    show(OK, "ban_until.txt", "not present — no local ban recorded")


async def check_feed(name, url, params=None, validate=None):
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
            async with s.get(url, params=params) as r:
                if r.status != 200:
                    show(FAIL, name, f"HTTP {r.status}")
                    return
                data = await r.json(content_type=None)
                detail = validate(data) if validate else "reachable"
                show(OK, name, detail)
    except asyncio.TimeoutError:
        show(FAIL, name, "timeout")
    except Exception as e:
        show(FAIL, name, f"{type(e).__name__}: {e}")


async def check_data_feeds():
    print("\n[3] External data feeds")
    await check_feed(
        "Fear & Greed (alternative.me)",
        "https://api.alternative.me/fng/?limit=1",
        validate=lambda d: f"value={d['data'][0]['value']} ({d['data'][0]['value_classification']})",
    )
    await check_feed(
        "BTC dominance (CoinGecko)",
        "https://api.coingecko.com/api/v3/global",
        validate=lambda d: f"{d['data']['market_cap_percentage']['btc']:.1f}%",
    )
    await check_feed(
        "Long/Short ratio (Binance)",
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        params={"symbol": "BTCUSDT", "period": "4h", "limit": 1},
        validate=lambda d: f"L/S={float(d[0]['longShortRatio']):.2f}",
    )
    await check_feed(
        "Open Interest (Binance)",
        "https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol": "BTCUSDT", "period": "4h", "limit": 1},
        validate=lambda d: f"OI=${float(d[0]['sumOpenInterestValue'])/1e9:.2f}B",
    )
    await check_feed(
        "Macro calendar (faireconomy)",
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        validate=lambda d: f"{len(d)} events this week",
    )


async def check_telegram():
    print("\n[4] Telegram")
    token = os.getenv("BOT_TOKEN", "")
    if not token:
        show(WARN, "BOT_TOKEN", "not set in this environment "
                                "(normal if running locally; must be set on Render)")
        return
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
            async with s.get(f"https://api.telegram.org/bot{token}/getMe") as r:
                d = await r.json()
                if d.get("ok"):
                    show(OK, "BOT_TOKEN", f"valid — @{d['result']['username']}")
                else:
                    show(FAIL, "BOT_TOKEN", f"invalid: {d.get('description','unknown error')}")
    except Exception as e:
        show(FAIL, "Telegram API", f"{type(e).__name__}: {e}")


async def main():
    print("=" * 60)
    print("AlphaStrike Health Check —", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("NOTE: run this on the bot's own machine/IP (Render shell).")
    print("=" * 60)
    binance_ok = await check_binance_futures()
    check_ban_file()
    await check_data_feeds()
    await check_telegram()
    print("\n" + "=" * 60)
    if binance_ok:
        print("Verdict: Binance reachable from this IP. If the bot still")
        print("fails on Render, run this same script IN the Render shell —")
        print("Render's IP may be banned even when yours isn't.")
    else:
        print("Verdict: Binance NOT reachable/banned from this IP.")
        print("Fixes: wait out the ban (418 Retry-After), increase the")
        print("sleep between symbol scans in engine.py (currently 0.8s),")
        print("and never run two scans concurrently.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
