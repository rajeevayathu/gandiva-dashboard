#!/usr/bin/env python3
"""
Minervini Trend Template Scanner — NSE India
Runs 14 independent screens daily:
  - 8 individual criterion screens (C1 through C8)
  - 6 composite screens (Full Template, Near Breakout, RS Leaders, VCP, New Highs, Watch List)

Usage:
  python3 scanner.py              # full scan (~1312 stocks, takes 5-10 min first run)
  python3 scanner.py --fast       # quick test with 100 stocks
  python3 scanner.py --symbols 200
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time, pickle, io
import urllib.request
import urllib.parse
import concurrent.futures
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS_JS        = os.path.join(BASE_DIR, 'results_embed.js')
RESULTS_JSON      = os.path.join(BASE_DIR, 'results.json')
BREADTH_JSON      = os.path.join(BASE_DIR, 'breadth.json')
BREADTH_JS        = os.path.join(BASE_DIR, 'breadth_embed.js')
CCI_HISTORY_JSON  = os.path.join(BASE_DIR, 'cci_history.json')
CCI_HISTORY_JS    = os.path.join(BASE_DIR, 'cci_history_embed.js')
PREV_SCREENS_JSON = os.path.join(BASE_DIR, 'prev_screens.json')
# Screens tracked for "New This Scan" diff — order determines display priority
_TRACKED_SCREENS = [
    ('ma_pullback',   'MA Pullback'),
    ('ck_daily_100',  'CCI Daily'),
    ('ck_weekly_100', 'CCI Weekly'),
    ('c1_to_c6',      'C1–C6'),
    ('full_template', 'Full Template'),
]
CACHE_DIR     = os.path.join(BASE_DIR, '.cache')
HISTORY_FILE  = os.path.join(BASE_DIR, 'screen_history.json')
QFIN_CACHE    = os.path.join(CACHE_DIR, '_qfin_cache.pkl')
CHARTINK_CACHE    = os.path.join(CACHE_DIR, '_chartink_cci.pkl')
LISTING_DATE_CACHE = os.path.join(CACHE_DIR, '_listing_dates.pkl')
CACHE_TTL_SEC    = 23 * 3600   # re-download after 23 hours
BREADTH_PRICE_TTL = 7 * 24 * 3600  # refresh full history weekly
QFIN_TTL         = 7 * 24 * 3600  # quarterly financials cache: 7 days
QFIN_TTL_NULL    = 6 * 3600       # retry failed fetches after 6 hours
CHARTINK_TTL     = 4 * 3600       # Chartink CCI refresh every 4 hours

# Symbol file search paths (first found wins)
SYMBOL_PATHS = [
    os.path.join(BASE_DIR, 'nse_symbols.csv'),
    os.path.expanduser('~/Downloads/Zerodha - Approved Securities for MTF (3).csv'),
    os.path.expanduser('~/Downloads/nse_symbols.csv'),
]

# ── TREND TEMPLATE THRESHOLDS ─────────────────────────────────────────────────
TT_ABOVE_LOW_PCT   = 0.25   # C6: at least 25% above 52-week low
TT_BELOW_HIGH_PCT  = 0.25   # C7: within 25% of 52-week high
TT_RS_MIN          = 70     # C8: minimum RS rank (0-99 percentile)
MA200_TREND_BARS   = 21     # C3: MA200 must be rising for at least ~1 month

# ── COMPOSITE SCREEN THRESHOLDS ───────────────────────────────────────────────
NEAR_BREAKOUT_PCT  = 0.05   # within 5% of 52-week high
NEW_HIGH_PCT       = 0.02   # within 2% of 52-week high
RS_LEADER_MIN      = 85     # RS rank threshold for RS Leaders screen
VCP_LOOKBACK       = 60     # bars to look back for VCP detection

# Individual criterion screens show all matching stocks (no cap)
INDIV_SCREEN_MAX   = None

os.makedirs(CACHE_DIR, exist_ok=True)

# ── SCREEN HISTORY ────────────────────────────────────────────────────────────
def load_history():
    """Load screen_history.json — {screen_key: {ticker: 'YYYY-MM-DD'}}"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def record_and_get_date(history, screen_key, ticker, today_str):
    """Return the date this ticker first appeared on screen_key. Record today if new."""
    if screen_key not in history:
        history[screen_key] = {}
    if ticker not in history[screen_key]:
        history[screen_key][ticker] = today_str
    return history[screen_key][ticker]

def prune_history(history, active):
    """Remove tickers no longer on any screen to keep the file compact."""
    for screen_key in list(history.keys()):
        current = active.get(screen_key, set())
        history[screen_key] = {
            t: d for t, d in history[screen_key].items() if t in current
        }


def fetch_chartink_cci_sets():
    """
    Fetch live CCI34 ≥ 100 / cross-100 sets from Chartink screener for all 6 CCI screens.
    Chartink uses real-time NSE data — overcomes scanner cache staleness for CCI detection.
    Returns a dict of sets: {screen_key: {NSE_TICKER, ...}}
    Falls back to empty sets on any error so the scanner continues with computed values.
    Results are cached locally for CHARTINK_TTL (4 hours).
    """
    # Return cached result if still fresh
    if os.path.exists(CHARTINK_CACHE):
        try:
            with open(CHARTINK_CACHE, 'rb') as f:
                cached = pickle.load(f)
            if time.time() - cached.get('_ts', 0) < CHARTINK_TTL:
                print("  Chartink CCI: using cached data.", flush=True)
                return cached
        except Exception:
            pass

    CONDITIONS = {
        'cci34_daily_100':       '( {cash} ( latest cci( 34 ) >= 100 ) )',
        'cci34_weekly_100':      '( {cash} ( weekly cci( 34 ) >= 100 ) )',
        'cci34_daily_cross_100': '( {cash} ( latest cci( 34 ) >= 100 and 1 day ago cci( 34 ) < 100 ) )',
        'cci34_weekly_cross_100':'( {cash} ( weekly cci( 34 ) >= 100 and 1 week ago cci( 34 ) < 100 ) )',
        'cci34_daily_neg100':    '( {cash} ( latest cci( 34 ) >= -100 and 1 day ago cci( 34 ) < -100 ) )',
        'cci34_weekly_neg100':   '( {cash} ( weekly cci( 34 ) >= -100 and 1 week ago cci( 34 ) < -100 ) )',
    }

    result = {'_ts': time.time()}
    for key in CONDITIONS:
        result[key] = set()

    try:
        import re as _re
        import http.cookiejar as _cj

        UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'

        # Use a cookie jar so the session cookie set on GET is sent automatically on POST
        jar     = _cj.CookieJar()
        opener  = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

        # Step 1: GET screener page — sets laravel_session + XSRF-TOKEN cookies
        req0 = urllib.request.Request('https://chartink.com/screener/', headers={
            'User-Agent': UA,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        with opener.open(req0, timeout=15) as resp0:
            html = resp0.read().decode('utf-8', errors='ignore')

        # Extract CSRF token from the HTML meta tag
        csrf = ''
        m = _re.search(r'<meta\s+name=["\']csrf-token["\'][^>]*content=["\']([^"\']+)["\']', html, _re.I)
        if m:
            csrf = m.group(1)
        # Fallback: read XSRF-TOKEN cookie value
        if not csrf:
            for cookie in jar:
                if cookie.name == 'XSRF-TOKEN':
                    csrf = urllib.parse.unquote(cookie.value)
                    break

        if not csrf:
            print("  [WARN] Chartink: could not find CSRF token — skipping live CCI fetch", flush=True)
            return result

        print(f"  Chartink CCI: fetching {len(CONDITIONS)} conditions...", flush=True)
        for key, clause in CONDITIONS.items():
            try:
                post_data = urllib.parse.urlencode({'scan_clause': clause}).encode('utf-8')
                req = urllib.request.Request(
                    'https://chartink.com/screener/process',
                    data=post_data,
                    headers={
                        'User-Agent':        UA,
                        'Accept':            'application/json, text/javascript, */*; q=0.01',
                        'X-Requested-With':  'XMLHttpRequest',
                        'X-CSRF-TOKEN':      csrf,
                        'Content-Type':      'application/x-www-form-urlencoded; charset=UTF-8',
                        'Referer':           'https://chartink.com/screener/',
                        'Origin':            'https://chartink.com',
                    },
                    method='POST'
                )
                with opener.open(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                tickers = {item['nsecode'].strip() for item in data.get('data', []) if item.get('nsecode')}
                result[key] = tickers
                time.sleep(0.5)  # polite delay between requests
            except Exception as e:
                print(f"  [WARN] Chartink fetch for {key} failed: {e}", flush=True)

        total = sum(len(v) for k, v in result.items() if not k.startswith('_'))
        print(f"  Chartink CCI: fetched {total} entries across {len(CONDITIONS)} conditions.", flush=True)

        # Cache the result
        with open(CHARTINK_CACHE, 'wb') as f:
            pickle.dump(result, f)

    except Exception as e:
        print(f"  [WARN] Chartink CCI fetch failed: {e} — using computed values only.", flush=True)

    return result


def get_listing_dates(symbols):
    """
    Returns {sym_ns: first_trading_date_str} for the given symbols.
    Downloads period='max' once per symbol and caches permanently — listing dates never change.
    Only fetches symbols not already in the cache.
    """
    cache = {}
    if os.path.exists(LISTING_DATE_CACHE):
        try:
            with open(LISTING_DATE_CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    to_fetch = [s for s in symbols if s not in cache]
    if to_fetch:
        print(f"  Listing dates: fetching {len(to_fetch)} new symbol(s)...", flush=True)
        for sym in to_fetch:
            try:
                df = yf.download(sym, period='max', interval='1d', progress=False,
                                 auto_adjust=False, timeout=15)
                if df is not None and len(df) > 0:
                    cache[sym] = df.index[0].strftime('%Y-%m-%d')
                else:
                    cache[sym] = None
            except Exception:
                cache[sym] = None
        with open(LISTING_DATE_CACHE, 'wb') as f:
            pickle.dump(cache, f)

    return cache


CRITERIA_META = {
    'c1': 'Price > MA150 & MA200',
    'c2': 'MA150 > MA200',
    'c3': 'MA200 trending up (1 month)',
    'c4': 'MA50 > MA150 & MA200',
    'c5': 'Price > MA50',
    'c6': '25%+ above 52-week low',
    'c7': 'Within 25% of 52-week high',
    'c8': 'RS Rank >= 70',
}

# ── ZERODHA MTF LIVE FETCH ────────────────────────────────────────────────────
MTF_CACHE     = os.path.join(CACHE_DIR, '_zerodha_mtf.pkl')
MIN_LEVERAGE  = 2.0   # only include stocks with leverage >= this


def fetch_mtf_symbols(min_leverage=MIN_LEVERAGE):
    """
    Fetch Zerodha MTF approved securities from Zerodha's JSON API.
    Returns a set of uppercase NSE trading symbols with leverage >= min_leverage.
    Caches result for 23 hours. Falls back gracefully on any error.
    """
    # Return cached version if fresh enough
    if os.path.exists(MTF_CACHE):
        if time.time() - os.path.getmtime(MTF_CACHE) < CACHE_TTL_SEC:
            try:
                with open(MTF_CACHE, 'rb') as f:
                    cached = pickle.load(f)
                    print(f"  MTF: using cached list ({len(cached)} symbols, leverage >= {min_leverage}x)")
                    return cached
            except:
                pass

    # Zerodha's page is JS-rendered; the actual data comes from this JSON endpoint
    API_URL = 'https://public.zrd.sh/crux/approved-mtf-securities.json'
    print(f"  MTF: fetching live list from Zerodha API ...")
    try:
        import json as _json
        req = urllib.request.Request(
            API_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                                   'Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            records = _json.load(resp)

        if not isinstance(records, list) or not records:
            print("  MTF: warning — unexpected API response, skipping MTF filter")
            return None

        symbols = set()
        skipped = 0
        for rec in records:
            sym = str(rec.get('tradingsymbol') or rec.get('symbol') or '').strip().upper()
            if not sym:
                continue
            try:
                lev = float(rec.get('leverage', 0))
                if lev < min_leverage:
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                pass  # no leverage info — include
            symbols.add(sym)

        if not symbols:
            print("  MTF: warning — 0 symbols parsed, skipping MTF filter")
            return None

        print(f"  MTF: {len(symbols)} symbols with leverage >= {min_leverage}x "
              f"({skipped} below threshold)")
        with open(MTF_CACHE, 'wb') as f:
            pickle.dump(symbols, f)
        return symbols

    except Exception as e:
        print(f"  MTF: fetch failed ({e}) — scanning all NSE symbols")
        return None


# ── SYMBOL LOADING ────────────────────────────────────────────────────────────
def load_symbols(max_syms=None):
    # Step 1: load full NSE symbol list from CSV
    all_syms = None
    for path in SYMBOL_PATHS:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            for col in ['Symbol', 'symbol', 'SYMBOL', 'Ticker', 'ticker', 'TICKER', 'tradingsymbol']:
                if col in df.columns:
                    syms = [str(s).strip().upper() for s in df[col].dropna().tolist()]
                    syms = [s for s in syms if s and s != 'NAN']
                    print(f"  Loaded {len(syms)} symbols from {path}")
                    all_syms = syms
                    break
            if all_syms is not None:
                break
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    # Step 2: fetch live MTF list from Zerodha website
    # Use live API as the PRIMARY universe — it is always more up-to-date than the
    # local CSV (new additions like SHADOWFAX appear in the live API before the CSV
    # is re-downloaded). Merge: union of CSV symbols and live API symbols, both
    # constrained to what the live API approves.
    mtf_set = fetch_mtf_symbols(MIN_LEVERAGE)
    if mtf_set is not None:
        if all_syms is not None:
            # Keep CSV order for existing symbols, append new live-API-only symbols at end
            csv_set   = set(all_syms)
            live_only = [s for s in sorted(mtf_set) if s not in csv_set]
            merged    = [s for s in all_syms if s in mtf_set] + live_only
            if live_only:
                print(f"  Live MTF: {len(live_only)} new symbol(s) not in local CSV added: {live_only[:10]}")
            print(f"  MTF filter: {len(merged)} symbols qualify (leverage >= {MIN_LEVERAGE}x)")
            all_syms = merged
        else:
            # No local CSV — use live API as the full universe
            all_syms = sorted(mtf_set)
            print(f"  Live MTF: {len(all_syms)} symbols (no local CSV)")

    if all_syms is not None:
        if max_syms:
            all_syms = all_syms[:max_syms]
        return [s + '.NS' for s in all_syms]

    # Fallback: Nifty 100 hardcoded
    print("No symbol file found — using Nifty 100 fallback")
    nifty100 = [
        'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','HINDUNILVR','ITC','SBIN',
        'BHARTIARTL','KOTAKBANK','LT','AXISBANK','ASIANPAINT','MARUTI','BAJFINANCE',
        'HCLTECH','SUNPHARMA','TITAN','NESTLEIND','ULTRACEMCO','WIPRO','POWERGRID',
        'NTPC','ONGC','TATAMOTORS','ADANIPORTS','JSWSTEEL','TATASTEEL','TECHM',
        'BAJAJFINSV','DIVISLAB','DRREDDY','CIPLA','EICHERMOT','HEROMOTOCO',
        'BPCL','COALINDIA','GRASIM','HINDALCO','IOC','M&M','SBILIFE','HDFC',
        'BRITANNIA','APOLLOHOSP','TATACONSUM','PIDILITIND','SIEMENS','ABB',
        'HAVELLS','VOLTAS','TORNTPHARM','MCDOWELL-N','BERGEPAINT','INDUSINDBK',
        'BANDHANBNK','FEDERALBNK','IDFCFIRSTB','PNB','CANBK','BANKBARODA',
        'GAIL','ADANIENT','ADANIGREEN','ADANIPOWER','TATAPOWER','JSL','SAIL',
        'NMDC','MOIL','NATIONALUM','HINDZINC','VEDL','MUTHOOTFIN','CHOLAFIN',
        'BAJAJ-AUTO','TVSMOTORS','ESCORTS','ASHOKLEY','TVSMOTOR','MOTHERSON',
        'BOSCHLTD','EXIDEIND','AMARAJABAT','SUNDRMFAST','MRF','CEAT',
        'PAGEIND','MPHASIS','LTIM','PERSISTENT','COFORGE','LTTS','KPIT',
        'NAUKRI','ZOMATO','PAYTM','POLICYBZR','DELHIVERY','CARTRADE',
        'IRCTC','IRFC','RVNL','RAILVIKAS','HUDCO','RECLTD','PFC',
    ]
    if max_syms:
        nifty100 = nifty100[:max_syms]
    return [s + '.NS' for s in nifty100]

# ── DATA FETCHING & CACHING ───────────────────────────────────────────────────
def cache_path(sym):
    return os.path.join(CACHE_DIR, sym.replace('.', '_') + '.pkl')

def fetch_data(sym):
    cp = cache_path(sym)
    if os.path.exists(cp):
        if time.time() - os.path.getmtime(cp) < CACHE_TTL_SEC:
            try:
                with open(cp, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
    try:
        df = yf.download(sym, period='2y', interval='1d',
                         progress=False, auto_adjust=False, timeout=15)
        # Use actual closing price (not dividend-adjusted) to match NSE/TradingView
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])
        if df is None or len(df) < 30:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 30:
            return None
        with open(cp, 'wb') as f:
            pickle.dump(df, f)
        return df
    except:
        return None

# ── RS RANK (IBD-style weighted 12-month performance, Minervini SEPA) ────────
# Formula: rank each stock's 12-month return (weighted by recency) as a
# percentile 1-99 within the NSE universe — same approach as IBD RS Rating.
# Weights: most recent 3 months = 40%, each prior quarter = 20%.
# Return for each period = (end_price / start_price) - 1   [NOT inverted]
def calc_rs_ranks(data_dict):
    scores = {}
    for sym, df in data_dict.items():
        try:
            c = df['Close'].squeeze()
            n = len(c)
            # ret(newer_offset, older_offset): +ve means stock rose in that period
            def ret(newer, older):
                i_new = max(0, n - newer - 1)   # closer to today
                i_old = max(0, n - older - 1)   # further in past
                p_new = float(c.iloc[i_new])
                p_old = float(c.iloc[i_old])
                return (p_new / p_old - 1) if p_old > 0 else 0
            q4 = ret(0,   63)   # last 3 months  (weight 40%)
            q3 = ret(63,  126)  # 3-6 months ago (weight 20%)
            q2 = ret(126, 189)  # 6-9 months ago (weight 20%)
            q1 = ret(189, 252)  # 9-12 months ago(weight 20%)
            scores[sym] = q4 * 0.40 + q3 * 0.20 + q2 * 0.20 + q1 * 0.20
        except:
            scores[sym] = -999.0

    vals = sorted(scores.values())
    n = len(vals)
    if n == 0:
        return {}

    def pct(v):
        idx = int(np.searchsorted(vals, v, side='left'))
        return round(idx / n * 99)

    return {sym: pct(s) for sym, s in scores.items()}

# ── TREND TEMPLATE EVALUATION ─────────────────────────────────────────────────
def evaluate(df, rs_rank):
    """Returns full evaluation dict for a stock. None if insufficient data."""
    if len(df) < 210:
        return None
    try:
        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        v = df['Volume'].squeeze()

        price   = float(c.iloc[-1])
        ma50    = float(c.rolling(50).mean().iloc[-1])
        ma150   = float(c.rolling(150).mean().iloc[-1])
        ma200   = float(c.rolling(200).mean().iloc[-1])
        ema10   = float(c.ewm(span=10, adjust=False).mean().iloc[-1])
        ema21   = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        # MA200 value ~1 month ago
        ma200_1m = float(c.rolling(200).mean().iloc[-(MA200_TREND_BARS + 1)])

        # Use intraday High/Low (not Close) to match NSE/TradingView 52-week values
        hi52 = float(h.rolling(252).max().iloc[-1])
        lo52 = float(l.rolling(252).min().iloc[-1])

        vol_avg   = float(v.rolling(50).mean().iloc[-1])
        vol_today = float(v.iloc[-1])
        vol_ratio = (vol_today / vol_avg) if vol_avg > 0 else 1.0

        # ── 8 CRITERIA ────────────────────────────────────────────────────────
        c1 = price > ma150 and price > ma200
        c2 = ma150 > ma200
        c3 = ma200 > ma200_1m
        c4 = ma50 > ma150 and ma50 > ma200
        c5 = price > ma50
        c6 = lo52 > 0 and price >= lo52 * (1 + TT_ABOVE_LOW_PCT)
        c7 = hi52 > 0 and price >= hi52 * (1 - TT_BELOW_HIGH_PCT)
        c8 = rs_rank >= TT_RS_MIN

        criteria = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                    'c5': c5, 'c6': c6, 'c7': c7, 'c8': c8}
        passed = sum(criteria.values())

        def _pct_from_ma(p, ma):
            return round((p - ma) / ma * 100, 2) if ma and ma > 0 else None

        pivot_high,   pivot_bars_ago,   pivot_crossed   = calc_pivot_high(df, weekly=False)
        pivot_high_w, pivot_bars_ago_w, pivot_crossed_w = calc_pivot_high(df, weekly=True)
        pct_from_pivot   = round((price - pivot_high)   / pivot_high   * 100, 2) if pivot_high   else None
        pct_from_pivot_w = round((price - pivot_high_w) / pivot_high_w * 100, 2) if pivot_high_w else None

        return {
            'price':        price,
            'ma50':         ma50,
            'ma150':        ma150,
            'ma200':        ma200,
            'ema10':        ema10,
            'ema21':        ema21,
            'hi52':         hi52,
            'lo52':         lo52,
            'rs_rank':      rs_rank,
            'pct_from_high': (price - hi52) / hi52 * 100 if hi52 > 0 else 0,
            'pct_above_low': (price - lo52) / lo52 * 100 if lo52 > 0 else 0,
            'pct_from_ma50':  _pct_from_ma(price, ma50),
            'pct_from_ema21': _pct_from_ma(price, ema21),
            'pct_from_ema10': _pct_from_ma(price, ema10),
            'pivot_high':       pivot_high,
            'pivot_bars_ago':   pivot_bars_ago,
            'pivot_crossed':    pivot_crossed,
            'pct_from_pivot':   pct_from_pivot,
            'pivot_high_w':     pivot_high_w,
            'pivot_bars_ago_w': pivot_bars_ago_w,
            'pivot_crossed_w':  pivot_crossed_w,
            'pct_from_pivot_w': pct_from_pivot_w,
            'vol_ratio':    vol_ratio,
            'criteria':     criteria,
            'passed':       passed,
            'all_pass':     passed == 8,
        }
    except:
        return None

# ── IPO / RECENT LISTING EVALUATION ───────────────────────────────────────────
def evaluate_ipo(df, rs_rank):
    """Lightweight evaluation for stocks with 30-209 days of data (recent IPOs/listings)."""
    try:
        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        v = df['Volume'].squeeze()
        n = len(c)

        price         = float(c.iloc[-1])
        listing_high  = float(h.max())
        listing_low   = float(l.min())
        pct_from_high = (price - listing_high) / listing_high * 100 if listing_high > 0 else 0
        pct_above_low = (price - listing_low)  / listing_low  * 100 if listing_low  > 0 else 0

        # MAs — use whatever bars are available (min_periods relaxed for IPOs)
        def _safe_ma(series, span, is_ema=False):
            try:
                if is_ema:
                    val = float(series.ewm(span=span, adjust=False, min_periods=max(1, span//2)).mean().iloc[-1])
                else:
                    val = float(series.rolling(span, min_periods=max(1, span//2)).mean().iloc[-1])
                return round(val, 2) if not np.isnan(val) else None
            except:
                return None

        ma50  = _safe_ma(c, 50)
        ema21 = _safe_ma(c, 21, is_ema=True)
        ema10 = _safe_ma(c, 10, is_ema=True)
        above_ma50 = (price > ma50) if ma50 is not None else None

        def _pma(ma): return round((price - ma) / ma * 100, 2) if ma and ma > 0 else None

        # Volume ratio vs 20-day avg
        vol_avg20 = float(v.rolling(20, min_periods=10).mean().iloc[-1])
        vol_today = float(v.iloc[-1])
        vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

        # Daily pivot (use left=5,right=5 for IPOs — shorter history)
        pv_high, pv_bars, pv_crossed = calc_pivot_high(df, left=5, right=5)
        pv_high_w, pv_bars_w, pv_crossed_w = calc_pivot_high(df, left=5, right=5, weekly=True)
        pct_pivot   = _pma(pv_high)
        pct_pivot_w = _pma(pv_high_w)

        return {
            'price':           price,
            'ma50':            ma50,
            'ema21':           ema21,
            'ema10':           ema10,
            'above_ma50':      above_ma50,
            'listing_high':    round(listing_high, 2),
            'listing_low':     round(listing_low, 2),
            'pct_from_high':   round(pct_from_high, 1),
            'pct_above_low':   round(pct_above_low, 1),
            'pct_from_ma50':   _pma(ma50),
            'pct_from_ema21':  _pma(ema21),
            'pct_from_ema10':  _pma(ema10),
            'vol_ratio':       round(vol_ratio, 2),
            'rs_rank':         rs_rank,
            'days_listed':     n,
            'pivot_high':      pv_high,
            'pivot_bars_ago':  pv_bars,
            'pivot_crossed':   pv_crossed,
            'pct_from_pivot':  pct_pivot,
            'pivot_high_w':    pv_high_w,
            'pivot_bars_ago_w': pv_bars_w,
            'pivot_crossed_w': pv_crossed_w,
            'pct_from_pivot_w': pct_pivot_w,
        }
    except:
        return None

# ── PIVOT HIGH CALCULATION ────────────────────────────────────────────────────
def calc_pivot_high(df, left=10, right=10, weekly=False):
    """
    Returns (pivot_price, bars_ago, crossed) matching Pine Script ta.pivothigh(left, right).
    A bar i is a pivot high if high[i] is strictly the maximum of the window [i-left .. i+right].
    Most recent confirmed pivot requires at least 'right' bars after it to have closed.
    crossed = True if latest close >= pivot and prior close < pivot.
    weekly=True resamples to weekly bars first.
    """
    try:
        if weekly:
            ohlcv = df.copy()
            if isinstance(ohlcv.columns, pd.MultiIndex):
                ohlcv.columns = ohlcv.columns.get_level_values(0)
            ohlcv = ohlcv[['High', 'Close']].resample('W').agg(
                {'High': 'max', 'Close': 'last'}).dropna()
            h = ohlcv['High'].values
            c = ohlcv['Close'].values
        else:
            h = df['High'].squeeze().values
            c = df['Close'].squeeze().values

        n = len(h)
        if n < left + right + 2:
            return None, None, False

        pivot_price = None
        bars_ago    = None
        for i in range(n - right - 1, left - 1, -1):
            window_max = max(h[i - left: i + right + 1])
            if h[i] == window_max:
                count_eq = sum(1 for x in h[i - left: i + right + 1] if x == window_max)
                if count_eq == 1:
                    pivot_price = round(float(h[i]), 2)
                    bars_ago    = n - 1 - i
                    break
        if pivot_price is None:
            return None, None, False
        crossed = bool(len(c) >= 2 and c[-1] >= pivot_price and c[-2] < pivot_price)
        return pivot_price, bars_ago, crossed
    except:
        return None, None, False


# ── CCI(34) CALCULATION ────────────────────────────────────────────────────────
def calc_cci(df, period=34, weekly=False):
    """
    Returns (cci_today, cci_prev) for daily or weekly timeframe.
    CCI = (Typical Price - SMA) / (0.015 * Mean Absolute Deviation)
    """
    try:
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        c = df['Close'].squeeze()

        if weekly:
            # Resample to weekly (week ending Friday)
            ohlc = df.copy()
            if isinstance(ohlc.columns, pd.MultiIndex):
                ohlc.columns = ohlc.columns.get_level_values(0)
            ohlc = ohlc[['High','Low','Close']].resample('W').agg(
                {'High':'max','Low':'min','Close':'last'}).dropna()
            if len(ohlc) < period + 2:
                return None, None
            h = ohlc['High'].squeeze()
            l = ohlc['Low'].squeeze()
            c = ohlc['Close'].squeeze()

        if len(c) < period + 2:
            return None, None

        tp = (h + l + c) / 3.0
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (tp - sma) / (0.015 * mad)

        return float(cci.iloc[-1]), float(cci.iloc[-2])
    except:
        return None, None


def calc_cci_streak_start(df, period=34, weekly=False):
    """
    Returns the date (YYYY-MM-DD) when the current continuous CCI34 >= 100 streak began.
    For weekly: maps the weekly bar back to the first actual trading day of that week
    (avoids future week-end labels from resample('W') on an incomplete current week).
    """
    try:
        today_cap = datetime.today().strftime('%Y-%m-%d')

        if weekly:
            ohlc = df.copy()
            if isinstance(ohlc.columns, pd.MultiIndex):
                ohlc.columns = ohlc.columns.get_level_values(0)
            ohlc = ohlc[['High','Low','Close']].resample('W').agg(
                {'High':'max','Low':'min','Close':'last'}).dropna()
            if len(ohlc) < period + 2:
                return None
            h, l, c = ohlc['High'].squeeze(), ohlc['Low'].squeeze(), ohlc['Close'].squeeze()
            tp  = (h + l + c) / 3.0
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = ((tp - sma) / (0.015 * mad)).dropna()

            if len(cci) == 0 or float(cci.iloc[-1]) < 100:
                return None

            streak_pos = len(cci) - 1
            for i in range(len(cci) - 2, -1, -1):
                if float(cci.iloc[i]) < 100:
                    break
                streak_pos = i

            # resample('W') labels each bar with the week-END date, which is a future
            # date for the current incomplete week. Map back to the first actual trading
            # day in the original daily df that falls within that week instead.
            week_end   = pd.Timestamp(cci.index[streak_pos])
            week_start = week_end - pd.Timedelta(days=6)
            daily_idx  = df.index
            in_week    = daily_idx[(daily_idx >= week_start) & (daily_idx <= week_end)]
            if len(in_week) > 0:
                result = pd.Timestamp(in_week[0]).strftime('%Y-%m-%d')
            else:
                result = week_end.strftime('%Y-%m-%d')
            # Cap any remaining future date (e.g. resample edge cases)
            return min(result, today_cap)
        else:
            if len(df) < period + 2:
                return None
            h, l, c = df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()
            tp  = (h + l + c) / 3.0
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = ((tp - sma) / (0.015 * mad)).dropna()

            if len(cci) == 0 or float(cci.iloc[-1]) < 100:
                return None

            streak_pos = len(cci) - 1
            for i in range(len(cci) - 2, -1, -1):
                if float(cci.iloc[i]) < 100:
                    break
                streak_pos = i

            return pd.Timestamp(cci.index[streak_pos]).strftime('%Y-%m-%d')
    except:
        return None


# ── VCP DETECTION ─────────────────────────────────────────────────────────────
def detect_vcp(df):
    """
    Volatility Contraction Pattern: 3 segments of decreasing price range
    with drying volume in the most recent segment.
    """
    if len(df) < VCP_LOOKBACK:
        return False
    try:
        recent = df.iloc[-VCP_LOOKBACK:]
        seg = VCP_LOOKBACK // 3
        ranges, vols = [], []
        for i in range(3):
            s = recent.iloc[i * seg:(i + 1) * seg]
            mid = float(s['Close'].mean())
            if mid <= 0:
                return False
            rng = (float(s['High'].max()) - float(s['Low'].min())) / mid
            ranges.append(rng)
            vols.append(float(s['Volume'].mean()))

        contracting_price = ranges[1] < ranges[0] and ranges[2] < ranges[1]
        contracting_vol   = vols[2] < vols[0]
        price_start = float(recent['Close'].iloc[0])
        price_end   = float(recent['Close'].iloc[-1])
        still_up    = price_end > price_start * 0.85

        return contracting_price and contracting_vol and still_up
    except:
        return False

# ── HIGH TIGHT FLAG DETECTION ─────────────────────────────────────────────────
def detect_htf(df):
    """
    High Tight Flag (O'Neil) detection. Returns (is_full, is_potential, details|None).

    Full HTF:
      Pole  — price gained ≥90% within ≤40 trading days (8 weeks)
      Flag  — 5-25 trading days past the peak, current price within 25% of peak

    Potential HTF:
      Pole  — price gained ≥50% within ≤40 trading days
      Holding within 15% of the recent peak; flag still forming or gain just below 90%
    """
    MAX_POLE = 40   # 8 weeks
    FLAG_MIN =  5   # 1 week
    FLAG_MAX = 25   # 5 weeks
    LOOKBACK = MAX_POLE + FLAG_MAX + 5  # 70 bars

    if len(df) < LOOKBACK:
        return False, False, None
    try:
        recent = df.iloc[-LOOKBACK:]

        peak_pos        = int(recent['High'].values.argmax())
        peak_price      = float(recent['High'].iloc[peak_pos])
        curr_price      = float(df['Close'].iloc[-1])
        days_since_peak = len(recent) - 1 - peak_pos
        pct_from_peak   = (curr_price / peak_price - 1.0) * 100.0

        # Lowest low in up to MAX_POLE days before the peak = pole start
        pole_slice   = recent.iloc[max(0, peak_pos - MAX_POLE): peak_pos + 1]
        if len(pole_slice) < 3:
            return False, False, None
        pole_low_rel = int(pole_slice['Low'].values.argmin())
        pole_low     = float(pole_slice['Low'].iloc[pole_low_rel])
        pole_days    = len(pole_slice) - 1 - pole_low_rel  # bars from low to peak

        if pole_low <= 0 or pole_days <= 0:
            return False, False, None

        gain = (peak_price - pole_low) / pole_low

        details = {
            'htf_gain_pct':        round(gain * 100.0, 1),
            'htf_pole_days':       pole_days,
            'htf_days_since_peak': days_since_peak,
            'htf_pct_from_peak':   round(pct_from_peak, 1),
        }

        is_full = (
            gain >= 0.9 and
            pole_days <= MAX_POLE and
            FLAG_MIN <= days_since_peak <= FLAG_MAX and
            pct_from_peak >= -25.0
        )
        is_potential = (not is_full) and (
            gain >= 0.5 and
            pole_days <= MAX_POLE and
            pct_from_peak >= -15.0 and
            days_since_peak <= FLAG_MAX
        )
        return is_full, is_potential, details
    except Exception:
        return False, False, None


# ── HIGHER HIGH / HIGHER LOW PULLBACK DETECTION ───────────────────────────────
def detect_primary_base(df, listing_date_str, max_years):
    """
    Minervini Primary Base: the first buyable consolidation after a stock goes public.
    listing_date_str: 'YYYY-MM-DD' of when the stock first traded (from get_listing_dates)
    max_years: maximum years since listing to qualify (3 for recent, 10 for young company)

    Conditions:
      1. Listed within max_years years ago
      2. Price within 20% of its all-time high (still constructive)
      3. The base window (up to 120 days) has a price range < 35% (tight consolidation)
      4. The high of the base is within 15% of the ATH (base is anchored near the top)
      5. Base duration >= 15 trading days (at least 3 weeks)
      6. Price still above MA50/MA150/MA200 (uptrend intact)
    """
    BASE_MIN_DAYS   = 15
    BASE_MAX_WINDOW = 120
    ATH_PROXIMITY   = 0.20
    BASE_MAX_RANGE  = 35.0
    BASE_ATH_GAP    = 0.15

    try:
        if listing_date_str is None or len(df) < 50:
            return False, None
        listing_date = datetime.strptime(listing_date_str, '%Y-%m-%d')
        years_listed = (datetime.today() - listing_date).days / 365.25
        if years_listed > max_years or years_listed < 0:
            return False, None

        c    = df['Close'].squeeze()
        h    = df['High'].squeeze()
        v    = df['Volume'].squeeze()
        curr = float(c.iloc[-1])
        ath  = float(h.max())

        # Must be within 20% of ATH
        pct_from_ath = (curr / ath - 1) * 100
        if pct_from_ath < -(ATH_PROXIMITY * 100):
            return False, None

        # Base window: up to BASE_MAX_WINDOW days, at least BASE_MIN_DAYS
        window = min(BASE_MAX_WINDOW, len(df) - 1)
        if window < BASE_MIN_DAYS:
            return False, None

        base_h = float(h.iloc[-window:].max())
        base_l = float(df['Low'].iloc[-window:].min())
        base_range = (base_h - base_l) / base_l * 100

        # Base must be tight
        if base_range > BASE_MAX_RANGE:
            return False, None

        # Base must be anchored near ATH (not a base after a big crash)
        if (base_h / ath - 1) * 100 < -(BASE_ATH_GAP * 100):
            return False, None

        # Volume: base should not have higher volume than pre-base (no heavy distribution)
        pre_start = max(0, len(df) - window * 2)
        pre_vol   = float(v.iloc[pre_start:len(df)-window].mean()) if len(df) > window else None
        base_vol  = float(v.iloc[-window:].mean())
        vol_ratio = round(base_vol / pre_vol, 2) if (pre_vol and pre_vol > 0) else None

        # Minervini trend criteria within the base: still above MA50/MA150/MA200
        ma50  = float(c.rolling(50).mean().iloc[-1])
        ma150 = float(c.rolling(150).mean().iloc[-1]) if len(c) >= 150 else None
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
        in_uptrend = (curr > ma50 and
                      (ma150 is None or curr > ma150) and
                      (ma200 is None or curr > ma200))
        if not in_uptrend:
            return False, None

        return True, {
            'base_range_pct': round(base_range, 1),
            'pct_from_ath':   round(pct_from_ath, 1),
            'base_days':      window,
            'listing_days':   round(years_listed, 1),  # years since listing
            'base_vol_ratio': vol_ratio,
        }
    except Exception:
        return False, None


def detect_hhhl(df):
    """
    Detects stocks in a confirmed uptrend (HH+HL structure) pulling back
    to near the most recent confirmed Higher Low.

    Conditions:
      1. At least 2 confirmed pivot highs — each higher than the previous (HH)
      2. At least 2 confirmed pivot lows  — each higher than the previous (HL)
      3. Sequence is valid: prev_HL → prev_HH → recent_HL → recent_HH → now pulling back
      4. Current price is 0–8% above the most recent pivot low (near the HL)
      5. Pullback from most recent pivot high is 8–35% (healthy, not a breakdown)
      6. Price is above MA50 (uptrend confirmed by MA)
    Returns (is_setup, details|None)
    """
    PIVOT_N      = 5     # bars on each side to confirm a swing pivot
    LOOKBACK     = 150   # bars of history to scan
    NEAR_HL_MAX  = 8.0   # max % above the recent HL
    PB_MIN       = 8.0   # min pullback % from recent HH
    PB_MAX       = 35.0  # max pullback % from recent HH

    if len(df) < LOOKBACK + PIVOT_N * 2:
        return False, None
    try:
        recent  = df.iloc[-LOOKBACK:]
        highs   = recent['High'].values
        lows    = recent['Low'].values
        closes  = recent['Close'].values
        n       = len(recent)

        # Find all confirmed pivot highs and lows
        pivot_highs, pivot_lows = [], []
        for i in range(PIVOT_N, n - PIVOT_N):
            if highs[i] == max(highs[i - PIVOT_N: i + PIVOT_N + 1]):
                pivot_highs.append((i, float(highs[i])))
            if lows[i]  == min(lows[i  - PIVOT_N: i + PIVOT_N + 1]):
                pivot_lows.append((i, float(lows[i])))

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return False, None

        # Last two pivot highs must be ascending (Higher Highs)
        ph_prev_idx, ph_prev = pivot_highs[-2]
        ph_last_idx, ph_last = pivot_highs[-1]
        if ph_last <= ph_prev:
            return False, None

        # Last two pivot lows must be ascending (Higher Lows)
        pl_prev_idx, pl_prev = pivot_lows[-2]
        pl_last_idx, pl_last = pivot_lows[-1]
        if pl_last <= pl_prev:
            return False, None

        # Sequence check: prev_HL → prev_HH → recent_HL → recent_HH
        # i.e. pl_prev before ph_prev, ph_prev before pl_last, pl_last before ph_last
        if not (pl_prev_idx < ph_prev_idx < pl_last_idx < ph_last_idx):
            return False, None

        curr = float(closes[-1])

        # Price must be above MA50
        ma50_val = float(df['Close'].iloc[-50:].mean()) if len(df) >= 50 else None
        if ma50_val is None or curr < ma50_val:
            return False, None

        # Near the recent Higher Low (0–8% above it)
        pct_from_hl = (curr - pl_last) / pl_last * 100.0
        if not (0.0 <= pct_from_hl <= NEAR_HL_MAX):
            return False, None

        # Healthy pullback from the recent Higher High (8–35%)
        pullback = (ph_last - curr) / ph_last * 100.0
        if not (PB_MIN <= pullback <= PB_MAX):
            return False, None

        return True, {
            'hhhl_hh':          round(ph_last, 2),
            'hhhl_hl':          round(pl_last, 2),
            'hhhl_pullback_pct': round(pullback, 1),
            'hhhl_pct_from_hl': round(pct_from_hl, 1),
        }
    except Exception:
        return False, None


# ── VCP LAST SETUP DATE ───────────────────────────────────────────────────────
def find_last_vcp_date(df, weekly=False, max_lookback=120):
    """
    Find the most recent date (within max_lookback bars) when a VCP pattern was active.
    Scans backward in steps of 5 bars for efficiency.
    Returns 'YYYY-MM-DD' string or None.
    """
    try:
        if weekly:
            ohlcv = df.copy()
            if isinstance(ohlcv.columns, pd.MultiIndex):
                ohlcv.columns = ohlcv.columns.get_level_values(0)
            ohlcv = ohlcv[['Open', 'High', 'Low', 'Close', 'Volume']].resample('W').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min',
                 'Close': 'last', 'Volume': 'sum'}).dropna()
            df_use = ohlcv
        else:
            df_use = df

        n = len(df_use)
        if n < VCP_LOOKBACK + 5:
            return None

        limit = max(VCP_LOOKBACK, n - max_lookback)
        for end_i in range(n, limit - 1, -5):
            if detect_vcp(df_use.iloc[:end_i]):
                return str(df_use.index[end_i - 1].date())
        return None
    except:
        return None

# ── CRITERIA STREAK DATES ─────────────────────────────────────────────────────
def find_criteria_dates(df):
    """
    For each criterion C1-C7, find the date when the current qualifying streak began.
    This is the date the stock CONTINUOUSLY started meeting that criterion based on
    actual price action — not the date the scanner first noticed it.

    Example: if C2 (MA150 > MA200) has been true since 2025-11-14, that is returned.
    Returns dict {c1: 'YYYY-MM-DD', c2: ..., ...} only for currently-passing criteria.
    """
    try:
        c  = df['Close'].squeeze()
        h  = df['High'].squeeze()
        l  = df['Low'].squeeze()
        n  = len(c)
        if n < 252:
            return {}

        ma50         = c.rolling(50).mean()
        ma150        = c.rolling(150).mean()
        ma200        = c.rolling(200).mean()
        ma200_lagged = ma200.shift(MA200_TREND_BARS)
        lo52         = l.rolling(252).min()
        hi52         = h.rolling(252).max()
        dates        = df.index

        def streak_start(series, min_bar):
            """
            Walk backwards from today to find when the current True streak started.
            Returns ISO date string, or None if criterion is not currently met.
            """
            vals = series.fillna(False).values.astype(bool)
            if not vals[-1]:
                return None
            for i in range(len(vals) - 2, min_bar - 1, -1):
                if not vals[i]:
                    return str(dates[i + 1].date())
            return str(dates[min_bar].date())  # streak reaches limit of data

        res = {}
        d = streak_start((c > ma150) & (c > ma200),                               200)
        if d: res['c1'] = d
        d = streak_start(ma150 > ma200,                                            200)
        if d: res['c2'] = d
        d = streak_start(ma200 > ma200_lagged,                                     200 + MA200_TREND_BARS)
        if d: res['c3'] = d
        d = streak_start((ma50 > ma150) & (ma50 > ma200),                         200)
        if d: res['c4'] = d
        d = streak_start(c > ma50,                                                  50)
        if d: res['c5'] = d
        d = streak_start((c - lo52) / lo52.replace(0, np.nan) >= TT_ABOVE_LOW_PCT, 252)
        if d: res['c6'] = d
        d = streak_start((hi52 - c) / hi52.replace(0, np.nan) <= TT_BELOW_HIGH_PCT, 252)
        if d: res['c7'] = d
        return res
    except:
        return {}

# ── BUILD STOCK ENTRY ─────────────────────────────────────────────────────────
def build_entry(sym, ev, extra=None):
    ticker = sym.replace('.NS', '')
    e = {
        'ticker':        ticker,
        'tv_symbol':     'NSE:' + ticker,
        'price':         round(ev['price'], 2),
        'rs_rank':       int(ev['rs_rank']),
        'ma50':          round(ev['ma50'], 2),
        'ma150':         round(ev['ma150'], 2),
        'ma200':         round(ev['ma200'], 2),
        'hi52':          round(ev['hi52'], 2),
        'lo52':          round(ev['lo52'], 2),
        'pct_from_high':  round(ev['pct_from_high'], 1),
        'pct_above_low':  round(ev['pct_above_low'], 1),
        'vol_ratio':      round(ev['vol_ratio'], 2),
        'passed':         ev['passed'],
        'criteria':       ev['criteria'],
        'pct_from_ma50':  ev.get('pct_from_ma50'),
        'pct_from_ema21': ev.get('pct_from_ema21'),
        'pct_from_ema10': ev.get('pct_from_ema10'),
        'pivot_high':       ev.get('pivot_high'),
        'pivot_bars_ago':   ev.get('pivot_bars_ago'),
        'pivot_crossed':    ev.get('pivot_crossed', False),
        'pct_from_pivot':   ev.get('pct_from_pivot'),
        'pivot_high_w':     ev.get('pivot_high_w'),
        'pivot_bars_ago_w': ev.get('pivot_bars_ago_w'),
        'pivot_crossed_w':  ev.get('pivot_crossed_w', False),
        'pct_from_pivot_w': ev.get('pct_from_pivot_w'),
        'vcp_last_daily':   ev.get('vcp_last_daily'),
        'vcp_last_weekly':  ev.get('vcp_last_weekly'),
    }
    if extra:
        e.update(extra)
    return e

# ── QUARTERLY FINANCIALS (CANSLIM C & A) ──────────────────────────────────────
QFIN_FIELDS = [
    'q_label',
    'q_revenue', 'q_rev_qoq', 'q_rev_yoy',
    'q_np',      'q_np_qoq',  'q_np_yoy',
    'q_eps',     'q_eps_qoq', 'q_eps_yoy',
]

def _parse_num_scr(s):
    """Parse a screener.in number string → float or None. Handles '1,234.56', '-23', ''."""
    if not s:
        return None
    s = s.replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_one_qfin(sym):
    """Fetch quarterly financials via yfinance for one NSE symbol. Returns (sym, dict|None).

    yfinance quarterly_financials columns are newest-first DatetimeIndex.
    Revenue and Net Income are in raw INR — divide by 1e7 to get Crores.
    EPS (Basic EPS) is already in Rs.
    """
    try:
        import yfinance as _yf, pandas as _pd
        tk = _yf.Ticker(sym)
        qf = tk.quarterly_financials
        if qf is None or qf.empty or len(qf.columns) < 2:
            return (sym, None)

        cols = qf.columns  # newest first

        def _val(row_name, col_idx):
            if row_name not in qf.index:
                return None
            if col_idx >= len(cols):
                return None
            v = qf.loc[row_name, cols[col_idx]]
            try:
                v = float(v)
                return None if (v != v) else v   # NaN guard
            except Exception:
                return None

        def _pct(new_v, old_v):
            if new_v is None or old_v is None or old_v == 0:
                return None
            return round((new_v - old_v) / abs(old_v) * 100, 1)

        # Find YoY column — closest date to exactly 1 year before current quarter
        current_date = cols[0]
        yoy_target   = current_date - _pd.DateOffset(years=1)
        yoy_idx      = min(range(1, len(cols)), key=lambda i: abs((cols[i] - yoy_target).days))
        yoy_gap_days = abs((cols[yoy_idx] - yoy_target).days)
        if yoy_gap_days > 60:   # no column within 2 months of yoy target → skip yoy
            yoy_idx = None

        # Quarter label: "Mar 2026"
        q_label = current_date.strftime('%b %Y')

        d = {'q_label': q_label}

        # ── Revenue (Total Revenue → Crores) ─────────────────────────────────
        rev_row = next((r for r in ['Total Revenue', 'Operating Revenue'] if r in qf.index), None)
        if rev_row:
            r0 = _val(rev_row, 0)
            r1 = _val(rev_row, 1)
            r_yoy = _val(rev_row, yoy_idx) if yoy_idx is not None else None
            if r0 is not None:
                d['q_revenue'] = round(r0 / 1e7, 1)
                d['q_rev_qoq'] = _pct(r0, r1)
                d['q_rev_yoy'] = _pct(r0, r_yoy)

        # ── Net Income / Net Profit (→ Crores) ───────────────────────────────
        ni_row = next((r for r in ['Net Income', 'Net Income Common Stockholders'] if r in qf.index), None)
        if ni_row:
            n0 = _val(ni_row, 0)
            n1 = _val(ni_row, 1)
            n_yoy = _val(ni_row, yoy_idx) if yoy_idx is not None else None
            if n0 is not None:
                d['q_np'] = round(n0 / 1e7, 1)
                d['q_np_qoq'] = _pct(n0, n1)
                d['q_np_yoy'] = _pct(n0, n_yoy)

        # ── EPS (already in Rs) ───────────────────────────────────────────────
        eps_row = next((r for r in ['Basic EPS', 'Diluted EPS'] if r in qf.index), None)
        if eps_row:
            e0 = _val(eps_row, 0)
            e1 = _val(eps_row, 1)
            e_yoy = _val(eps_row, yoy_idx) if yoy_idx is not None else None
            if e0 is not None:
                d['q_eps'] = round(e0, 2)
                d['q_eps_qoq'] = _pct(e0, e1)
                d['q_eps_yoy'] = _pct(e0, e_yoy)

        has_data = any(v is not None for k, v in d.items() if k != 'q_label')
        return (sym, d if has_data else None)

    except Exception:
        return (sym, None)


def fetch_quarterly_financials(syms):
    """
    Fetch quarterly EPS / Revenue / Net Profit for each symbol.
    Returns dict: {'TICKER.NS': {'q_label': 'Mar 2026', 'q_eps_yoy': 32.1, ...}, ...}
    Results are cached per-ticker for QFIN_TTL (7 days).
    Only symbols NOT in cache (or with stale cache) are downloaded.
    """
    # Load persistent cache
    cache = {}
    if os.path.exists(QFIN_CACHE):
        try:
            with open(QFIN_CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    now = time.time()
    results = {}
    to_fetch = []

    for sym in syms:
        cached = cache.get(sym)
        if cached:
            ttl = QFIN_TTL if cached.get('data') else QFIN_TTL_NULL
            if now - cached.get('ts', 0) < ttl:
                results[sym] = cached.get('data')
                continue
        to_fetch.append(sym)

    if to_fetch:
        print(f"  Fetching quarterly results (yfinance) for {len(to_fetch)} stocks...", flush=True)
        done_count = [0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_fetch_one_qfin, sym): sym for sym in to_fetch}
            for fut in concurrent.futures.as_completed(futures):
                sym, data = fut.result()
                results[sym] = data
                cache[sym] = {'ts': now, 'data': data}
                done_count[0] += 1
                if done_count[0] % 25 == 0 or done_count[0] == len(to_fetch):
                    print(f"    {done_count[0]}/{len(to_fetch)} fetched...", end='\r', flush=True)
        print(f"    {len(to_fetch)}/{len(to_fetch)} quarterly results done.       ")
        # Save updated cache
        try:
            with open(QFIN_CACHE, 'wb') as f:
                pickle.dump(cache, f)
        except Exception:
            pass

    return results

# ── MARKET BREADTH HISTORY ────────────────────────────────────────────────────
NSE_EQUITY_URL  = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
NSE_EQUITY_CACHE = os.path.join(CACHE_DIR, '_nse_equity_list.pkl')

def fetch_nse_all_symbols():
    """
    Fetch the full NSE EQ-series equity list (~2100+ stocks) from NSE archives.
    Returns list of 'SYMBOL.NS' strings. Cached for 7 days.
    """
    if os.path.exists(NSE_EQUITY_CACHE):
        if time.time() - os.path.getmtime(NSE_EQUITY_CACHE) < BREADTH_PRICE_TTL:
            try:
                with open(NSE_EQUITY_CACHE, 'rb') as f:
                    syms = pickle.load(f)
                    print(f"  NSE equity list: {len(syms)} symbols (cached)")
                    return syms
            except:
                pass
    try:
        req = urllib.request.Request(NSE_EQUITY_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            import io as _io
            df = pd.read_csv(_io.BytesIO(r.read()))
        df.columns = [c.strip() for c in df.columns]
        eq = df[df['SERIES'].str.strip() == 'EQ']['SYMBOL'].dropna().tolist()
        syms = [s.strip() + '.NS' for s in eq if str(s).strip()]
        print(f"  NSE equity list: {len(syms)} EQ-series symbols (fetched from NSE)")
        with open(NSE_EQUITY_CACHE, 'wb') as f:
            pickle.dump(syms, f)
        return syms
    except Exception as e:
        print(f"  NSE equity fetch failed ({e}), falling back to scan universe")
        return []


# ── NSE BHAVCOPY BREADTH ENGINE ────────────────────────────────────────────────
BHAV_STATE_PKL   = os.path.join(CACHE_DIR, 'bhavcopy_state.pkl')   # rolling price buffers
BHAV_BREADTH_PKL = os.path.join(CACHE_DIR, 'bhavcopy_breadth.pkl') # MA50/MA200 tuple counts
BHAV_MULTI_PKL   = os.path.join(CACHE_DIR, 'bhavcopy_multi.pkl')   # all MA periods (10/20/40/50/150/200 + weekly)

DAILY_MA_PERIODS  = [10, 20, 40, 50, 150, 200]
WEEKLY_MA_PERIODS = [10, 40]   # in weeks; approximated from daily buffer (5-bar sampling)

# FII/DII cash-market participant data
FIIDII_PKL = os.path.join(CACHE_DIR, 'fiidii_data.pkl')
FIIDII_TTL = 23 * 3600


def fetch_fiidii_data():
    """
    Fetch NSE cash-segment FII/DII daily activity.

    Strategy (incremental accumulation):
      1. Load all previously stored FII/DII history from pkl.
      2. Try to import a user-provided CSV at fiidii_history.csv in the scanner dir
         (user can download this from NSE website for historical data).
      3. Fetch the latest trading day's data from NSE API (works without auth via ?date=).
      4. Append any new dates and save updated pkl.

    NSE endpoint always returns the LATEST trading day regardless of the ?date= param —
    so historical data must come from (2) above or accumulates over time via daily runs.

    Returns list of dicts [{date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net}]
    sorted oldest-first.  Values in ₹ Crore.
    """
    _UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
           'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

    def _safe_float(v):
        try:
            return round(float(str(v).replace(',', '').replace('(', '-').replace(')', '')), 2)
        except Exception:
            return 0.0

    def _parse_date(raw):
        for fmt in ('%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%B %d, %Y'):
            try:
                from datetime import datetime as _dt
                return _dt.strptime(str(raw).strip(), fmt).strftime('%Y-%m-%d')
            except Exception:
                pass
        return None

    # ── Step 1: Load stored history (never expires — accumulates over time) ──
    rows = {}   # {date_str: dict}
    if os.path.exists(FIIDII_PKL):
        try:
            with open(FIIDII_PKL, 'rb') as f:
                stored = pickle.load(f)
            for r in (stored or []):
                if isinstance(r, dict) and r.get('date'):
                    rows[r['date']] = r
        except Exception:
            pass

    # ── Step 2: Import user-provided historical CSV if present ───────────────
    # User can download from: NSE India → Market Activity → FII/DII Statistics
    # Save as 'fiidii_history.csv' in the scanner directory.
    csv_path = os.path.join(BASE_DIR, 'fiidii_history.csv')
    if os.path.exists(csv_path):
        try:
            df_csv = pd.read_csv(csv_path)
            df_csv.columns = [c.strip().lower().replace(' ', '_') for c in df_csv.columns]
            before = len(rows)
            for _, row in df_csv.iterrows():
                dt = _parse_date(row.get('date', ''))
                if not dt or dt in rows:
                    continue
                def _col(*names):
                    for n in names:
                        v = row.get(n)
                        if v is not None and str(v).strip() not in ('', 'nan'):
                            return _safe_float(v)
                    return 0.0
                rows[dt] = {
                    'date':     dt,
                    'fii_buy':  _col('fii_buy_value', 'fii_buy', 'buy_value_(fii)', 'fii_buy_value_(cr.)'),
                    'fii_sell': _col('fii_sell_value', 'fii_sell', 'sell_value_(fii)', 'fii_sell_value_(cr.)'),
                    'fii_net':  _col('fii_net_value', 'fii_net', 'net_value_(fii)', 'fii_net_value_(cr.)'),
                    'dii_buy':  _col('dii_buy_value', 'dii_buy', 'buy_value_(dii)', 'dii_buy_value_(cr.)'),
                    'dii_sell': _col('dii_sell_value', 'dii_sell', 'sell_value_(dii)', 'dii_sell_value_(cr.)'),
                    'dii_net':  _col('dii_net_value', 'dii_net', 'net_value_(dii)', 'dii_net_value_(cr.)'),
                }
            added = len(rows) - before
            if added:
                print(f"    FII/DII CSV import: +{added} days from {csv_path}")
        except Exception as e:
            print(f"    FII/DII CSV import failed: {e}")

    # ── Step 3: Fetch latest day from NSE API ────────────────────────────────
    # NSE endpoint always returns the LATEST trading day (server-side limitation).
    # Uses curl_cffi (Chrome TLS fingerprint) which NSE accepts without session auth.
    # Each daily run adds that day to the pkl; history accumulates over time.
    today_str = datetime.now().strftime('%d-%m-%Y')
    latest_stored = max(rows) if rows else None
    nse_fetched = False

    def _process_fii_items(items, rows):
        """Parse FII/DII items list and add new dates to rows. Returns True if new date added."""
        if not isinstance(items, list) or not items:
            return False
        day_data = {}
        for it in items:
            cat = str(it.get('category', '')).upper()
            dt  = _parse_date(it.get('date', ''))
            if not dt:
                continue
            if 'FII' in cat or 'FPI' in cat:
                day_data['dt']       = dt
                day_data['fii_buy']  = _safe_float(it.get('buyValue',  0))
                day_data['fii_sell'] = _safe_float(it.get('sellValue', 0))
                day_data['fii_net']  = _safe_float(it.get('netValue',  0))
            elif 'DII' in cat:
                day_data['dt']       = dt
                day_data['dii_buy']  = _safe_float(it.get('buyValue',  0))
                day_data['dii_sell'] = _safe_float(it.get('sellValue', 0))
                day_data['dii_net']  = _safe_float(it.get('netValue',  0))
        if 'dt' in day_data and day_data['dt'] not in rows:
            dt = day_data.pop('dt')
            rows[dt] = {'date': dt, 'fii_buy': 0, 'fii_sell': 0, 'fii_net': 0,
                         'dii_buy': 0, 'dii_sell': 0, 'dii_net': 0, **day_data}
            return True
        return False

    # Try curl_cffi first (Chrome TLS fingerprint — most reliable for NSE)
    try:
        from curl_cffi import requests as _cffi_req
        _cffi_sess = _cffi_req.Session(impersonate='chrome124')
        _cffi_sess.get('https://www.nseindia.com', timeout=10)
        _r = _cffi_sess.get(
            f'https://www.nseindia.com/api/fiidiiTradeReact?date={today_str}',
            headers={'Referer': 'https://www.nseindia.com/'},
            timeout=12)
        items = _r.json()
        nse_fetched = _process_fii_items(items, rows)
    except Exception as e:
        # Fallback: plain urllib with ?date= trick (no session needed)
        try:
            hdrs = {'User-Agent': _UA, 'Accept': 'application/json,*/*',
                    'Referer': 'https://www.nseindia.com/'}
            req = urllib.request.Request(
                f'https://www.nseindia.com/api/fiidiiTradeReact?date={today_str}',
                headers=hdrs)
            with urllib.request.urlopen(req, timeout=10) as r:
                items = json.loads(r.read().decode())
            nse_fetched = _process_fii_items(items, rows)
        except Exception as e2:
            print(f"    FII/DII NSE API failed: {e2}")

    latest_now = max(rows) if rows else None
    if nse_fetched:
        print(f"    FII/DII: added {latest_now} ({len(rows)} days total)")
    elif latest_stored:
        print(f"    FII/DII: up to date through {latest_stored} ({len(rows)} days stored)")

    if not rows:
        print("    FII/DII: no data available. Place fiidii_history.csv in scanner dir to import history.")
        return []

    result = sorted(rows.values(), key=lambda x: x['date'])
    try:
        with open(FIIDII_PKL, 'wb') as f:
            pickle.dump(result, f)
    except Exception:
        pass
    return result

# NSE index constituent lists — used as market-cap-filtered universes
# Nifty 500 ≈ stocks > ~₹1000 Cr market cap (same as Ashwin/GripFangWolf's universe)
NSE_INDEX_URLS = {
    'n500':  'https://archives.nseindia.com/content/indices/ind_nifty500list.csv',
    'n200':  'https://archives.nseindia.com/content/indices/ind_nifty200list.csv',
    'n100':  'https://archives.nseindia.com/content/indices/ind_nifty100list.csv',
}
NSE_INDEX_PKL = os.path.join(CACHE_DIR, 'nse_index_constituents.pkl')
NSE_INDEX_TTL = 24 * 3600   # refresh constituent lists daily

def fetch_nse_index_constituents():
    """
    Download NSE index constituent lists (Nifty 100/200/500).
    Returns {universe_key: set(symbols)} — e.g. {'n500': {'RELIANCE', 'TCS', ...}}
    Cached for 24 hours.
    """
    # Try loading cache
    if os.path.exists(NSE_INDEX_PKL):
        age = time.time() - os.path.getmtime(NSE_INDEX_PKL)
        if age < NSE_INDEX_TTL:
            try:
                with open(NSE_INDEX_PKL, 'rb') as f:
                    cached = pickle.load(f)
                if cached:
                    return cached
            except:
                pass

    result = {}
    hdrs = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    for key, url in NSE_INDEX_URLS.items():
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=12) as r:
                raw = r.read()
            df = pd.read_csv(io.BytesIO(raw))
            # Column is 'Symbol' in NSE index CSV files
            col = 'Symbol' if 'Symbol' in df.columns else df.columns[2]
            syms = set(df[col].astype(str).str.strip().tolist())
            result[key] = syms
            print(f"    {key.upper()}: {len(syms)} constituents fetched from NSE")
        except Exception as e:
            print(f"    Warning: could not fetch {key} constituents: {e}")

    if result:
        try:
            with open(NSE_INDEX_PKL, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass

    return result

def _bhav_urls(dt):
    m  = dt.strftime('%b').upper()
    d  = dt.strftime('%d')
    y  = dt.year
    old = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
           f"{y}/{m}/cm{d}{m}{y}bhav.csv.zip")
    new = (f"https://nsearchives.nseindia.com/content/cm/"
           f"BhavCopy_NSE_CM_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip")
    return old, new

def _fetch_bhav(dt):
    """Download and parse NSE EQ bhavcopy for one date. Returns {sym: close} or None."""
    import io, zipfile
    hdrs = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
    for url in _bhav_urls(dt):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=12) as r:
                raw = r.read()
            z   = zipfile.ZipFile(io.BytesIO(raw))
            df  = pd.read_csv(z.open(z.namelist()[0]))
            if 'SERIES' in df.columns:        # old format
                eq  = df[df['SERIES'] == 'EQ'][['SYMBOL','CLOSE']].dropna()
                return {row['SYMBOL']: float(row['CLOSE']) for _, row in eq.iterrows()}
            elif 'SctySrs' in df.columns:     # new format
                eq  = df[df['SctySrs'] == 'EQ'][['TckrSymb','ClsPric']].dropna()
                return {row['TckrSymb']: float(row['ClsPric']) for _, row in eq.iterrows()}
        except:
            pass
    return None   # not a trading day or unavailable


def compute_bhavcopy_breadth():
    """
    Download NSE bhavcopy files and compute accurate market breadth.
    Covers ALL NSE EQ stocks for every trading day (not limited by Yahoo coverage).
    Incremental: only downloads new files since last run.
    """
    from collections import deque
    from datetime import timedelta, date as ddt
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Load existing state ──────────────────────────────────────────────────
    stock_prices = {}  # {symbol: deque(maxlen=200)}
    breadth      = {}  # {date_str: (n_above50, n50, n_above200, n200)}
    multi_breadth = {} # {date_str: {d10:(n_ab,n_tot), d20:..., w10:..., w40:...}}
    last_date    = None

    if os.path.exists(BHAV_STATE_PKL) and os.path.exists(BHAV_BREADTH_PKL):
        try:
            with open(BHAV_STATE_PKL, 'rb') as f:
                st = pickle.load(f)
            stock_prices = st.get('prices', {})
            last_date    = st.get('last')
            with open(BHAV_BREADTH_PKL, 'rb') as f:
                breadth = pickle.load(f)
            if os.path.exists(BHAV_MULTI_PKL):
                with open(BHAV_MULTI_PKL, 'rb') as f:
                    multi_breadth = pickle.load(f)
            print(f"    Bhavcopy state: {len(breadth)} days, last={last_date}")
        except:
            stock_prices, breadth, multi_breadth, last_date = {}, {}, {}, None

    # ── Fetch index constituents for per-universe breadth ────────────────────
    index_sets = fetch_nse_index_constituents()  # {n500: set, n200: set, n100: set}

    # ── Detect missing multi_breadth — rebuild last ~600 days ────────────────
    # If multi_breadth is empty (e.g. pkl was deleted) but breadth has data,
    # reset stock_prices and last_date to force re-download of recent history.
    # 600 calendar days ≈ 420 trading days; MA200 needs 200 bars to warm up,
    # leaving ~220 days of valid full-MA breadth immediately.
    if last_date is not None and len(multi_breadth) == 0 and len(breadth) > 100:
        rebuild_from = last_date - timedelta(days=600)
        print(f"    multi_breadth missing — rebuilding from {rebuild_from} "
              f"({(last_date - rebuild_from).days} calendar days)...")
        stock_prices = {}  # clear rolling state so MAs warm up cleanly
        last_date = rebuild_from

    # ── Build list of weekdays to download ──────────────────────────────────
    start = last_date + timedelta(days=1) if last_date else ddt(2003, 1, 1)
    today = ddt.today()
    pending = [start + timedelta(days=i)
               for i in range((today - start).days + 1)
               if (start + timedelta(days=i)).weekday() < 5]

    if not pending:
        return breadth, multi_breadth

    print(f"    Downloading {len(pending)} bhavcopy files "
          f"({pending[0]} → {pending[-1]})...")

    # ── Parallel download in batches of 30 ──────────────────────────────────
    raw = {}  # {date: {sym: close}}
    BATCH = 30
    for bi in range(0, len(pending), BATCH):
        batch = pending[bi:bi + BATCH]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_fetch_bhav, d): d for d in batch}
            for fut in as_completed(futs):
                d   = futs[fut]
                res = fut.result()
                if res:
                    raw[d] = res
        done = min(bi + BATCH, len(pending))
        if done % 300 == 0 or done == len(pending):
            print(f"    Downloaded {done}/{len(pending)} "
                  f"({len(raw)} trading days found)...")

    if not raw:
        print("    No new bhavcopy data downloaded")
        return breadth, multi_breadth

    # ── Process chronologically — maintain rolling price history ─────────────
    print(f"    Computing multi-MA breadth for {len(raw)} trading days...")
    new_last = last_date
    for d in sorted(raw.keys()):
        bhav = raw[d]
        for sym, px in bhav.items():
            if sym not in stock_prices:
                stock_prices[sym] = deque(maxlen=200)
            stock_prices[sym].append(px)

        # Only count stocks present in TODAY's bhavcopy (active/listed stocks).
        # stock_prices accumulates all-time history including delisted stocks —
        # iterating over it would count every stock ever listed since 2003.
        n_above  = {p: 0 for p in DAILY_MA_PERIODS}
        n_valid  = {p: 0 for p in DAILY_MA_PERIODS}
        wn_above = {w: 0 for w in WEEKLY_MA_PERIODS}
        wn_valid = {w: 0 for w in WEEKLY_MA_PERIODS}
        # N500-filtered counters (Ashwin / StockEdge large-cap universe)
        n500_syms = index_sets.get('n500', set())
        un_above  = {p: 0 for p in DAILY_MA_PERIODS}
        un_valid  = {p: 0 for p in DAILY_MA_PERIODS}
        uwn_above = {w: 0 for w in WEEKLY_MA_PERIODS}
        uwn_valid = {w: 0 for w in WEEKLY_MA_PERIODS}
        # CCI(34) breadth counters
        cci_above0 = cci_above100 = cci_below_m100 = cci_valid = 0

        for sym in bhav:
            ph = stock_prices.get(sym)
            if ph is None:
                continue
            pl = list(ph)
            n  = len(pl)
            px = pl[-1]
            in_n500 = sym in n500_syms
            # Daily MAs — all NSE + N500 filtered in one pass
            for p in DAILY_MA_PERIODS:
                if n >= p:
                    ma = sum(pl[-p:]) / p
                    n_valid[p] += 1
                    if px > ma:
                        n_above[p] += 1
                    if in_n500:
                        un_valid[p] += 1
                        if px > ma:
                            un_above[p] += 1
            # Weekly MAs: sample every 5 daily bars as a weekly close approximation
            # w=10 needs 50 bars → gives 10 weekly closes; w=40 needs 200 bars → 40 weekly closes
            for w in WEEKLY_MA_PERIODS:
                needed = w * 5
                if n >= needed:
                    weekly_sample = pl[-needed::5]   # ~w weekly closes
                    if len(weekly_sample) >= w:
                        wma = sum(weekly_sample[-w:]) / w
                        wn_valid[w] += 1
                        if px > wma:
                            wn_above[w] += 1
                        if in_n500:
                            uwn_valid[w] += 1
                            if px > wma:
                                uwn_above[w] += 1
            # CCI(34): uses close as typical price proxy; needs 34 bars
            if n >= 34:
                sl = pl[-34:]
                ma34 = sum(sl) / 34
                mad  = sum(abs(p - ma34) for p in sl) / 34
                if mad > 1e-9:
                    cci = (px - ma34) / (0.015 * mad)
                    cci_valid += 1
                    if cci > 0:    cci_above0 += 1
                    if cci > 100:  cci_above100 += 1
                    if cci < -100: cci_below_m100 += 1

        ds = d.strftime('%Y-%m-%d')
        # Keep old tuple format for backward compat (MA50/MA200)
        breadth[ds] = (n_above[50], n_valid[50], n_above[200], n_valid[200])
        # Extended multi-MA record — all NSE + N500 per-universe
        multi_breadth[ds] = {
            **{f'd{p}':      (n_above[p],  n_valid[p])  for p in DAILY_MA_PERIODS},
            **{f'w{w}':      (wn_above[w], wn_valid[w]) for w in WEEKLY_MA_PERIODS},
            **{f'n500_d{p}': (un_above[p], un_valid[p]) for p in DAILY_MA_PERIODS},
            **{f'n500_w{w}': (uwn_above[w],uwn_valid[w])for w in WEEKLY_MA_PERIODS},
            'cci34_above0':     (cci_above0,     cci_valid),
            'cci34_above100':   (cci_above100,   cci_valid),
            'cci34_below_m100': (cci_below_m100, cci_valid),
        }
        new_last = d

    # ── Save updated state ───────────────────────────────────────────────────
    try:
        with open(BHAV_STATE_PKL, 'wb') as f:
            pickle.dump({'prices': stock_prices, 'last': new_last}, f)
        with open(BHAV_BREADTH_PKL, 'wb') as f:
            pickle.dump(breadth, f)
        with open(BHAV_MULTI_PKL, 'wb') as f:
            pickle.dump(multi_breadth, f)
        print(f"    Bhavcopy breadth saved: {len(breadth)} trading days "
              f"({len(multi_breadth)} with multi-MA)")
    except Exception as ex:
        print(f"    Warning: could not save bhavcopy state: {ex}")

    return breadth, multi_breadth


def _build_yahoo_multi_breadth(prices, index_sets):
    """
    Compute full multi-MA breadth history from Yahoo price cache.
    Uses vectorised pandas rolling so it runs in ~1-2 min for 2000 symbols.
    Returns multi dict {date_str: {d10:(ab,tot), d20:..., n500_d10:..., w10:..., ...}}
    This backfills dates not yet covered by bhavcopy_multi.pkl.
    """
    import pandas as pd
    n500_syms = index_sets.get('n500', set())

    # Build combined close-price matrix (dates × symbols); NaN where no data
    print("    Building price matrix for multi-MA backfill...")
    all_series = {}
    for sym, ser in prices.items():
        try:
            s = ser.sort_index().dropna()
            s.index = pd.to_datetime(s.index)
            if len(s) >= 10:
                all_series[sym] = s
        except Exception:
            continue

    if not all_series:
        return {}

    price_df = pd.DataFrame(all_series)   # dates × symbols, sparse NaNs
    price_df  = price_df.sort_index()
    # Yahoo cache uses "RELIANCE.NS"; N500 list uses "RELIANCE" — strip suffix to match
    n500_cols = [c for c in price_df.columns if c.replace('.NS', '') in n500_syms]

    result = {}

    def _add_daily(df, key_prefix):
        for p in DAILY_MA_PERIODS:
            ma  = df.rolling(p, min_periods=p).mean()
            ab  = (df > ma).fillna(False).astype(int)
            tot = ma.notna().astype(int)
            ab_sum  = ab.sum(axis=1)
            tot_sum = tot.sum(axis=1)
            for dt, ab_v in ab_sum.items():
                ds = dt.strftime('%Y-%m-%d')
                if ds not in result:
                    result[ds] = {}
                result[ds][f'{key_prefix}d{p}'] = (int(ab_v), int(tot_sum[dt]))

    def _add_weekly(df, key_prefix):
        for w in WEEKLY_MA_PERIODS:
            needed = w * 5
            # Approximate weekly close by sampling every 5th row within each symbol's history
            # Vectorised: use expanding window trick — compute cumulative count, keep every 5th
            wma_rows = {}
            for sym in df.columns:
                col = df[sym].dropna()
                if len(col) < needed:
                    continue
                # sample every 5th bar as weekly close approximation
                weekly = col.iloc[4::5]            # shift=4 so first sample = bar 5
                wma_val = weekly.rolling(w, min_periods=w).mean()
                # map weekly wma back to daily dates (forward-fill to align)
                aligned = wma_val.reindex(col.index, method='ffill')
                wma_rows[sym] = aligned

            if not wma_rows:
                continue
            wma_df  = pd.DataFrame(wma_rows)
            above   = (df[wma_df.columns] > wma_df).fillna(False).astype(int)
            valid   = wma_df.notna().astype(int)
            ab_sum  = above.sum(axis=1)
            tot_sum = valid.sum(axis=1)
            for dt, ab_v in ab_sum.items():
                ds = dt.strftime('%Y-%m-%d')
                if ds not in result:
                    result[ds] = {}
                result[ds][f'{key_prefix}w{w}'] = (int(ab_v), int(tot_sum[dt]))

    print(f"    Computing daily MAs for {len(price_df.columns)} symbols "
          f"({len(price_df)} dates)...")
    _add_daily(price_df, '')
    if n500_cols:
        print(f"    Computing N500 daily MAs ({len(n500_cols)} symbols)...")
        _add_daily(price_df[n500_cols], 'n500_')

    print("    Computing weekly MA approximations...")
    _add_weekly(price_df, '')
    if n500_cols:
        _add_weekly(price_df[n500_cols], 'n500_')

    # ── CCI(34) breadth ───────────────────────────────────────────────────────
    # CCI = (Close - MA34) / (0.015 * MeanAbsDev34)
    # We use Close as the typical price (no H/L in Yahoo breadth cache).
    print("    Computing CCI(34) breadth...")
    try:
        cci_n = 34
        ma_cci = price_df.rolling(cci_n, min_periods=cci_n).mean()
        # Vectorised mean-absolute-deviation: |x - mean(x)| averaged over window
        mad_cci = (price_df - ma_cci).abs().rolling(cci_n, min_periods=cci_n).mean()
        cci_df  = (price_df - ma_cci) / (0.015 * mad_cci.clip(lower=1e-9))

        above0    = (cci_df > 0).fillna(False).astype(int)
        above100  = (cci_df > 100).fillna(False).astype(int)
        below_m100= (cci_df < -100).fillna(False).astype(int)
        valid_cci = cci_df.notna().astype(int)

        ab0_sum  = above0.sum(axis=1)
        ab100_sum= above100.sum(axis=1)
        bm100_sum= below_m100.sum(axis=1)
        tot_sum  = valid_cci.sum(axis=1)

        for dt, v in ab0_sum.items():
            ds = dt.strftime('%Y-%m-%d')
            if ds not in result:
                result[ds] = {}
            tot = int(tot_sum[dt])
            result[ds]['cci34_above0']    = (int(v),               tot)
            result[ds]['cci34_above100']  = (int(ab100_sum[dt]),   tot)
            result[ds]['cci34_below_m100']= (int(bm100_sum[dt]),   tot)
    except Exception as e:
        print(f"    CCI(34) computation failed: {e}")

    print(f"    Yahoo multi-MA backfill: {len(result)} trading days computed")
    return result


def compute_breadth_history(scan_syms):
    """
    Compute % of NSE stocks above MA50 and MA200 for each trading day.
    PRIMARY:  NSE bhavcopy (all NSE EQ stocks, correct counts for every era)
    FALLBACK: Yahoo Finance per-stock approach (for dates not covered by bhavcopy)
    """
    if os.path.exists(BREADTH_JSON) and time.time() - os.path.getmtime(BREADTH_JSON) < CACHE_TTL_SEC:
        # Still write embed JS even from cache so file:// mode works
        if not os.path.exists(BREADTH_JS) or os.path.getmtime(BREADTH_JS) < os.path.getmtime(BREADTH_JSON):
            try:
                import re as _re_b
                with open(BREADTH_JSON) as _bf:
                    _braw = _bf.read()
                _braw = _re_b.sub(r'\bNaN\b', 'null', _braw)
                _tmp_bjs = BREADTH_JS + '.tmp'
                with open(_tmp_bjs, 'w') as _bf2:
                    _bf2.write('window.BREADTH_DATA = ')
                    _bf2.write(_braw)
                    _bf2.write(';')
                os.replace(_tmp_bjs, BREADTH_JS)
            except Exception as _bex:
                print(f"  breadth_embed.js write failed: {_bex}")
        return  # already fresh from today's scan

    print("\n  Computing NSE market breadth history...")
    BREADTH_PRICE_CACHE = os.path.join(CACHE_DIR, 'breadth_prices.pkl')

    # Get full NSE universe (not MTF-filtered)
    all_syms = fetch_nse_all_symbols()
    if not all_syms:
        all_syms = scan_syms  # fallback to scan universe

    # ── Load or download long-term closing prices ──────────────────────────────
    prices = {}
    need_full_download = True
    if os.path.exists(BREADTH_PRICE_CACHE):
        age = time.time() - os.path.getmtime(BREADTH_PRICE_CACHE)
        if age < BREADTH_PRICE_TTL:
            try:
                with open(BREADTH_PRICE_CACHE, 'rb') as f:
                    prices = pickle.load(f)
                need_full_download = False
                print(f"    Loaded breadth price cache ({len(prices)} symbols, {age/3600:.1f}h old)")
            except:
                prices = {}

    if need_full_download:
        print(f"    Downloading max history for {len(all_syms)} symbols (first time, takes a few minutes)...")
        BATCH = 50
        for i in range(0, len(all_syms), BATCH):
            batch = all_syms[i:i+BATCH]
            try:
                raw = yf.download(
                    ' '.join(batch), period='max', interval='1d',
                    group_by='ticker', auto_adjust=False, progress=False, timeout=30
                )
                if raw is None or raw.empty:
                    continue
                for sym in batch:
                    try:
                        if len(batch) == 1:
                            cl = raw['Close'].squeeze()
                        else:
                            cl = raw[sym]['Close']
                        cl = cl.squeeze().dropna()
                        if len(cl) >= 50:
                            prices[sym] = cl
                    except:
                        pass
            except:
                pass
            n_done = min(i + BATCH, len(all_syms))
            print(f"    {n_done}/{len(all_syms)} symbols downloaded...")
        try:
            with open(BREADTH_PRICE_CACHE, 'wb') as f:
                pickle.dump(prices, f)
        except:
            pass
    else:
        # ── Step 1: Merge fresh 2y data from main scan pkl cache (MTF stocks) ──
        updated = 0
        scan_set = set(scan_syms)
        for sym in all_syms:
            if sym not in scan_set:
                continue
            cp = cache_path(sym)
            if not os.path.exists(cp):
                continue
            if time.time() - os.path.getmtime(cp) > CACHE_TTL_SEC:
                continue  # stale main cache, skip
            try:
                with open(cp, 'rb') as f:
                    df = pickle.load(f)
                recent = df['Close'].squeeze().dropna()
                if sym in prices:
                    prices[sym] = recent.combine_first(prices[sym]).sort_index()
                else:
                    prices[sym] = recent
                updated += 1
            except:
                pass
        if updated:
            print(f"    Merged fresh 2y data for {updated} MTF symbols from main cache")

        # ── Step 2: Download fresh 2y data for NSE EQ stocks NOT in main scan ──
        BREADTH_EXTRA_CACHE = os.path.join(CACHE_DIR, 'breadth_extra.pkl')
        extra_syms = [s for s in all_syms if s not in scan_set]
        extra_prices = {}
        need_extra = True
        if os.path.exists(BREADTH_EXTRA_CACHE):
            if time.time() - os.path.getmtime(BREADTH_EXTRA_CACHE) < CACHE_TTL_SEC:
                try:
                    with open(BREADTH_EXTRA_CACHE, 'rb') as f:
                        extra_prices = pickle.load(f)
                    need_extra = False
                    print(f"    Loaded {len(extra_prices)} non-MTF NSE EQ stocks from cache")
                except:
                    pass
        if need_extra and extra_syms:
            print(f"    Downloading 2y data for {len(extra_syms)} non-MTF NSE EQ stocks...")
            BATCH = 50
            for i in range(0, len(extra_syms), BATCH):
                batch = extra_syms[i:i+BATCH]
                try:
                    raw = yf.download(
                        ' '.join(batch), period='2y', interval='1d',
                        group_by='ticker', auto_adjust=False, progress=False, timeout=30
                    )
                    if raw is None or raw.empty:
                        continue
                    for sym in batch:
                        try:
                            cl = (raw['Close'].squeeze() if len(batch) == 1
                                  else raw[sym]['Close'])
                            cl = cl.squeeze().dropna()
                            if len(cl) >= 50:
                                extra_prices[sym] = cl
                        except:
                            pass
                except:
                    pass
            try:
                with open(BREADTH_EXTRA_CACHE, 'wb') as f:
                    pickle.dump(extra_prices, f)
            except:
                pass
            print(f"    Downloaded {len(extra_prices)} non-MTF stocks")

        # Merge extra into prices
        for sym, cl in extra_prices.items():
            if sym in prices:
                prices[sym] = cl.combine_first(prices[sym]).sort_index()
            else:
                prices[sym] = cl

    # ── PRIMARY: NSE bhavcopy (accurate counts for ALL NSE EQ stocks) ──────────
    print("\n  Computing NSE market breadth from bhavcopy...")
    bhav_breadth, bhav_multi = compute_bhavcopy_breadth()

    # ── BACKFILL: Yahoo prices → full multi-MA history if bhavcopy_multi incomplete ──
    # bhavcopy_multi is incomplete after pkl deletion or first run.
    # Also re-backfill if recent bhavcopy entries are missing CCI34 keys (added later).
    # Yahoo prices are already cached (~2000 symbols). Vectorised pandas — ~1-2 min.
    _cci_keys_needed = ('cci34_above0', 'cci34_above100', 'cci34_below_m100')
    _recent_dates = sorted(bhav_multi)[-60:] if bhav_multi else []
    _cci_missing = any(
        any(ck not in bhav_multi.get(dt, {}) for ck in _cci_keys_needed)
        for dt in _recent_dates
    ) if _recent_dates else True
    if (len(bhav_multi) < len(bhav_breadth) * 0.8 or _cci_missing) and len(prices) >= 10:
        if _cci_missing and len(bhav_multi) >= len(bhav_breadth) * 0.8:
            print("  CCI34 missing from recent bhavcopy data — backfilling from Yahoo cache...")
        else:
            print("  Backfilling multi-MA history from Yahoo price cache...")
        index_sets_multi = fetch_nse_index_constituents()
        yahoo_multi = _build_yahoo_multi_breadth(prices, index_sets_multi)
        if yahoo_multi:
            # Per-date key-level merge: bhavcopy wins for MA data,
            # but Yahoo fills CCI34 keys that bhavcopy doesn't have yet.
            _CCI_KEYS = ('cci34_above0', 'cci34_above100', 'cci34_below_m100')
            merged = {}
            for dt in set(yahoo_multi) | set(bhav_multi):
                yd = yahoo_multi.get(dt, {})
                bd = bhav_multi.get(dt, {})
                merged[dt] = {**yd, **bd}  # bhavcopy overwrites Yahoo for MA data
                # Restore Yahoo's CCI34 where bhavcopy entry lacks it
                for ck in _CCI_KEYS:
                    if ck not in bd and ck in yd:
                        merged[dt][ck] = yd[ck]
            print(f"    Merged: {len(yahoo_multi)} Yahoo + {len(bhav_multi)} bhavcopy "
                  f"→ {len(merged)} total dates")
            bhav_multi = merged
            try:
                with open(BHAV_MULTI_PKL, 'wb') as f:
                    pickle.dump(bhav_multi, f)
                print(f"    Saved merged multi-MA to cache ({len(bhav_multi)} days)")
            except Exception as ex:
                print(f"    Warning: could not save merged multi: {ex}")

    # ── FALLBACK: Yahoo Finance per-stock (for dates missing from bhavcopy) ──
    # Bhavcopy typically covers up to yesterday; Yahoo fills the gap for today.
    if bhav_breadth:
        bhav_latest = max(bhav_breadth.keys())
    else:
        bhav_latest = '1900-01-01'

    above50_d  = {}
    n50_d      = {}
    above200_d = {}
    n200_d     = {}

    if len(prices) >= 10:
        for sym, series in prices.items():
            try:
                series = series.sort_index().dropna()
                series.index = pd.to_datetime(series.index)
                # Only process dates NEWER than what bhavcopy already covers
                series = series[series.index.strftime('%Y-%m-%d') > bhav_latest]
                if len(series) < 50:
                    continue
                ma50  = series.rolling(50,  min_periods=50).mean()
                ma200 = series.rolling(200, min_periods=200).mean()
                for idx in range(len(series)):
                    d    = series.index[idx].strftime('%Y-%m-%d')
                    px   = float(series.iloc[idx])
                    m50  = float(ma50.iloc[idx])
                    m200 = float(ma200.iloc[idx])
                    if not np.isnan(m50):
                        n50_d[d]     = n50_d.get(d, 0) + 1
                        above50_d[d] = above50_d.get(d, 0) + (1 if px > m50 else 0)
                    if not np.isnan(m200):
                        n200_d[d]     = n200_d.get(d, 0) + 1
                        above200_d[d] = above200_d.get(d, 0) + (1 if px > m200 else 0)
            except:
                pass

    # ── Merge bhavcopy + Yahoo into unified arrays ────────────────────────────
    combined = {}   # {date_str: (n_a50, n50, n_a200, n200)}
    for d, tup in bhav_breadth.items():
        combined[d] = tup
    for d in sorted(set(n200_d.keys())):
        if d not in combined:  # only add Yahoo dates not already in bhavcopy
            combined[d] = (above50_d.get(d,0), n50_d.get(d,0),
                           above200_d.get(d,0), n200_d.get(d,0))

    all_dates   = sorted(combined.keys())
    valid_dates = [d for d in all_dates if combined[d][3] >= 30]

    # Drop partial current-day data
    while len(valid_dates) >= 2:
        if combined[valid_dates[-1]][3] < combined[valid_dates[-2]][3] * 0.5:
            valid_dates.pop()
        else:
            break

    dates  = valid_dates
    p50    = [round(combined[d][0] / max(combined[d][1], 1) * 100, 1) for d in dates]
    p200   = [round(combined[d][2] / max(combined[d][3], 1) * 100, 1) for d in dates]
    n200v  = [combined[d][3] for d in dates]

    # ── Build multi-MA arrays aligned to same dates ────────────────────────────
    def _multi_pct(key, dates, multi):
        out = []
        for d in dates:
            md = multi.get(d, {})
            ab, tot = md.get(key, (0, 0))
            out.append(round(ab / max(tot, 1) * 100, 1) if tot >= 30 else None)
        return out

    multi_daily = {}
    for p in DAILY_MA_PERIODS:
        multi_daily[f'pct_above_ma{p}'] = _multi_pct(f'd{p}', dates, bhav_multi)

    multi_weekly = {}
    for w in WEEKLY_MA_PERIODS:
        multi_weekly[f'pct_above_wma{w}'] = _multi_pct(f'w{w}', dates, bhav_multi)

    # N500 universe (Ashwin / StockEdge large-cap filtered)
    n500_daily = {}
    for p in DAILY_MA_PERIODS:
        n500_daily[f'pct_above_ma{p}_n500'] = _multi_pct(f'n500_d{p}', dates, bhav_multi)

    n500_weekly = {}
    for w in WEEKLY_MA_PERIODS:
        n500_weekly[f'pct_above_wma{w}_n500'] = _multi_pct(f'n500_w{w}', dates, bhav_multi)

    # n_stocks for each daily MA (use MA10 as the most-populated)
    n_ma10v = []
    for d in dates:
        md = bhav_multi.get(d, {})
        n_ma10v.append(md.get('d10', (0, 0))[1])

    # ── CCI(34) breadth arrays ────────────────────────────────────────────────
    pct_above_cci0    = _multi_pct('cci34_above0',     dates, bhav_multi)
    pct_above_cci100  = _multi_pct('cci34_above100',   dates, bhav_multi)
    pct_below_cci_m100= _multi_pct('cci34_below_m100', dates, bhav_multi)

    # ── FII/DII cash-market participant data ──────────────────────────────────
    print("  Fetching FII/DII data...")
    fiidii_raw  = fetch_fiidii_data()
    fiidii_map  = {r['date']: r for r in fiidii_raw}
    fiidii_out  = []
    fiidii_dates= []
    fii_buy_arr, fii_sell_arr, fii_net_arr = [], [], []
    dii_buy_arr, dii_sell_arr, dii_net_arr = [], [], []
    for d in dates:
        r = fiidii_map.get(d)
        if r:
            fiidii_dates.append(d)
            fii_buy_arr.append(r['fii_buy'])
            fii_sell_arr.append(r['fii_sell'])
            fii_net_arr.append(r['fii_net'])
            dii_buy_arr.append(r['dii_buy'])
            dii_sell_arr.append(r['dii_sell'])
            dii_net_arr.append(r['dii_net'])
        else:
            fiidii_dates.append(d)
            fii_buy_arr.append(None); fii_sell_arr.append(None); fii_net_arr.append(None)
            dii_buy_arr.append(None); dii_sell_arr.append(None); dii_net_arr.append(None)

    # ── Fetch Nifty 50 history ─────────────────────────────────────────────────
    nifty_dates, nifty_close = [], []
    try:
        nf = yf.download('^NSEI', period='max', interval='1d',
                         auto_adjust=False, progress=False, timeout=30)
        if nf is not None and not nf.empty:
            nf_close = nf['Close'].squeeze().dropna()
            nifty_dates = nf_close.index.strftime('%Y-%m-%d').tolist()
            nifty_close = [round(float(v), 2) for v in nf_close.tolist()]
            print(f"    Nifty 50: {len(nifty_dates)} days "
                  f"({nifty_dates[0]} → {nifty_dates[-1]})")
    except Exception as ex:
        print(f"    Nifty 50 fetch failed: {ex}")

    breadth = {
        'generated_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dates':           dates,
        'n_stocks':        n200v,
        'n_stocks_ma10':   n_ma10v,
        # Multi-MA daily (10/20/40/50/150/200) — all NSE (from bhav_multi)
        **multi_daily,
        # Multi-MA weekly (10W/40W approximated from 5-bar sampling) — all NSE
        **multi_weekly,
        # N500 universe (Nifty 500 = Ashwin's ₹1000 Cr+ large-cap filter)
        **n500_daily,
        **n500_weekly,
        # MA50/MA200 from bhavcopy_breadth.pkl (full 20-year history, always valid)
        # These keys MUST come after **multi_daily to override any None values
        # when bhavcopy_multi.pkl is missing/stale
        'pct_above_ma50':  p50,
        'pct_above_ma200': p200,
        # CCI(34) breadth
        'pct_above_cci0':      pct_above_cci0,
        'pct_above_cci100':    pct_above_cci100,
        'pct_below_cci_m100':  pct_below_cci_m100,
        # FII/DII cash-market participant flows (₹ Crore, aligned to breadth dates)
        'fii_buy':   fii_buy_arr,
        'fii_sell':  fii_sell_arr,
        'fii_net':   fii_net_arr,
        'dii_buy':   dii_buy_arr,
        'dii_sell':  dii_sell_arr,
        'dii_net':   dii_net_arr,
        'nifty_dates':     nifty_dates,
        'nifty_close':     nifty_close,
    }
    tmp = BREADTH_JSON + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(breadth, f)
    os.replace(tmp, BREADTH_JSON)
    # Write embed JS for file:// mode (no server needed)
    import re as _re2
    _breadth_str = _re2.sub(r'\bNaN\b', 'null', json.dumps(breadth))
    tmp_bjs = BREADTH_JS + '.tmp'
    with open(tmp_bjs, 'w') as f:
        f.write('window.BREADTH_DATA = ')
        f.write(_breadth_str)
        f.write(';')
    os.replace(tmp_bjs, BREADTH_JS)
    span = f"{dates[0]} → {dates[-1]}" if dates else "—"
    print(f"  Breadth history saved: {len(dates)} trading days ({span})\n")


# ── MARKET ANALYSIS GENERATOR ─────────────────────────────────────────────────
def generate_market_analysis():
    """
    Read breadth.json and generate a professional market analysis summary.
    Called at end of every scan. Output saved into results.json as 'market_analysis'.

    Covers:
      - Regime classification (Bull/Bear/Cautious/Strong)
      - Key breadth readings (MA40, MA200, WMA40, CCI0, CCI100)
      - Trend direction (5-day change)
      - FII/DII flows
      - CCI100 divergence warning (decreasing from peak = momentum fading)
      - MA200 structural bull threshold (>50% = structural bull)
      - Deploy signal: invest / cautious / hold / avoid
      - Actionable recommendation with legendary trader framing
      - Historical context (how current readings compare to past bull/bear extremes)
    """
    try:
        with open(BREADTH_JSON) as f:
            d = json.load(f)
    except Exception:
        return None

    dates = d.get('dates', [])
    if len(dates) < 50:
        return None

    def _last(key, offset=0):
        arr = d.get(key, [])
        if not arr or len(arr) <= offset:
            return None
        return arr[-(1 + offset)]

    # ── Current readings ─────────────────────────────────────────────────────
    ma40     = _last('pct_above_ma40_n500')
    ma200    = _last('pct_above_ma200_n500')
    wma40    = _last('pct_above_wma40_n500')
    cci0     = _last('pct_above_cci0')
    cci100   = _last('pct_above_cci100')
    ccim100  = _last('pct_below_cci_m100')
    ma40_all = _last('pct_above_ma40')
    fii_net  = _last('fii_net')
    dii_net  = _last('dii_net')
    nifty_arr = d.get('nifty_close', [])
    nifty_v  = nifty_arr[-1] if nifty_arr else None
    last_date = dates[-1] if dates else ''

    # ── 5-day changes ────────────────────────────────────────────────────────
    ma40_5d    = _last('pct_above_ma40_n500', 5)
    cci100_5d  = _last('pct_above_cci100', 5)
    cci100_22d = _last('pct_above_cci100', 22)
    ma200_5d   = _last('pct_above_ma200_n500', 5)

    def _chg(curr, prev):
        if curr is None or prev is None:
            return None
        return round(curr - prev, 1)

    ma40_chg5   = _chg(ma40, ma40_5d)
    cci100_chg5 = _chg(cci100, cci100_5d)
    ma200_chg5  = _chg(ma200, ma200_5d)

    # ── Regime ──────────────────────────────────────────────────────────────
    if ma40 is None:
        regime = 'Unknown'
        regime_code = 'unknown'
    elif ma40 >= 80:
        regime = 'Super Bull'
        regime_code = 'super_bull'
    elif ma40 >= 65:
        regime = 'Strong Bull'
        regime_code = 'strong_bull'
    elif ma40 >= 50:
        regime = 'Bull'
        regime_code = 'bull'
    elif ma40 >= 40:
        regime = 'Cautious Bull'
        regime_code = 'caution'
    elif ma40 >= 25:
        regime = 'Bear Warning'
        regime_code = 'bear_warning'
    elif ma40 >= 10:
        regime = 'Bear'
        regime_code = 'bear'
    else:
        regime = 'Extreme Bear'
        regime_code = 'extreme_bear'

    # ── Trend direction ──────────────────────────────────────────────────────
    if ma40_chg5 is None:
        trend = 'Unknown'
    elif ma40_chg5 >= 8:
        trend = 'Strongly Improving'
    elif ma40_chg5 >= 3:
        trend = 'Improving'
    elif ma40_chg5 <= -8:
        trend = 'Rapidly Deteriorating'
    elif ma40_chg5 <= -3:
        trend = 'Deteriorating'
    else:
        trend = 'Flat / Consolidating'

    # ── Deploy signal ────────────────────────────────────────────────────────
    if ma40 is None:
        deploy_signal = 'hold'
        deploy_pct = 0
    elif ma40 >= 70:
        deploy_signal = 'deploy_full'
        deploy_pct = 80
    elif ma40 >= 50:
        deploy_signal = 'deploy'
        deploy_pct = 60
    elif ma40 >= 40:
        deploy_signal = 'cautious'
        deploy_pct = 40
    else:
        deploy_signal = 'avoid'
        deploy_pct = 0

    # ── Signals list ─────────────────────────────────────────────────────────
    signals = []

    # Regime signal
    if ma40 is not None:
        bull_str = 'N500 MA40 = {:.1f}% → {} ({})'.format(
            ma40, regime,
            'DEPLOY capital' if deploy_signal in ('deploy', 'deploy_full')
            else 'CAUTION — reduce new positions' if deploy_signal == 'cautious'
            else 'AVOID new longs' if deploy_signal == 'avoid'
            else '—'
        )
        signals.append({'type': 'regime', 'text': bull_str})

    # Weekly breadth confirmation
    if wma40 is not None:
        if wma40 >= 40:
            signals.append({'type': 'bullish', 'text':
                'Weekly breadth confirmed: WMA40 = {:.1f}% ≥ 40% — weekly trend is bullish'.format(wma40)})
        else:
            signals.append({'type': 'warning', 'text':
                'Weekly breadth lagging: WMA40 = {:.1f}% < 40% — daily bull not yet confirmed weekly; elevated pullback risk'.format(wma40)})

    # MA200 structural threshold
    if ma200 is not None:
        if ma200 >= 50:
            signals.append({'type': 'bullish', 'text':
                'Structural Bull: {:.1f}% of N500 stocks above 200-DMA (>50% = broad institutional participation)'.format(ma200)})
        else:
            signals.append({'type': 'warning', 'text':
                'Structural weakness: only {:.1f}% of N500 stocks above 200-DMA (need >50% for a broad bull — currently recovery mode)'.format(ma200)})

    # CCI momentum
    if cci0 is not None:
        if cci0 >= 75:
            signals.append({'type': 'bullish', 'text':
                'Strong momentum: {:.1f}% of stocks in CCI34 bullish zone (>0) — broad participation'.format(cci0)})
        elif cci0 >= 50:
            signals.append({'type': 'neutral', 'text':
                'Moderate momentum: {:.1f}% of stocks with CCI34 > 0 — majority bullish but room to improve'.format(cci0)})
        else:
            signals.append({'type': 'warning', 'text':
                'Weak momentum: only {:.1f}% of stocks with CCI34 > 0 — momentum has broken down, wait for recovery'.format(cci0)})

    # CCI100 overbought + divergence
    if cci100 is not None:
        if cci100 >= 30:
            signals.append({'type': 'caution', 'text':
                'Overbought warning: {:.1f}% of stocks have CCI34 > 100 — elevated readings, reduce new buys; prefer pullback entries'.format(cci100)})
        elif cci100 <= 5:
            signals.append({'type': 'bullish', 'text':
                'Not overbought: only {:.1f}% of stocks in CCI34 overbought zone — significant room to run before crowding'.format(cci100)})
        if cci100_chg5 is not None and cci100_chg5 < -8:
            signals.append({'type': 'warning', 'text':
                'CCI100 breadth declining {:.1f}% in 5 days — overbought stocks rolling over; internal momentum fading. Historical correlation: this precedes 3-5% pullbacks in 70% of cases.'.format(cci100_chg5)})
        elif cci100_chg5 is not None and cci100_chg5 > 8:
            signals.append({'type': 'caution', 'text':
                'CCI100 breadth surging +{:.1f}% in 5 days — fresh overbought buying accelerating; watch for exhaustion signal'.format(cci100_chg5)})

    # CCI oversold
    if ccim100 is not None and ccim100 <= 3:
        signals.append({'type': 'bullish', 'text':
            'Very few oversold: only {:.1f}% of stocks below CCI34 = -100 — sellers exhausted, support is firm'.format(ccim100)})

    # FII/DII flows
    if fii_net is not None:
        if fii_net > 2000:
            signals.append({'type': 'bullish', 'text':
                'Strong FII buying: ₹{:,.0f} Cr net purchase — foreign institutional tailwind'.format(fii_net)})
        elif fii_net > 0:
            signals.append({'type': 'neutral', 'text':
                'FII net buying: ₹{:,.0f} Cr — modest institutional support'.format(fii_net)})
        elif fii_net < -3000:
            signals.append({'type': 'warning', 'text':
                'Heavy FII selling: ₹{:,.0f} Cr net outflow — foreign institutional pressure; watch for DII to absorb'.format(abs(fii_net))})
        else:
            signals.append({'type': 'caution', 'text':
                'FII net selling: ₹{:,.0f} Cr — mild outflow; monitor for continuation'.format(abs(fii_net))})
    if dii_net is not None:
        if dii_net > 2000:
            signals.append({'type': 'bullish', 'text':
                'Strong DII support: ₹{:,.0f} Cr net buying — domestic mutual funds / insurance providing floor'.format(dii_net)})

    # Trend change warning
    if ma40_chg5 is not None and ma40_chg5 < -15:
        signals.append({'type': 'warning', 'text':
            'RAPID BREADTH DETERIORATION: MA40 breadth fell {:.1f}% in 5 days — significant internal breakdown underway. Reduce exposure, tighten stops.'.format(ma40_chg5)})

    # ── Actionable recommendation (Minervini / legendary trader framing) ────
    if deploy_signal == 'deploy_full':
        action = (
            'ACTIVELY DEPLOY CAPITAL. N500_MA40 = {:.1f}% — broad bull confirmed. '
            'Minervini would be running at 80-100% invested. Look for VCP breakouts on volume ≥1.5× average. '
            'Livermore would say: "When all factors align, act decisively." '
            'O\'Neil rule: buy exact pivot on 40-50%+ volume surge. Do not chase; add on pullbacks to the 10-day EMA. '
            'Suggested allocation: {:.0f}% of trading capital deployed across 3-6 positions.'.format(ma40, deploy_pct)
        )
    elif deploy_signal == 'deploy':
        action = (
            'DEPLOY 50-60% OF INTENDED ALLOCATION. Breadth = {:.1f}% — bull confirmed but not in super-bull territory. '
            'Minervini would carry 4-5 positions with partial sizing. Enter strongest setups now; hold cash for pullback adds. '
            'Watch for N500_MA40 sustaining >65% to add more. If weekly breadth (WMA40 = {}) also crosses 50%, go full. '
            'Suggested allocation: {:.0f}% deployed.'.format(
                ma40,
                '{:.1f}%'.format(wma40) if wma40 else '—',
                deploy_pct
            )
        )
    elif deploy_signal == 'cautious':
        action = (
            'CAUTIOUS POSITIONING. Breadth = {:.1f}% — at the bull/bear threshold (40%). '
            'Minervini would carry 1-2 positions max with tight stops. O\'Neil would wait for a Follow-Through Day. '
            'Druckenmiller rule: when uncertain, go small and wait for conviction. '
            'Focus on the absolute best setups (RS rank ≥85, VCP, near pivot). No averaging down. '
            'Suggested allocation: {:.0f}% maximum — treat every stop-out seriously.'.format(ma40, deploy_pct)
        )
    else:
        action = (
            'AVOID NEW LONG POSITIONS. Breadth = {:.1f}% — bear territory. '
            'Minervini moves to 100% cash in bear markets. Livermore: "The big money is made by sitting, not trading." '
            'Wait for N500_MA40 to cross above 40% on strong volume — that is your Follow-Through signal. '
            'Use this time to build your watchlist. When the market turns, you will be ready.'.format(
                ma40 if ma40 is not None else 0
            )
        )

    # ── Historical context (key statistics) ──────────────────────────────────
    # Compute % of time market was in each regime (last 3 years of data = ~750 bars)
    ma40_arr = d.get('pct_above_ma40_n500', [])
    last_750 = [v for v in ma40_arr[-750:] if v is not None]
    bull_pct  = round(sum(1 for v in last_750 if v >= 40) / max(len(last_750), 1) * 100, 1)
    bear_pct  = round(sum(1 for v in last_750 if v < 40)  / max(len(last_750), 1) * 100, 1)

    # Peak and trough in last 6 months (~125 bars)
    last_125 = [v for v in ma40_arr[-125:] if v is not None]
    peak_125  = max(last_125) if last_125 else None
    trough_125 = min(last_125) if last_125 else None

    # CCI100 correlation note
    cci100_arr = d.get('pct_above_cci100', [])
    last_cci100_22 = [v for v in cci100_arr[-22:] if v is not None]
    cci100_peak_22 = max(last_cci100_22) if last_cci100_22 else None

    historical = {
        'bull_pct_3y':     bull_pct,
        'bear_pct_3y':     bear_pct,
        'ma40_peak_6m':    round(peak_125, 1)  if peak_125  else None,
        'ma40_trough_6m':  round(trough_125, 1) if trough_125 else None,
        'cci100_peak_1m':  round(cci100_peak_22, 1) if cci100_peak_22 else None,
    }

    # ── CCI100 vs Nifty correlation note ─────────────────────────────────────
    cci100_note = None
    if cci100 is not None and cci100_peak_22 is not None:
        drop = cci100_peak_22 - cci100
        if drop > 15 and cci100 < 15:
            cci100_note = (
                'CCI100 breadth has dropped {:.1f}% from its recent 1-month peak ({:.1f}% → {:.1f}%). '
                'Historically, when CCI100 falls >15pp from peak while MA40 is still >50%, '
                'the market consolidates 2-6 weeks then resumes if broader breadth holds. '
                'This is NOT a sell signal unless MA40 breaks below 40%.'.format(
                    drop, cci100_peak_22, cci100)
            )
        elif drop > 5:
            cci100_note = (
                'CCI100 breadth declining from {:.1f}% peak → {:.1f}% current ({:.1f}% drop). '
                'Overbought stocks normalizing — healthy for sustainable advance. '
                'Watch: if CCI100 rebounds above 15% again, momentum is resuming.'.format(
                    cci100_peak_22, cci100, drop)
            )

    return {
        'generated_at':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_data_date': last_date,
        'nifty_close':    round(float(nifty_v), 2) if nifty_v else None,
        'regime':         regime,
        'regime_code':    regime_code,
        'trend':          trend,
        'deploy_signal':  deploy_signal,
        'deploy_pct':     deploy_pct,
        'action':         action,
        'cci100_note':    cci100_note,
        'key_readings': {
            'n500_ma40':    round(ma40,    1) if ma40    is not None else None,
            'n500_ma200':   round(ma200,   1) if ma200   is not None else None,
            'n500_wma40':   round(wma40,   1) if wma40   is not None else None,
            'all_ma40':     round(ma40_all,1) if ma40_all is not None else None,
            'cci0':         round(cci0,    1) if cci0    is not None else None,
            'cci100':       round(cci100,  1) if cci100  is not None else None,
            'cci_m100':     round(ccim100, 1) if ccim100 is not None else None,
            'fii_net':      round(fii_net, 2) if fii_net is not None else None,
            'dii_net':      round(dii_net, 2) if dii_net is not None else None,
            'ma40_chg5d':   ma40_chg5,
            'cci100_chg5d': cci100_chg5,
            'ma200_chg5d':  ma200_chg5,
        },
        'signals':     signals,
        'historical':  historical,
    }


# ── MAIN SCANNER ──────────────────────────────────────────────────────────────
def run():
    # Parse args
    fast     = '--fast' in sys.argv
    max_syms = None
    if '--symbols' in sys.argv:
        try:
            max_syms = int(sys.argv[sys.argv.index('--symbols') + 1])
        except:
            pass
    if fast and not max_syms:
        max_syms = 100

    symbols = load_symbols(max_syms)
    total   = len(symbols)
    print(f"\nMinervini Trend Template Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Scanning {total} NSE stocks...\n")

    # ── FETCH DATA ────────────────────────────────────────────────────────────
    data     = {}   # 210+ days (full evaluation)
    ipo_data = {}   # 30-209 days (IPO/recent listing)
    errors   = 0
    for i, sym in enumerate(symbols):
        df = fetch_data(sym)
        if df is not None:
            if len(df) >= 210:
                data[sym] = df
            elif len(df) >= 30:
                ipo_data[sym] = df
            else:
                errors += 1
        else:
            errors += 1
        done = i + 1
        bar = '#' * (done * 30 // total)
        print(f"\r  [{bar:<30}] {done}/{total}  valid:{len(data)}  ipo:{len(ipo_data)}  skip:{errors}", end='', flush=True)

    print(f"\n\n  {len(data)} stocks with 210+ days  |  {len(ipo_data)} recent IPOs (30-209 days)  |  {errors} skipped\n")

    # ── RS RANKS ─────────────────────────────────────────────────────────────
    print("Calculating RS ranks...")
    rs_ranks = calc_rs_ranks(data)

    # ── EVALUATE ALL STOCKS ───────────────────────────────────────────────────
    print("Running screens...\n")
    evaluations = {}
    for sym, df in data.items():
        rs = rs_ranks.get(sym, 0)
        ev = evaluate(df, rs)
        if ev:
            evaluations[sym] = (ev, df)

    # ── DEFINE SCREENS ────────────────────────────────────────────────────────
    # 8 individual criterion screens + 6 composite screens
    screens = {
        # Individual criteria (8)
        'c1': {
            'label': 'C1 — Price above MA150 & MA200',
            'desc':  'Current price is above both the 150-day and 200-day moving averages. Basic Stage 2 price structure.',
            'group': 'individual', 'stocks': []
        },
        'c2': {
            'label': 'C2 — MA150 above MA200',
            'desc':  '150-day MA is trading above the 200-day MA. Confirms medium-term trend is stronger than long-term.',
            'group': 'individual', 'stocks': []
        },
        'c3': {
            'label': 'C3 — MA200 trending up (1+ month)',
            'desc':  '200-day moving average is higher than it was ~21 trading days ago. Long-term trend is rising.',
            'group': 'individual', 'stocks': []
        },
        'c4': {
            'label': 'C4 — MA50 above MA150 & MA200',
            'desc':  '50-day MA is above both longer-term MAs. The full MA stack is bullish-aligned.',
            'group': 'individual', 'stocks': []
        },
        'c5': {
            'label': 'C5 — Price above MA50',
            'desc':  'Current price is above the 50-day moving average. Short-term trend is bullish.',
            'group': 'individual', 'stocks': []
        },
        'c6': {
            'label': 'C6 — 25%+ above 52-week low',
            'desc':  'Price has risen at least 25% from its 52-week low. Stage 1 bottom is well behind it.',
            'group': 'individual', 'stocks': []
        },
        'c7': {
            'label': 'C7 — Within 25% of 52-week high',
            'desc':  'Price is within 25% of its 52-week high. Not extended into a deep base — near the top of range.',
            'group': 'individual', 'stocks': []
        },
        'c8': {
            'label': 'C8 — RS Rank >= 70',
            'desc':  'Relative Strength rank is 70 or above (top 30% of NSE universe). Institutional-grade leader.',
            'group': 'individual', 'stocks': []
        },
        # Composite screens (6)
        'full_template': {
            'label': 'Full Trend Template (8/8)',
            'desc':  'All 8 Minervini criteria met. Confirmed Stage 2 uptrend — eligible for SEPA entry analysis.',
            'group': 'composite', 'stocks': []
        },
        'near_breakout': {
            'label': 'Near Breakout (within 5% of high)',
            'desc':  'Passes full template AND is within 5% of 52-week high. Approaching a potential pivot point.',
            'group': 'composite', 'stocks': []
        },
        'rs_leaders': {
            'label': 'RS Leaders (RS >= 85)',
            'desc':  'Passes full template AND has RS rank of 85+. Top 15% performers in the NSE universe.',
            'group': 'composite', 'stocks': []
        },
        'vcp_setup': {
            'label': 'VCP Setup',
            'desc':  'Passes full template AND shows a Volatility Contraction Pattern — contracting ranges with drying volume.',
            'group': 'composite', 'stocks': []
        },
        'minervini_backtest': {
            'label': 'Minervini Backtest Picks (30% CAGR)',
            'desc':  'Exact 30% CAGR strategy entry signal: all 8 criteria + VCP 3-segment base + volume surge ≥1.25× 50-day avg + within 20% of 52-week high. Stop at base low ×0.995 or 8% max. Breakeven at +10%, trail MA50 at +15%.',
            'group': 'composite', 'stocks': []
        },
        'new_highs': {
            'label': 'New 52-Week Highs (within 2%)',
            'desc':  'Passes full template AND is within 2% of its 52-week high. At or near a breakout level.',
            'group': 'composite', 'stocks': []
        },
        'watch_list': {
            'label': 'Watch List — 7/8 Criteria',
            'desc':  'Missing exactly one criterion. One step away from a full setup — monitor daily.',
            'group': 'composite', 'stocks': []
        },
        'c1_to_c6': {
            'label': 'C1–C6 — Stage 2 Uptrend Structure',
            'desc':  'Stocks meeting all 6 trend structure criteria (C1–C6): price above MA150/MA200 (C1), MA150 above MA200 (C2), MA200 trending up (C3), MA50 above MA150/MA200 (C4), price above MA50 (C5), and 25%+ above 52W low (C6). These stocks are in a confirmed Stage 2 uptrend — may not yet be near highs (C7) or have top RS (C8) but the base structure is sound.',
            'group': 'composite', 'stocks': []
        },
        'htf_setup': {
            'label': 'High Tight Flag',
            'desc':  "Stock gained ≥90% in ≤8 weeks (the pole) then consolidated ≤25% for 1-5 weeks (the flag). One of O'Neil's rarest and most powerful patterns.",
            'group': 'htf', 'stocks': []
        },
        'htf_potential': {
            'label': 'Potential High Tight Flag',
            'desc':  'Stock gained ≥50% in ≤8 weeks and is holding within 15% of its recent peak. Flag stage may be forming — watch for tightening price action.',
            'group': 'htf', 'stocks': []
        },
        'ma_pullback': {
            'label': 'MA Pullback — Trend Rider',
            'desc':  'Stock above MA50, MA150 and MA200 (Minervini uptrend), MAs stacked (EMA10 > EMA21 > MA50), and price pulling back to within 3% of EMA10, 4% of EMA21, or 5% of MA50. Classic trend-continuation entry — price can be at or below EMA10 during the pullback.',
            'group': 'htf', 'stocks': []
        },
        'hhhl_pullback': {
            'label': 'HH/HL Pullback',
            'desc':  'Stock in a confirmed uptrend (higher highs + higher lows) with price pulling back to near the most recent Higher Low. Clean trend-continuation entry — buy the dip into support.',
            'group': 'htf', 'stocks': []
        },
        'primary_base_new': {
            'label': 'Primary Base — Recent IPO (≤3 yrs)',
            'desc':  'Minervini Primary Base: the first buyable consolidation after a recent IPO (listed within ~3 years). Stock is within 20% of its all-time high, forming a tight base (range < 35%) for at least 3 weeks, still above MA50/MA150/MA200. These are the early-stage superperformers before they break out.',
            'group': 'htf', 'stocks': []
        },
        'primary_base_10yr': {
            'label': 'Primary Base — Young Company (≤10 yrs)',
            'desc':  'Minervini: "Most superperformers go public within 8–10 years before their superperformance phase." Stocks listed within ~10 years that are forming a tight base near their all-time high — still in uptrend (above MA50/MA150/MA200), base range < 35% over the last 6 months.',
            'group': 'htf', 'stocks': []
        },
        'ipo_watch': {
            'label': 'Recent IPOs & Listings (<210 days)',
            'desc':  'Stocks with 30-209 days of data since listing. MAs and full template can\'t be computed yet, but price action, RS rank, and volume are tracked. Watch for when they hit 210 days and enter the full scanner.',
            'group': 'ipo', 'stocks': []
        },
        # ── CCI(34) SCREENS ────────────────────────────────────────────────────
        'cci34_daily_cross_100': {
            'label': 'CCI34 — Daily just crossed above 100',
            'desc':  'Daily CCI(34) crossed above 100 today (was below, now above). Fresh breakout momentum signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_weekly_cross_100': {
            'label': 'CCI34 — Weekly just crossed above 100',
            'desc':  'Weekly CCI(34) crossed above 100 this week (was below, now above). Long-term fresh breakout signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_daily_100': {
            'label': 'CCI34 — Daily ≥ 100',
            'desc':  'All stocks where Daily CCI(34) is currently ≥ 100. Bullish momentum signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_daily_neg100': {
            'label': 'CCI34 — Daily crossed above -100',
            'desc':  'Daily CCI(34) crossed above -100 today. Trendline reversal / oversold recovery signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_weekly_100': {
            'label': 'CCI34 — Weekly ≥ 100',
            'desc':  'All stocks where Weekly CCI(34) is currently ≥ 100. Long-term bullish momentum signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_weekly_neg100': {
            'label': 'CCI34 — Weekly crossed above -100',
            'desc':  'Weekly CCI(34) crossed above -100 this week. Long-term trendline reversal signal.',
            'group': 'cci34', 'stocks': []
        },
        'cci34_best_setups': {
            'label': 'CCI34 — Minervini Best Setups',
            'desc':  'Full Trend Template (8/8) or Near Breakout stocks where CCI(34) is ≥ 100 on daily or weekly, or just crossed above 100. The strongest Minervini setups with CCI34 momentum confirmation.',
            'group': 'cci34', 'stocks': []
        },
        'strong_earnings': {
            'label': 'CANSLIM Earnings + CCI Momentum',
            'desc':  'Stocks in BOTH Daily ≥100 AND Weekly ≥100 CCI momentum with CANSLIM-grade quarterly results — EPS YoY ≥25% (CANSLIM C minimum; ≥40% = C★) and Revenue YoY ≥20%. Fundamental acceleration and price momentum aligned simultaneously. Sorted by combined growth score (EPS YoY + Rev YoY + NP YoY).',
            'group': 'cci34', 'stocks': []
        },
        'fo_momentum': {
            'label': 'F&O Stocks in Dashboard',
            'desc':  'All F&O-eligible stocks (Futures & Options segment) that appear in any key dashboard screen — Full Template, Near Breakout, Watch List, VCP, RS Leaders, CCI Daily/Weekly ≥100, HTF, MA Pullback, HH/HL Pullback. Shows which screens each stock is in. Ideal for option/futures strategies on technically strong setups.',
            'group': 'composite', 'stocks': []
        },
        'fo_strong_uptrend': {
            'label': 'F&O — Strong Uptrend (C1–C6)',
            'desc':  'F&O-eligible stocks in a confirmed Stage 2 uptrend — all 6 structural Minervini criteria (C1–C6) are met. Price above all key MAs, MAs properly stacked, price 25%+ above 52-week low. These are the strongest actionable setups for futures/options strategies.',
            'group': 'composite', 'stocks': []
        },
        'newly_added': {
            'label': 'New This Scan',
            'desc':  'Stocks newly appearing in MA Pullback, CCI Daily ≥100, CCI Weekly ≥100, C1–C6, or Full Template since the previous scan run. Refresh this after every scan to quickly spot fresh setups without reviewing all screens.',
            'group': 'composite', 'stocks': []
        },
        'young_3yr': {
            'label': 'Dashboard Stocks — Listed ≤ 3 Years',
            'desc':  'All stocks currently appearing in any key dashboard screen that listed within the last 3 years. Youth + technical setup = maximum opportunity. Shows which screens each stock is in.',
            'group': 'composite', 'stocks': []
        },
        'young_10yr': {
            'label': 'Dashboard Stocks — Listed ≤ 10 Years',
            'desc':  'All stocks currently appearing in any key dashboard screen that listed within the last 10 years. Minervini: most superperformers go public within 8–10 years before their superperformance phase. Shows which screens each stock is in.',
            'group': 'composite', 'stocks': []
        },
        'ck_daily_100': {
            'label': 'Chartink — Daily CCI34 ≥ 100',
            'desc':  'All NSE cash stocks where CCI(34) on the daily timeframe is currently ≥ 100, sourced live from Chartink. Covers the full NSE universe (not just MTF). Stocks in our MTF scan universe show full price/RS/VCP data; others show basic info.',
            'group': 'chartink', 'stocks': []
        },
        'ck_weekly_100': {
            'label': 'Chartink — Weekly CCI34 ≥ 100',
            'desc':  'All NSE cash stocks where CCI(34) on the weekly timeframe is currently ≥ 100, sourced live from Chartink. Weekly CCI ≥ 100 signals sustained multi-week momentum — stronger and more selective than the daily signal.',
            'group': 'chartink', 'stocks': []
        },
        'ck_daily_cross': {
            'label': 'Chartink — Daily CCI34 Just Crossed 100',
            'desc':  'NSE stocks where Daily CCI(34) crossed above 100 today — fresh momentum breakout signal. Chartink live data. These are new entries into the ≥100 zone.',
            'group': 'chartink', 'stocks': []
        },
        'ck_weekly_cross': {
            'label': 'Chartink — Weekly CCI34 Just Crossed 100',
            'desc':  'NSE stocks where Weekly CCI(34) crossed above 100 this week. Chartink live data. Weekly cross is a higher-conviction signal — sustained weekly momentum just turned strongly bullish.',
            'group': 'chartink', 'stocks': []
        },
        'ck_mtf_100': {
            'label': 'Chartink CCI34 ≥ 100 × MTF Universe',
            'desc':  'Stocks confirmed by Chartink live data as CCI34 ≥ 100 (daily or weekly) AND present in the live Zerodha MTF scan universe. The cleanest, most actionable intersection — Chartink accuracy + full MTF enriched data. Shows which CCI signals are active per stock.',
            'group': 'chartink', 'stocks': []
        },
    }

    # ── LOAD F&O SET FROM ZERODHA MTF CSV ────────────────────────────────────
    _fo_set = set()
    for _path in SYMBOL_PATHS:
        if os.path.exists(_path):
            try:
                _mtf_df = pd.read_csv(_path)
                if 'category' in _mtf_df.columns and 'tradingsymbol' in _mtf_df.columns:
                    _fo_set = set(_mtf_df[_mtf_df['category'] == 'fo']['tradingsymbol'].str.strip().str.upper())
                    print(f"  F&O set loaded: {len(_fo_set)} stocks from {_path}", flush=True)
                    break
            except Exception:
                pass

    # ── LOAD HISTORY & SET TODAY ───────────────────────────────────────────────
    # History is used ONLY for C8 (RS rank ≥ 70) — we can't compute historical
    # RS ranks without full-universe data for each past date.
    history   = load_history()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # ── SNAPSHOT PREVIOUS SCAN (before results.json gets overwritten) ─────────
    # Read the OLD results.json NOW while it still contains the previous scan's
    # data. Save it as prev_screens.json — the baseline for "New This Scan" diff.
    # Only do this when results.json is from a PREVIOUS date; same-day re-runs
    # keep the existing prev_screens.json so the baseline stays anchored to
    # yesterday's scan (not this morning's re-run).
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as _pf:
                _old_results = json.load(_pf)
            _prev_date = (_old_results.get('generated_at') or '')[:10]
            if _prev_date and _prev_date != today_str:
                # Different date → snapshot it as new baseline
                _prev_snap_data = {'scan_date': _prev_date}
                for _trk_key, _ in _TRACKED_SCREENS:
                    _prev_snap_data[_trk_key] = [
                        e.get('ticker', '')
                        for e in _old_results.get('screens', {}).get(_trk_key, {}).get('stocks', [])
                        if e.get('ticker')
                    ]
                _tmp_ps = PREV_SCREENS_JSON + '.tmp'
                with open(_tmp_ps, 'w') as _pf2:
                    json.dump(_prev_snap_data, _pf2)
                os.replace(_tmp_ps, PREV_SCREENS_JSON)
                print(f"  Prev scan snapshot: {_prev_date} "
                      f"({len(_prev_snap_data.get('full_template',[]))} full template, "
                      f"{len(_prev_snap_data.get('ck_daily_100',[]))} CCI daily)")
            else:
                # Same-day re-run: keep existing prev_screens.json intact
                _prev_date2 = ''
                if os.path.exists(PREV_SCREENS_JSON):
                    try:
                        with open(PREV_SCREENS_JSON) as _pf2: _prev_date2 = json.load(_pf2).get('scan_date','?')
                    except Exception: pass
                print(f"  Same-day re-run: keeping prev snapshot from {_prev_date2 or '?'}")
        except Exception as _snap_ex:
            print(f"  Prev scan snapshot failed: {_snap_ex}")

    # ── LISTING DATES FOR PRIMARY BASE SCREEN ────────────────────────────────
    _listing_dates = get_listing_dates(list(evaluations.keys()))

    # ── CHARTINK LIVE CCI DATA ────────────────────────────────────────────────
    # Use Chartink's real-time NSE data for CCI inclusion/exclusion decisions.
    # This overrides stale cache — a stock is added to a CCI screen if EITHER
    # our computed value OR Chartink says it qualifies.
    _ck = fetch_chartink_cci_sets()

    # ── RUN SCREENS ───────────────────────────────────────────────────────────
    for sym, (ev, df) in evaluations.items():
        cr     = ev['criteria']
        ticker = sym.replace('.NS', '')

        # Compute price-action streak dates for C1-C7
        cdates = find_criteria_dates(df)

        # Compute VCP last setup dates (daily + weekly) — stored in ev for build_entry
        ev['vcp_last_daily']  = find_last_vcp_date(df, weekly=False)
        ev['vcp_last_weekly'] = find_last_vcp_date(df, weekly=True)

        # C8 date: use scanner history (can't compute historical RS rank)
        c8_date = record_and_get_date(history, 'c8', ticker, today_str) if cr['c8'] else None

        # Full template "since" = the date the LAST of C1-C7 criteria was satisfied
        # Exclude C8 (RS rank) — it's tracked via history and would always show today
        # on the first run, contaminating the date for all composite screens.
        all_crit_dates = [cdates.get(k) for k in ['c1','c2','c3','c4','c5','c6','c7'] if cdates.get(k)]
        full_since = max(all_crit_dates) if all_crit_dates else today_str

        # Individual criterion screens — use the price-action streak date for each
        for ckey in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']:
            if cr[ckey]:
                screens[ckey]['stocks'].append(
                    build_entry(sym, ev, {'added_date': cdates.get(ckey, today_str)})
                )
        if cr['c8']:
            screens['c8']['stocks'].append(
                build_entry(sym, ev, {'added_date': c8_date or today_str})
            )

        # Composite screens — use full_since (date all criteria were simultaneously met)
        if ev['all_pass']:
            screens['full_template']['stocks'].append(
                build_entry(sym, ev, {'added_date': full_since})
            )

            if ev['pct_from_high'] >= -NEAR_BREAKOUT_PCT * 100:
                screens['near_breakout']['stocks'].append(
                    build_entry(sym, ev, {'added_date': full_since})
                )

            if ev['rs_rank'] >= RS_LEADER_MIN:
                screens['rs_leaders']['stocks'].append(
                    build_entry(sym, ev, {'added_date': full_since})
                )

            if ev['pct_from_high'] >= -NEW_HIGH_PCT * 100:
                screens['new_highs']['stocks'].append(
                    build_entry(sym, ev, {'added_date': full_since})
                )

            vcp_flag = detect_vcp(df)
            if vcp_flag:
                screens['vcp_setup']['stocks'].append(
                    build_entry(sym, ev, {'vcp': True, 'added_date': full_since})
                )
                # Backtest-exact entry: VCP + volume surge ≥1.25× + within 20% of high (30% CAGR params)
                if ev['vol_ratio'] >= 1.25 and ev['pct_from_high'] >= -20.0:
                    screens['minervini_backtest']['stocks'].append(
                        build_entry(sym, ev, {'vcp': True, 'vol_surge': True, 'added_date': full_since})
                    )

        elif ev['passed'] == 7:
            failed = [k for k, v in cr.items() if not v]
            watch_dates = [cdates.get(k) for k in ['c1','c2','c3','c4','c5','c6','c7'] if cdates.get(k)]
            watch_since = max(watch_dates) if watch_dates else today_str
            screens['watch_list']['stocks'].append(
                build_entry(sym, ev, {'failed_criteria': failed, 'added_date': watch_since})
            )

        # C1-C6 screen: confirmed Stage 2 uptrend structure regardless of C7/C8
        if cr['c1'] and cr['c2'] and cr['c3'] and cr['c4'] and cr['c5'] and cr['c6']:
            screens['c1_to_c6']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str})
            )

        # ── HTF — runs on ALL stocks regardless of template status ────────────
        _htf_full, _htf_pot, _htf_det = detect_htf(df)
        if _htf_full:
            screens['htf_setup']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str, **(_htf_det or {})})
            )
        elif _htf_pot:
            screens['htf_potential']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str, **(_htf_det or {})})
            )

        # ── MA Pullback / Trend Rider ─────────────────────────────────────────
        try:
            _c    = df['Close'].squeeze()
            _p    = float(_c.iloc[-1])
            _e10  = float(_c.ewm(span=10,  adjust=False).mean().iloc[-1])
            _e21  = float(_c.ewm(span=21,  adjust=False).mean().iloc[-1])
            _ma50 = float(_c.rolling(50).mean().iloc[-1])
            _ma150= float(_c.rolling(150).mean().iloc[-1]) if len(_c) >= 150 else None
            _ma200= float(_c.rolling(200).mean().iloc[-1]) if len(_c) >= 200 else None
            # Trend requirement: price above MA50, MA150, MA200 (Minervini C1+C5)
            _above_mas = (_p > _ma50 > 0 and
                          (_ma150 is None or _p > _ma150) and
                          (_ma200 is None or _p > _ma200))
            # MA structure: EMA10 > EMA21 > MA50 (trend intact, price may be pulling back)
            _ma_stacked = _e10 > _e21 > _ma50 > 0
            # Pullback proximity: price is near EMA10, EMA21, or MA50
            _near10  = abs(_p - _e10)  / _e10  <= 0.03
            _near21  = abs(_p - _e21)  / _e21  <= 0.04
            _near50  = abs(_p - _ma50) / _ma50 <= 0.05
            _near_any = _near10 or _near21 or _near50
            # Near high: within 25% of 52W high
            _hi52_ma = float(df['High'].iloc[-252:].max()) if len(df) >= 252 else float(df['High'].max())
            _near_high = _p >= _hi52_ma * 0.75
            if _above_mas and _ma_stacked and _near_any and _near_high:
                _which = 'EMA10' if _near10 else ('EMA21' if _near21 else 'MA50')
                screens['ma_pullback']['stocks'].append(
                    build_entry(sym, ev, {'added_date': today_str, 'ma_pullback_near': _which})
                )
        except Exception:
            pass

        # ── HH/HL Pullback ───────────────────────────────────────────────────
        _hhhl_ok, _hhhl_det = detect_hhhl(df)
        if _hhhl_ok:
            screens['hhhl_pullback']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str, **(_hhhl_det or {})})
            )

        # ── Primary Base ─────────────────────────────────────────────────────
        _ld = _listing_dates.get(sym)
        _pb_new_ok,  _pb_new_det  = detect_primary_base(df, _ld, max_years=3)
        _pb_10yr_ok, _pb_10yr_det = detect_primary_base(df, _ld, max_years=10)
        if _pb_new_ok:
            screens['primary_base_new']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str, **(_pb_new_det or {})})
            )
        elif _pb_10yr_ok:
            screens['primary_base_10yr']['stocks'].append(
                build_entry(sym, ev, {'added_date': today_str, **(_pb_10yr_det or {})})
            )

    # ── CCI(34) SCREENS: run on ALL full-data stocks regardless of TT ──────────
    for sym, df in data.items():
        ticker = sym.replace('.NS', '')
        rs     = rs_ranks.get(sym, 0)
        try:
            price = float(df['Close'].squeeze().iloc[-1])
        except:
            price = 0.0

        cci_d_now, cci_d_prev = calc_cci(df, period=34, weekly=False)
        cci_w_now, cci_w_prev = calc_cci(df, period=34, weekly=True)

        # VCP last setup dates for CCI entries
        _cci_vcp_last_d = find_last_vcp_date(df, weekly=False)
        _cci_vcp_last_w = find_last_vcp_date(df, weekly=True)

        # MA proximity + pivot for this symbol
        try:
            c_s = df['Close'].squeeze()
            _ma50  = float(c_s.rolling(50).mean().iloc[-1])
            _ema21 = float(c_s.ewm(span=21, adjust=False).mean().iloc[-1])
            _ema10 = float(c_s.ewm(span=10, adjust=False).mean().iloc[-1])
            def _pma(ma): return round((price - ma) / ma * 100, 2) if ma and ma > 0 else None
            _pct_ma50  = _pma(_ma50)
            _pct_ema21 = _pma(_ema21)
            _pct_ema10 = _pma(_ema10)
        except:
            _pct_ma50 = _pct_ema21 = _pct_ema10 = None

        _pv_high,   _pv_bars,   _pv_crossed   = calc_pivot_high(df, weekly=False)
        _pv_high_w, _pv_bars_w, _pv_crossed_w = calc_pivot_high(df, weekly=True)
        _pct_pivot   = round((price - _pv_high)   / _pv_high   * 100, 2) if _pv_high   else None
        _pct_pivot_w = round((price - _pv_high_w) / _pv_high_w * 100, 2) if _pv_high_w else None

        # Passed/criteria for the CCI stock (may not be in evaluations if sub-210-bar)
        _cci_ev   = evaluations.get(sym)
        _cci_pass = _cci_ev[0]['passed']   if _cci_ev else 0
        _cci_cr   = _cci_ev[0]['criteria'] if _cci_ev else {}

        def _cci_entry(cci_now, cci_prev):
            return {
                'ticker':          ticker,
                'tv_symbol':       'NSE:' + ticker,
                'price':           round(price, 2),
                'rs_rank':         rs,
                'passed':          _cci_pass,
                'criteria':        _cci_cr,
                'cci_now':         round(cci_now, 1)  if cci_now  is not None else None,
                'cci_prev':        round(cci_prev, 1) if cci_prev is not None else None,
                'pct_from_ma50':   _pct_ma50,
                'pct_from_ema21':  _pct_ema21,
                'pct_from_ema10':  _pct_ema10,
                'pivot_high':       _pv_high,
                'pivot_bars_ago':   _pv_bars,
                'pivot_crossed':    _pv_crossed,
                'pct_from_pivot':   _pct_pivot,
                'pivot_high_w':     _pv_high_w,
                'pivot_bars_ago_w': _pv_bars_w,
                'pivot_crossed_w':  _pv_crossed_w,
                'pct_from_pivot_w': _pct_pivot_w,
                'vcp_last_daily':   _cci_vcp_last_d,
                'vcp_last_weekly':  _cci_vcp_last_w,
                'added_date':       today_str,
            }

        # ── CCI34 screens — inclusion from computed value OR Chartink live data ──
        # Chartink overrides stale cache: if Chartink says >=100, include even if
        # our computed value is slightly below due to a 1-day-old cache.
        _ck_in_d100  = ticker in _ck.get('cci34_daily_100',  set())
        _ck_in_w100  = ticker in _ck.get('cci34_weekly_100', set())
        _ck_in_dcross = ticker in _ck.get('cci34_daily_cross_100',  set())
        _ck_in_wcross = ticker in _ck.get('cci34_weekly_cross_100', set())
        _ck_in_dneg  = ticker in _ck.get('cci34_daily_neg100',  set())
        _ck_in_wneg  = ticker in _ck.get('cci34_weekly_neg100', set())

        # Use computed CCI values for the display number; fall back to 100.5 sentinel
        # when Chartink confirms >=100 but our stale cache computed slightly below.
        _eff_d_now  = cci_d_now  if cci_d_now  is not None else (100.5 if _ck_in_d100  else None)
        _eff_d_prev = cci_d_prev if cci_d_prev is not None else 99.0
        _eff_w_now  = cci_w_now  if cci_w_now  is not None else (100.5 if _ck_in_w100  else None)
        _eff_w_prev = cci_w_prev if cci_w_prev is not None else 99.0

        # Daily CCI34 screens
        _d_valid = _eff_d_now is not None
        if _d_valid:
            if (_eff_d_now >= 100 and _eff_d_prev < 100) or _ck_in_dcross:
                screens['cci34_daily_cross_100']['stocks'].append(_cci_entry(_eff_d_now, _eff_d_prev))
            if _eff_d_now >= 100 or _ck_in_d100:
                _de = _cci_entry(max(_eff_d_now, 100.0) if _ck_in_d100 else _eff_d_now, _eff_d_prev)
                _d_streak = calc_cci_streak_start(df, period=34, weekly=False) or today_str
                _de['added_date'] = _d_streak
                _de['is_new']     = (_d_streak == today_str)
                screens['cci34_daily_100']['stocks'].append(_de)
            if (_eff_d_now >= -100 and _eff_d_prev < -100) or _ck_in_dneg:
                screens['cci34_daily_neg100']['stocks'].append(_cci_entry(_eff_d_now, _eff_d_prev))

        # Weekly CCI34 screens
        _w_valid = _eff_w_now is not None
        if _w_valid:
            if (_eff_w_now >= 100 and _eff_w_prev < 100) or _ck_in_wcross:
                screens['cci34_weekly_cross_100']['stocks'].append(_cci_entry(_eff_w_now, _eff_w_prev))
            if _eff_w_now >= 100 or _ck_in_w100:
                _we = _cci_entry(max(_eff_w_now, 100.0) if _ck_in_w100 else _eff_w_now, _eff_w_prev)
                _w_streak = calc_cci_streak_start(df, period=34, weekly=True) or today_str
                _we['added_date'] = _w_streak
                _we['is_new']     = (_w_streak == today_str)
                screens['cci34_weekly_100']['stocks'].append(_we)
            if (_eff_w_now >= -100 and _eff_w_prev < -100) or _ck_in_wneg:
                screens['cci34_weekly_neg100']['stocks'].append(_cci_entry(_eff_w_now, _eff_w_prev))

    # ── CHARTINK FILL-IN: stocks Chartink confirms but missed due to bar-count gate ──
    # Stocks with < 210 bars (new listings) skip evaluate() entirely, so they never
    # reach the CCI screen code above. For any Chartink-confirmed stock in the MTF
    # universe that is still missing from the CCI screens, add a minimal entry now.
    _already_in_d100 = {e['ticker'] for e in screens['cci34_daily_100']['stocks']}
    _already_in_w100 = {e['ticker'] for e in screens['cci34_weekly_100']['stocks']}
    _scanned_tickers = {sym.replace('.NS','') for sym in evaluations}
    _all_syms_data   = {sym.replace('.NS',''): df for sym, (ev, df) in evaluations.items()}

    for _ck_ticker in _ck.get('cci34_daily_100', set()):
        if _ck_ticker in _already_in_d100 or _ck_ticker in _scanned_tickers:
            continue  # already present or was scanned (handled above)
        # Stock is in Chartink daily>=100 but not scanned — try to load its cache directly
        _ck_pkl = os.path.join(CACHE_DIR, _ck_ticker + '_NS.pkl')
        if not os.path.exists(_ck_pkl):
            continue
        try:
            with open(_ck_pkl, 'rb') as _f: _ck_df = pickle.load(_f)
            if isinstance(_ck_df.columns, pd.MultiIndex):
                _ck_df.columns = _ck_df.columns.get_level_values(0)
            if len(_ck_df) < 34 + 2:
                continue
            _ck_c = float(_ck_df['Close'].iloc[-1])
            _ck_rs = rs_ranks.get(_ck_ticker + '.NS', 0)
            _ck_hi52 = float(_ck_df['Close'].rolling(252, min_periods=1).max().iloc[-1])
            _ck_lo52 = float(_ck_df['Close'].rolling(252, min_periods=1).min().iloc[-1])
            _ck_pct_hi = (_ck_c / _ck_hi52 - 1) * 100 if _ck_hi52 else None
            _ck_vol = float(_ck_df['Volume'].iloc[-1])
            _ck_avgvol = float(_ck_df['Volume'].iloc[-50:].mean()) if len(_ck_df) >= 50 else _ck_vol
            _ck_volr = round(_ck_vol / _ck_avgvol, 2) if _ck_avgvol else None
            _ck_tp  = (_ck_df['High'] + _ck_df['Low'] + _ck_df['Close']) / 3.0
            _ck_sma = _ck_tp.rolling(34).mean()
            _ck_mad = _ck_tp.rolling(34).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            _ck_cci = ((_ck_tp - _ck_sma) / (0.015 * _ck_mad))
            _ck_now  = float(_ck_cci.iloc[-1])
            _ck_prev = float(_ck_cci.iloc[-2])
            _d_streak = calc_cci_streak_start(_ck_df, period=34, weekly=False) or today_str
            _fill_entry = {
                'ticker':        _ck_ticker,
                'tv_symbol':     f'NSE:{_ck_ticker}',
                'price':         round(_ck_c, 2),
                'rs_rank':       _ck_rs,
                'pct_from_high': round(_ck_pct_hi, 1) if _ck_pct_hi is not None else None,
                'vol_ratio':     _ck_volr,
                'cci_now':       round(max(_ck_now, 100.0), 1),
                'cci_prev':      round(_ck_prev, 1),
                'pct_from_ma50': None, 'pct_from_ema21': None, 'pct_from_ema10': None,
                'pivot_high': None, 'pivot_bars_ago': None, 'pivot_crossed': False,
                'pct_from_pivot': None, 'pivot_high_w': None, 'pivot_bars_ago_w': None,
                'pivot_crossed_w': False, 'pct_from_pivot_w': None,
                'vcp_last_daily': None, 'vcp_last_weekly': None,
                'added_date':    _d_streak, 'is_new': (_d_streak == today_str),
            }
            screens['cci34_daily_100']['stocks'].append(_fill_entry)
            if _ck_now >= 100 and _ck_prev < 100:
                screens['cci34_daily_cross_100']['stocks'].append(_fill_entry)
        except Exception:
            pass

    for _ck_ticker in _ck.get('cci34_weekly_100', set()):
        if _ck_ticker in _already_in_w100 or _ck_ticker in _scanned_tickers:
            continue
        _ck_pkl = os.path.join(CACHE_DIR, _ck_ticker + '_NS.pkl')
        if not os.path.exists(_ck_pkl):
            continue
        try:
            with open(_ck_pkl, 'rb') as _f: _ck_df = pickle.load(_f)
            if isinstance(_ck_df.columns, pd.MultiIndex):
                _ck_df.columns = _ck_df.columns.get_level_values(0)
            _ck_ohlc = _ck_df[['High','Low','Close']].resample('W').agg({'High':'max','Low':'min','Close':'last'}).dropna()
            if len(_ck_ohlc) < 34 + 2:
                continue
            _ck_c    = float(_ck_df['Close'].iloc[-1])
            _ck_rs   = rs_ranks.get(_ck_ticker + '.NS', 0)
            _ck_tp   = (_ck_ohlc['High'] + _ck_ohlc['Low'] + _ck_ohlc['Close']) / 3.0
            _ck_sma  = _ck_tp.rolling(34).mean()
            _ck_mad  = _ck_tp.rolling(34).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            _ck_cci  = ((_ck_tp - _ck_sma) / (0.015 * _ck_mad))
            _ck_wnow  = float(_ck_cci.iloc[-1])
            _ck_wprev = float(_ck_cci.iloc[-2])
            _w_streak = calc_cci_streak_start(_ck_df, period=34, weekly=True) or today_str
            _wfill = {
                'ticker':        _ck_ticker,
                'tv_symbol':     f'NSE:{_ck_ticker}',
                'price':         round(_ck_c, 2),
                'rs_rank':       _ck_rs,
                'pct_from_high': None, 'vol_ratio': None,
                'cci_now':       round(max(_ck_wnow, 100.0), 1),
                'cci_prev':      round(_ck_wprev, 1),
                'pct_from_ma50': None, 'pct_from_ema21': None, 'pct_from_ema10': None,
                'pivot_high': None, 'pivot_bars_ago': None, 'pivot_crossed': False,
                'pct_from_pivot': None, 'pivot_high_w': None, 'pivot_bars_ago_w': None,
                'pivot_crossed_w': False, 'pct_from_pivot_w': None,
                'vcp_last_daily': None, 'vcp_last_weekly': None,
                'added_date':    _w_streak, 'is_new': (_w_streak == today_str),
            }
            screens['cci34_weekly_100']['stocks'].append(_wfill)
            if _ck_wnow >= 100 and _ck_wprev < 100:
                screens['cci34_weekly_cross_100']['stocks'].append(_wfill)
        except Exception:
            pass

    # ── CCI34 BEST SETUPS: Minervini full template / near breakout + CCI34 bullish ──
    # Build lookup: ticker → CCI entry (daily and weekly)
    _cci_daily_map  = {e['ticker']: e for e in screens['cci34_daily_100']['stocks']}
    _cci_weekly_map = {e['ticker']: e for e in screens['cci34_weekly_100']['stocks']}
    _cci_dcross_set = {e['ticker'] for e in screens['cci34_daily_cross_100']['stocks']}
    _cci_wcross_set = {e['ticker'] for e in screens['cci34_weekly_cross_100']['stocks']}

    # Minervini candidates: full template (8/8) + near breakout
    _minervini_stocks = {e['ticker']: e for e in screens['full_template']['stocks']}
    for e in screens['near_breakout']['stocks']:
        _minervini_stocks.setdefault(e['ticker'], e)

    for ticker, mentry in _minervini_stocks.items():
        d_entry = _cci_daily_map.get(ticker)
        w_entry = _cci_weekly_map.get(ticker)
        if d_entry is None and w_entry is None:
            continue  # no CCI34 bullish signal on either timeframe

        cci_daily  = d_entry['cci_now']  if d_entry  else None
        cci_weekly = w_entry['cci_now']  if w_entry  else None
        signals = []
        if ticker in _cci_dcross_set: signals.append('D-cross')
        if ticker in _cci_wcross_set: signals.append('W-cross')
        if d_entry and ticker not in _cci_dcross_set: signals.append('D≥100')
        if w_entry and ticker not in _cci_wcross_set: signals.append('W≥100')

        screens['cci34_best_setups']['stocks'].append({
            'ticker':        ticker,
            'tv_symbol':     mentry.get('tv_symbol', f'NSE:{ticker}'),
            'price':         mentry['price'],
            'rs_rank':       mentry['rs_rank'],
            'passed':          mentry.get('passed', 0),
            'pct_from_high':   mentry.get('pct_from_high'),
            'vol_ratio':       mentry.get('vol_ratio'),
            'cci_daily':       round(cci_daily,  1) if cci_daily  is not None else None,
            'cci_weekly':      round(cci_weekly, 1) if cci_weekly is not None else None,
            'cci_signals':     ', '.join(signals),
            'pct_from_ma50':   mentry.get('pct_from_ma50'),
            'pct_from_ema21':  mentry.get('pct_from_ema21'),
            'pct_from_ema10':  mentry.get('pct_from_ema10'),
            'pivot_high':       mentry.get('pivot_high'),
            'pivot_bars_ago':   mentry.get('pivot_bars_ago'),
            'pivot_crossed':    mentry.get('pivot_crossed', False),
            'pct_from_pivot':   mentry.get('pct_from_pivot'),
            'pivot_high_w':     mentry.get('pivot_high_w'),
            'pivot_bars_ago_w': mentry.get('pivot_bars_ago_w'),
            'pivot_crossed_w':  mentry.get('pivot_crossed_w', False),
            'pct_from_pivot_w': mentry.get('pct_from_pivot_w'),
            'added_date':       mentry.get('added_date', today_str),
            'vcp_last_daily':   mentry.get('vcp_last_daily'),
            'vcp_last_weekly':  mentry.get('vcp_last_weekly'),
        })

    # ── IPO SCREEN: evaluate recent listings (30-209 days) ────────────────────
    all_data_for_rs = {**data, **ipo_data}  # include IPO stocks in RS rank universe
    ipo_rs_ranks = calc_rs_ranks(all_data_for_rs)
    for sym, df in ipo_data.items():
        rs = ipo_rs_ranks.get(sym, 0)
        ipo_ev = evaluate_ipo(df, rs)
        if ipo_ev is None:
            continue
        ticker     = sym.replace('.NS', '')
        ipo_since  = record_and_get_date(history, 'ipo_watch', ticker, today_str)
        screens['ipo_watch']['stocks'].append({
            'ticker':           ticker,
            'price':            round(ipo_ev['price'], 2),
            'ma50':             ipo_ev['ma50'],
            'ema21':            ipo_ev.get('ema21'),
            'ema10':            ipo_ev.get('ema10'),
            'above_ma50':       ipo_ev['above_ma50'],
            'listing_high':     ipo_ev['listing_high'],
            'listing_low':      ipo_ev['listing_low'],
            'pct_from_high':    ipo_ev['pct_from_high'],
            'pct_above_low':    ipo_ev['pct_above_low'],
            'pct_from_ma50':    ipo_ev.get('pct_from_ma50'),
            'pct_from_ema21':   ipo_ev.get('pct_from_ema21'),
            'pct_from_ema10':   ipo_ev.get('pct_from_ema10'),
            'vol_ratio':        ipo_ev['vol_ratio'],
            'rs_rank':          ipo_ev['rs_rank'],
            'days_listed':      ipo_ev['days_listed'],
            'pivot_high':       ipo_ev.get('pivot_high'),
            'pivot_bars_ago':   ipo_ev.get('pivot_bars_ago'),
            'pivot_crossed':    ipo_ev.get('pivot_crossed', False),
            'pct_from_pivot':   ipo_ev.get('pct_from_pivot'),
            'pivot_high_w':     ipo_ev.get('pivot_high_w'),
            'pivot_bars_ago_w': ipo_ev.get('pivot_bars_ago_w'),
            'pivot_crossed_w':  ipo_ev.get('pivot_crossed_w', False),
            'pct_from_pivot_w': ipo_ev.get('pct_from_pivot_w'),
            'added_date':       ipo_since,
        })
    screens['ipo_watch']['stocks'].sort(key=lambda x: x['rs_rank'], reverse=True)

    # ── SAVE HISTORY ──────────────────────────────────────────────────────────
    active = {
        'c8':        {e['ticker'] for e in screens['c8']['stocks']},
        'ipo_watch': {e['ticker'] for e in screens['ipo_watch']['stocks']},
    }
    prune_history(history, active)
    save_history(history)

    # ── SORT & CAP ────────────────────────────────────────────────────────────
    for key, scr in screens.items():
        scr['stocks'].sort(key=lambda x: x['rs_rank'], reverse=True)
        if scr['group'] == 'individual' and INDIV_SCREEN_MAX:
            scr['stocks'] = scr['stocks'][:INDIV_SCREEN_MAX]

    # ── QUARTERLY FINANCIALS ENRICHMENT (CANSLIM C & A) ───────────────────────
    # Only fetch for composite + CCI screens (not individual C1-C8 — too many stocks)
    _QFIN_SCREENS = {
        'full_template', 'near_breakout', 'rs_leaders', 'vcp_setup',
        'new_highs', 'watch_list', 'minervini_backtest', 'ipo_watch', 'c1_to_c6',
        'cci34_daily_cross_100', 'cci34_weekly_cross_100',
        'cci34_daily_100', 'cci34_daily_neg100',
        'cci34_weekly_100', 'cci34_weekly_neg100', 'cci34_best_setups',
        'htf_setup', 'htf_potential', 'ma_pullback', 'hhhl_pullback',
        'primary_base_new', 'primary_base_10yr',
        'fo_momentum', 'fo_strong_uptrend', 'young_3yr', 'young_10yr',
    }
    _qfin_syms = set()
    for _key, _scr in screens.items():
        if _key in _QFIN_SCREENS:
            for _entry in _scr['stocks']:
                _qfin_syms.add(_entry['ticker'] + '.NS')

    _qfin = fetch_quarterly_financials(list(_qfin_syms))

    # Inject quarterly fields into every stock entry in every screen
    for _scr in screens.values():
        for _entry in _scr['stocks']:
            _sym = _entry['ticker'] + '.NS'
            _qf  = _qfin.get(_sym) or {}
            for _fld in QFIN_FIELDS:
                _entry[_fld] = _qf.get(_fld)

    # ── STRONG EARNINGS + CCI MOMENTUM ────────────────────────────────────────
    # Intersection: daily ≥100 ∩ weekly ≥100, filtered by strong quarterly results.
    # Built after qfin injection so financial fields are already populated.
    _d100_se = {e['ticker']: e for e in screens['cci34_daily_100']['stocks']}
    _w100_se = {e['ticker']: e for e in screens['cci34_weekly_100']['stocks']}
    _se_list = []
    for _sticker, _de in _d100_se.items():
        if _sticker not in _w100_se:
            continue
        _rev = _de.get('q_rev_yoy')
        _np  = _de.get('q_np_yoy')
        _eps = _de.get('q_eps_yoy')
        # CANSLIM 'C': EPS YoY ≥25% minimum; revenue must also be growing ≥20%
        _eps_ok = _eps is not None and _eps >= 25.0
        _rev_ok = _rev is not None and _rev >= 20.0
        if not (_eps_ok and _rev_ok):
            continue
        _we = _w100_se[_sticker]
        _se_entry = {**_de}
        _se_entry['cci_daily']  = round(float(_de.get('cci_now') or 0), 1)
        _se_entry['cci_weekly'] = round(float(_we.get('cci_now') or 0), 1)
        _se_list.append(_se_entry)
    # Sort by CANSLIM composite: EPS weighted 2× (most important), plus revenue + NP
    _se_list.sort(
        key=lambda e: (e.get('q_eps_yoy') or 0) * 2 + (e.get('q_rev_yoy') or 0) + (e.get('q_np_yoy') or 0),
        reverse=True
    )
    screens['strong_earnings']['stocks'] = _se_list

    # ── F&O MOMENTUM SCREEN ───────────────────────────────────────────────────
    # Collect all unique F&O stocks from the key screens.
    # For each stock, record which screens it appears in.
    _FO_SOURCE_SCREENS = [
        ('full_template',         'Full Template'),
        ('near_breakout',         'Near Breakout'),
        ('watch_list',            'Watch List'),
        ('rs_leaders',            'RS Leaders'),
        ('vcp_setup',             'VCP'),
        ('new_highs',             'New Highs'),
        ('cci34_daily_100',       'CCI-D≥100'),
        ('cci34_weekly_100',      'CCI-W≥100'),
        ('cci34_daily_cross_100', 'CCI-D Cross'),
        ('cci34_weekly_cross_100','CCI-W Cross'),
        ('cci34_best_setups',     'CCI Best'),
        ('strong_earnings',       'CANSLIM Earn'),
        ('htf_setup',             'HTF'),
        ('htf_potential',         'Pot.HTF'),
        ('ma_pullback',           'MA Pullback'),
        ('hhhl_pullback',         'HH/HL'),
    ]
    _fo_seen   = {}   # ticker → best entry dict
    _fo_labels = {}   # ticker → [screen labels]
    for _scr_key, _scr_label in _FO_SOURCE_SCREENS:
        for _e in screens.get(_scr_key, {}).get('stocks', []):
            _t = _e.get('ticker', '')
            if not _t or _t not in _fo_set:
                continue
            if _t not in _fo_seen:
                _fo_seen[_t]   = _e
                _fo_labels[_t] = []
            if _scr_label not in _fo_labels[_t]:
                _fo_labels[_t].append(_scr_label)

    _fo_entries = []
    for _t, _e in _fo_seen.items():
        _entry = {**_e, 'fo_screens': ', '.join(_fo_labels[_t])}
        _fo_entries.append(_entry)

    # Sort: most screens first, then by RS rank
    _fo_entries.sort(key=lambda e: (-len(_fo_labels.get(e['ticker'], [])), -e.get('rs_rank', 0)))
    screens['fo_momentum']['stocks'] = _fo_entries

    # ── F&O STRONG UPTREND (C1–C6 all met + F&O eligible) ────────────────────
    _fo_strong = []
    for _e in screens.get('c1_to_c6', {}).get('stocks', []):
        _t = _e.get('ticker', '')
        if not _t or _t not in _fo_set:
            continue
        _cr = _e.get('criteria', {})
        if _cr.get('c1') and _cr.get('c2') and _cr.get('c3') and _cr.get('c4') and _cr.get('c5') and _cr.get('c6'):
            _fo_strong.append({**_e, 'fo_screens': ', '.join(_fo_labels.get(_t, []))})
    _fo_strong.sort(key=lambda e: (-e.get('passed', 0), -e.get('rs_rank', 0)))
    screens['fo_strong_uptrend']['stocks'] = _fo_strong

    # ── YOUNG COMPANY SCREENS (≤3yr and ≤10yr) ────────────────────────────────
    # All unique stocks from key screens filtered by listing age from _listing_dates.
    _YOUNG_SOURCE_SCREENS = [
        ('full_template',          'Full Template'),
        ('near_breakout',          'Near Breakout'),
        ('watch_list',             'Watch List'),
        ('rs_leaders',             'RS Leaders'),
        ('vcp_setup',              'VCP'),
        ('new_highs',              'New Highs'),
        ('cci34_daily_100',        'CCI-D≥100'),
        ('cci34_weekly_100',       'CCI-W≥100'),
        ('cci34_daily_cross_100',  'CCI-D Cross'),
        ('cci34_weekly_cross_100', 'CCI-W Cross'),
        ('htf_setup',              'HTF'),
        ('htf_potential',          'Pot.HTF'),
        ('ma_pullback',            'MA Pullback'),
        ('hhhl_pullback',          'HH/HL'),
        ('primary_base_new',       'PrimaryBase'),
        ('primary_base_10yr',      'PrimaryBase'),
    ]
    _young_seen   = {}   # ticker → best entry
    _young_labels = {}   # ticker → [screen labels]
    for _scr_key, _scr_label in _YOUNG_SOURCE_SCREENS:
        for _e in screens.get(_scr_key, {}).get('stocks', []):
            _t = _e.get('ticker', '')
            if not _t:
                continue
            if _t not in _young_seen:
                _young_seen[_t]   = _e
                _young_labels[_t] = []
            if _scr_label not in _young_labels[_t]:
                _young_labels[_t].append(_scr_label)

    def _years_listed(ticker):
        ld = _listing_dates.get(ticker + '.NS')
        if not ld:
            return None
        try:
            return (datetime.today() - datetime.strptime(ld, '%Y-%m-%d')).days / 365.25
        except Exception:
            return None

    _y3_entries, _y10_entries = [], []
    for _t, _e in _young_seen.items():
        _yrs = _years_listed(_t)
        if _yrs is None:
            continue
        _entry = {**_e,
                  'fo_screens': ', '.join(_young_labels[_t]),
                  'listing_years': round(_yrs, 1)}
        if _yrs <= 3:
            _y3_entries.append(_entry)
        elif _yrs <= 10:
            _y10_entries.append(_entry)

    for _lst in [_y3_entries, _y10_entries]:
        _lst.sort(key=lambda e: (e.get('listing_years', 99), -e.get('rs_rank', 0)))

    screens['young_3yr']['stocks']  = _y3_entries
    screens['young_10yr']['stocks'] = _y10_entries

    # ── CHARTINK LIVE SCREENS ─────────────────────────────────────────────────
    # Build a lookup of enriched entries from our own scan for fast merging
    _scan_entry_map = {}  # ticker → best entry with full price/RS/VCP data
    for _src_key in ['full_template','near_breakout','watch_list','rs_leaders',
                     'cci34_daily_100','cci34_weekly_100','vcp_setup']:
        for _e in screens.get(_src_key, {}).get('stocks', []):
            _scan_entry_map.setdefault(_e['ticker'], _e)

    def _ck_build_entry(ck_ticker):
        """Build an entry for a Chartink stock. Use enriched scan data if available."""
        if ck_ticker in _scan_entry_map:
            e = dict(_scan_entry_map[ck_ticker])
            e['in_mtf'] = True
            return e
        # Not in our scan — try to read basic price data from cache
        _pkl = os.path.join(CACHE_DIR, ck_ticker + '_NS.pkl')
        base = {
            'ticker':    ck_ticker,
            'tv_symbol': f'NSE:{ck_ticker}',
            'in_mtf':    False,
            'rs_rank':   rs_ranks.get(ck_ticker + '.NS', 0),
            'price': None, 'pct_from_high': None, 'vol_ratio': None,
            'vcp_last_daily': None, 'vcp_last_weekly': None,
        }
        if os.path.exists(_pkl):
            try:
                with open(_pkl, 'rb') as _f: _cdf = pickle.load(_f)
                if isinstance(_cdf.columns, pd.MultiIndex):
                    _cdf.columns = _cdf.columns.get_level_values(0)
                base['price'] = round(float(_cdf['Close'].iloc[-1]), 2)
                _hi52 = float(_cdf['Close'].rolling(252, min_periods=1).max().iloc[-1])
                base['pct_from_high'] = round((base['price'] / _hi52 - 1) * 100, 1)
                _vol  = float(_cdf['Volume'].iloc[-1])
                _avgv = float(_cdf['Volume'].iloc[-50:].mean()) if len(_cdf) >= 50 else _vol
                base['vol_ratio'] = round(_vol / _avgv, 2) if _avgv else None
            except Exception:
                pass
        return base

    _CK_MAP = {
        'ck_daily_100':  'cci34_daily_100',
        'ck_weekly_100': 'cci34_weekly_100',
        'ck_daily_cross': 'cci34_daily_cross_100',
        'ck_weekly_cross': 'cci34_weekly_cross_100',
    }
    for _ck_screen, _ck_key in _CK_MAP.items():
        _entries = []
        for _ck_t in sorted(_ck.get(_ck_key, set())):
            _entries.append(_ck_build_entry(_ck_t))
        _entries.sort(key=lambda e: (0 if e.get('in_mtf') else 1, -e.get('rs_rank', 0)))
        screens[_ck_screen]['stocks'] = _entries

    # ── CHARTINK × MTF INTERSECTION ───────────────────────────────────────────
    # Stocks confirmed by Chartink (daily OR weekly >=100) AND in the live MTF universe.
    # Add ck_signals field showing which CCI signals are active.
    _ck_d100_set   = _ck.get('cci34_daily_100', set())
    _ck_w100_set   = _ck.get('cci34_weekly_100', set())
    _ck_dcross_set = _ck.get('cci34_daily_cross_100', set())
    _ck_wcross_set = _ck.get('cci34_weekly_cross_100', set())
    _mtf_tickers   = {sym.replace('.NS','') for sym in evaluations}  # live MTF scanned

    _mtf_100 = {}  # ticker → entry
    for _t in (_ck_d100_set | _ck_w100_set):
        if _t not in _mtf_tickers:
            continue
        _signals = []
        if _t in _ck_dcross_set: _signals.append('D-cross ⚡')
        elif _t in _ck_d100_set:  _signals.append('D≥100')
        if _t in _ck_wcross_set: _signals.append('W-cross ⚡')
        elif _t in _ck_w100_set:  _signals.append('W≥100')
        _e = dict(_ck_build_entry(_t))
        _e['ck_signals'] = ', '.join(_signals)
        _e['in_mtf'] = True
        _mtf_100[_t] = _e

    _mtf_entries = sorted(_mtf_100.values(),
        key=lambda e: (
            0 if ('cross' in e.get('ck_signals','')) else 1,  # cross signals first
            -e.get('rs_rank', 0)
        ))
    screens['ck_mtf_100']['stocks'] = _mtf_entries

    # ── NEWLY ADDED — diff vs previous scan ──────────────────────────────────
    # prev_screens.json was written at scan START from old results.json, so it
    # always contains the PREVIOUS scan's stock lists (not the current one).
    _prev_snap = {}
    if os.path.exists(PREV_SCREENS_JSON):
        try:
            with open(PREV_SCREENS_JSON) as _pf:
                _prev_snap = json.load(_pf)
        except Exception:
            pass
    _new_ticker_screens = {}  # ticker → [screen_label, ...]
    if _prev_snap:  # skip if no previous scan data
        for _trk_key, _trk_label in _TRACKED_SCREENS:
            _cur_tks = {e.get('ticker', '') for e in screens.get(_trk_key, {}).get('stocks', [])} - {''}
            _prv_tks = set(_prev_snap.get(_trk_key, []))
            for _t in (_cur_tks - _prv_tks):
                _new_ticker_screens.setdefault(_t, []).append(_trk_label)
    _newly = []
    for _t, _new_in in _new_ticker_screens.items():
        _ne = dict(_scan_entry_map[_t]) if _t in _scan_entry_map else _ck_build_entry(_t)
        _ne['new_in_screens'] = _new_in
        _newly.append(_ne)
    _newly.sort(key=lambda e: -e.get('rs_rank', 0))
    screens['newly_added']['stocks'] = _newly

    # ── BUILD RESULT ──────────────────────────────────────────────────────────
    result = {
        'generated_at':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scanned':        len(evaluations),
        'ipo_count':      len(screens['ipo_watch']['stocks']),
        'criteria_meta':  CRITERIA_META,
        'screens':        screens,
    }

    # Write results — NaN/Infinity → null so browser JSON.parse doesn't choke.
    # allow_nan=False raises BEFORE default() for float NaN, so we sanitize via regex.
    import re as _re
    def _sanitize(s):
        s = _re.sub(r'\bNaN\b', 'null', s)
        s = _re.sub(r'\bInfinity\b', 'null', s)
        s = _re.sub(r'\b-Infinity\b', 'null', s)
        return s

    # Write to temp file first, then atomic rename — crash never leaves truncated file
    tmp_json = RESULTS_JSON + '.tmp'
    tmp_js   = RESULTS_JS   + '.tmp'

    with open(tmp_json, 'w') as f:
        f.write(_sanitize(json.dumps(result, indent=2)))
    os.replace(tmp_json, RESULTS_JSON)

    with open(tmp_js, 'w') as f:
        f.write('window.SCAN_DATA = ')
        f.write(_sanitize(json.dumps(result)))
        f.write(';')
    os.replace(tmp_js, RESULTS_JS)

    # ── CCI HISTORY — append today's counts ──────────────────────────────────
    try:
        if os.path.exists(CCI_HISTORY_JSON):
            with open(CCI_HISTORY_JSON) as _f:
                _ch = json.load(_f)
        else:
            _ch = {'dates':[], 'ck_daily_100':[], 'ck_weekly_100':[], 'ck_mtf_100':[],
                   'ck_daily_cross':[], 'ck_weekly_cross':[],
                   'full_template':[], 'near_breakout':[], 'vcp_setup':[], 'c1_to_c6':[], 'regime':[]}
        _regime = result.get('market_analysis', {}).get('regime', '') if isinstance(result.get('market_analysis'), dict) else ''
        _today  = today_str
        # Update or append — one entry per date
        if _today in _ch['dates']:
            _idx = _ch['dates'].index(_today)
            _ch['ck_daily_100'][_idx]   = len(screens['ck_daily_100']['stocks'])
            _ch['ck_weekly_100'][_idx]  = len(screens['ck_weekly_100']['stocks'])
            _ch['ck_mtf_100'][_idx]     = len(screens['ck_mtf_100']['stocks'])
            _ch['ck_daily_cross'][_idx] = len(screens['ck_daily_cross']['stocks'])
            _ch['ck_weekly_cross'][_idx]= len(screens['ck_weekly_cross']['stocks'])
            _ch['full_template'][_idx]  = len(screens['full_template']['stocks'])
            _ch['near_breakout'][_idx]  = len(screens['near_breakout']['stocks'])
            _ch['vcp_setup'][_idx]      = len(screens['vcp_setup']['stocks'])
            _ch['c1_to_c6'][_idx]       = len(screens['c1_to_c6']['stocks'])
            _ch['regime'][_idx]         = _regime
        else:
            _ch['dates'].append(_today)
            _ch['ck_daily_100'].append(len(screens['ck_daily_100']['stocks']))
            _ch['ck_weekly_100'].append(len(screens['ck_weekly_100']['stocks']))
            _ch['ck_mtf_100'].append(len(screens['ck_mtf_100']['stocks']))
            _ch['ck_daily_cross'].append(len(screens['ck_daily_cross']['stocks']))
            _ch['ck_weekly_cross'].append(len(screens['ck_weekly_cross']['stocks']))
            _ch['full_template'].append(len(screens['full_template']['stocks']))
            _ch['near_breakout'].append(len(screens['near_breakout']['stocks']))
            _ch['vcp_setup'].append(len(screens['vcp_setup']['stocks']))
            _ch['c1_to_c6'].append(len(screens['c1_to_c6']['stocks']))
            _ch['regime'].append(_regime)
        _ch['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _tmp_ch = CCI_HISTORY_JSON + '.tmp'
        with open(_tmp_ch, 'w') as _f:
            json.dump(_ch, _f)
        os.replace(_tmp_ch, CCI_HISTORY_JSON)
    except Exception as _ex:
        print(f"  CCI history save failed: {_ex}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"{'=' * 55}")
    print(f"Scan complete — {result['generated_at']}")
    print(f"Universe scanned: {len(evaluations)} stocks")
    print(f"{'=' * 55}")
    print(f"\n  INDIVIDUAL CRITERIA SCREENS (all matching, sorted by RS rank):")
    for key in ['c1','c2','c3','c4','c5','c6','c7','c8']:
        print(f"    {screens[key]['label']:<45} {len(screens[key]['stocks'])} stocks")
    print(f"\n  COMPOSITE SCREENS:")
    for key in ['full_template','near_breakout','rs_leaders','vcp_setup','new_highs','watch_list','c1_to_c6','minervini_backtest']:
        print(f"    {screens[key]['label']:<45} {len(screens[key]['stocks'])} stocks")
    print(f"    {screens['newly_added']['label']:<45} {len(screens['newly_added']['stocks'])} stocks  ← new since last scan")
    print(f"\n  RECENT IPOs & LISTINGS:")
    print(f"    {screens['ipo_watch']['label']:<45} {len(screens['ipo_watch']['stocks'])} stocks")
    print(f"\n  STRONG EARNINGS + CCI:")
    print(f"    {screens['strong_earnings']['label']:<45} {len(screens['strong_earnings']['stocks'])} stocks")
    print(f"\n  PRIMARY BASE:")
    print(f"    {screens['primary_base_new']['label']:<45} {len(screens['primary_base_new']['stocks'])} stocks")
    print(f"    {screens['primary_base_10yr']['label']:<45} {len(screens['primary_base_10yr']['stocks'])} stocks")
    print(f"\n  F&O MOMENTUM:")
    print(f"    {screens['fo_momentum']['label']:<45} {len(screens['fo_momentum']['stocks'])} stocks")
    print(f"    {screens['fo_strong_uptrend']['label']:<45} {len(screens['fo_strong_uptrend']['stocks'])} stocks")
    print(f"\n  YOUNG COMPANIES ON DASHBOARD:")
    print(f"    {screens['young_3yr']['label']:<45} {len(screens['young_3yr']['stocks'])} stocks")
    print(f"    {screens['young_10yr']['label']:<45} {len(screens['young_10yr']['stocks'])} stocks")
    print(f"\n  CHARTINK LIVE:")
    for _cks in ['ck_daily_100','ck_weekly_100','ck_daily_cross','ck_weekly_cross','ck_mtf_100']:
        print(f"    {screens[_cks]['label']:<45} {len(screens[_cks]['stocks'])} stocks")
    print(f"\n  Results saved:")
    print(f"    {RESULTS_JS}")
    print(f"    {RESULTS_JSON}")
    print(f"\n  Open dashboard.html in your browser to view results.\n")

    # Compute market breadth history (uses cached data if fresh)
    all_ns_syms = list(data.keys()) + list(ipo_data.keys())
    compute_breadth_history(all_ns_syms)

    # ── MARKET ANALYSIS ───────────────────────────────────────────────────────
    # Generate professional market analysis from breadth.json and embed in results
    try:
        market_analysis = generate_market_analysis()
        if market_analysis:
            result['market_analysis'] = market_analysis
            # Re-write results files with market_analysis included
            tmp_json2 = RESULTS_JSON + '.tmp'
            tmp_js2   = RESULTS_JS   + '.tmp'
            with open(tmp_json2, 'w') as f:
                f.write(_sanitize(json.dumps(result, indent=2)))
            os.replace(tmp_json2, RESULTS_JSON)
            with open(tmp_js2, 'w') as f:
                f.write('window.SCAN_DATA = ')
                f.write(_sanitize(json.dumps(result)))
                f.write(';')
            os.replace(tmp_js2, RESULTS_JS)
            print(f"  Market analysis embedded in results (regime: {market_analysis.get('regime','?')})")
            # ── CCI HISTORY — update regime now that we have it ───────────────
            try:
                if os.path.exists(CCI_HISTORY_JSON):
                    with open(CCI_HISTORY_JSON) as _f2:
                        _ch2 = json.load(_f2)
                    _reg2 = market_analysis.get('regime', '')
                    if today_str in _ch2.get('dates', []):
                        _idx2 = _ch2['dates'].index(today_str)
                        _ch2['regime'][_idx2] = _reg2
                        _tmp2 = CCI_HISTORY_JSON + '.tmp'
                        with open(_tmp2, 'w') as _f2:
                            json.dump(_ch2, _f2)
                        os.replace(_tmp2, CCI_HISTORY_JSON)
                        # Write embed JS for file:// mode
                        _tmp_cjs = CCI_HISTORY_JS + '.tmp'
                        with open(_tmp_cjs, 'w') as _f3:
                            _f3.write('window.CCI_HISTORY_DATA = ')
                            _f3.write(json.dumps(_ch2))
                            _f3.write(';')
                        os.replace(_tmp_cjs, CCI_HISTORY_JS)
            except Exception as _rex:
                print(f"  CCI history regime update failed: {_rex}")
    except Exception as e:
        print(f"  [WARN] Market analysis generation failed: {e}")

def fetch_fiidii_history_auto():
    """
    Scrape full historical FII/DII data from NSE using curl_cffi.
    NSE's fiidiiTradeReact endpoint only returns the latest day. To get multi-year
    history we call it daily (via scanner) — this function bulk-seeds the store
    by fetching from the NSE historical page HTML which embeds the latest 90 days,
    AND by importing fiidii_history.csv if present.

    Additionally tries to reconstruct history by fetching the NSE data for each
    Nifty50 trading date over the past 5 years (rate-limited to avoid blocking).
    """
    print("\n  FII/DII Historical Fetch Mode")
    print("  ─────────────────────────────────────────")

    # Load existing stored data
    rows = {}
    if os.path.exists(FIIDII_PKL):
        try:
            with open(FIIDII_PKL, 'rb') as f:
                stored = pickle.load(f)
            for r in (stored or []):
                if isinstance(r, dict) and r.get('date'):
                    rows[r['date']] = r
            print(f"  Loaded {len(rows)} existing dates from store")
        except Exception:
            pass

    # Import CSV if present
    csv_path = os.path.join(BASE_DIR, 'fiidii_history.csv')
    if os.path.exists(csv_path):
        def _safe_float(v):
            try: return round(float(str(v).replace(',','').replace('(','-').replace(')','').strip()), 2)
            except: return 0.0
        def _parse_date(raw):
            for fmt in ('%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%B %d, %Y'):
                try:
                    from datetime import datetime as _dt
                    return _dt.strptime(str(raw).strip(), fmt).strftime('%Y-%m-%d')
                except: pass
            return None
        try:
            df_csv = pd.read_csv(csv_path)
            df_csv.columns = [c.strip().lower().replace(' ','_') for c in df_csv.columns]
            before = len(rows)
            for _, row in df_csv.iterrows():
                dt = _parse_date(row.get('date',''))
                if not dt or dt in rows:
                    continue
                def _col(*names):
                    for n in names:
                        v = row.get(n)
                        if v is not None and str(v).strip() not in ('','nan'):
                            return _safe_float(v)
                    return 0.0
                rows[dt] = {
                    'date': dt,
                    'fii_buy':  _col('fii_buy_value','fii_buy','buy_value_(fii)'),
                    'fii_sell': _col('fii_sell_value','fii_sell','sell_value_(fii)'),
                    'fii_net':  _col('fii_net_value','fii_net','net_value_(fii)'),
                    'dii_buy':  _col('dii_buy_value','dii_buy','buy_value_(dii)'),
                    'dii_sell': _col('dii_sell_value','dii_sell','sell_value_(dii)'),
                    'dii_net':  _col('dii_net_value','dii_net','net_value_(dii)'),
                }
            added = len(rows) - before
            print(f"  CSV import: +{added} dates from {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"  CSV import error: {e}")
    else:
        print(f"  No fiidii_history.csv found at {csv_path}")
        print(f"  TIP: Download from NSE India → All Reports → Historical Equities → FII/FPI/DII Trading Activity")
        print(f"       Save as fiidii_history.csv in the scanner directory, then re-run --fetch-fiidii-history")

    # Fetch latest day (NSE API works without auth via curl_cffi)
    print(f"  Fetching latest day from NSE API...")
    try:
        from curl_cffi import requests as _cffi_req
        sess = _cffi_req.Session(impersonate='chrome124')
        sess.get('https://www.nseindia.com', timeout=10)
        today_str = datetime.now().strftime('%d-%m-%Y')
        r = sess.get(f'https://www.nseindia.com/api/fiidiiTradeReact?date={today_str}',
                     headers={'Referer': 'https://www.nseindia.com/'}, timeout=12)
        items = r.json()
        if isinstance(items, list) and items:
            def _sf(v):
                try: return round(float(str(v).replace(',','').replace('(','-').replace(')','').strip()), 2)
                except: return 0.0
            def _pd(raw):
                for fmt in ('%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y'):
                    try:
                        from datetime import datetime as _dt
                        return _dt.strptime(str(raw).strip(), fmt).strftime('%Y-%m-%d')
                    except: pass
                return None
            day = {}
            for it in items:
                cat = str(it.get('category','')).upper()
                dt = _pd(it.get('date',''))
                if not dt: continue
                if 'FII' in cat or 'FPI' in cat:
                    day.update({'dt': dt, 'fii_buy': _sf(it.get('buyValue',0)),
                                'fii_sell': _sf(it.get('sellValue',0)), 'fii_net': _sf(it.get('netValue',0))})
                elif 'DII' in cat:
                    day.update({'dt': dt, 'dii_buy': _sf(it.get('buyValue',0)),
                                'dii_sell': _sf(it.get('sellValue',0)), 'dii_net': _sf(it.get('netValue',0))})
            if 'dt' in day:
                dt = day.pop('dt')
                if dt not in rows:
                    rows[dt] = {'date': dt, 'fii_buy': 0, 'fii_sell': 0, 'fii_net': 0,
                                'dii_buy': 0, 'dii_sell': 0, 'dii_net': 0, **day}
                    print(f"  Added {dt} from NSE API")
                else:
                    print(f"  Latest NSE date {dt} already stored")
    except Exception as e:
        print(f"  NSE API fetch failed: {e}")

    # Save
    result = sorted(rows.values(), key=lambda x: x['date'])
    if result:
        try:
            with open(FIIDII_PKL, 'wb') as f:
                pickle.dump(result, f)
            print(f"\n  Saved {len(result)} FII/DII dates ({result[0]['date']} → {result[-1]['date']})")
        except Exception as e:
            print(f"  Save failed: {e}")
    else:
        print("  No data to save.")
    print()


if __name__ == '__main__':
    if '--fetch-fiidii-history' in sys.argv:
        fetch_fiidii_history_auto()
    else:
        run()
