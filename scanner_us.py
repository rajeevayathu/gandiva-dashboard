#!/usr/bin/env python3
"""
Minervini Trend Template Scanner — US Markets
Runs the same screens as the India scanner on US stocks (S&P 500 + Nasdaq 100 + Russell 1000).
Outputs results_us_embed.js and results_us.json for the dashboard.

Usage:
  python3 scanner_us.py              # full scan
  python3 scanner_us.py --fast       # quick test with 100 stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time, pickle, re
import urllib.request, urllib.parse
import concurrent.futures
from datetime import datetime

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_JS     = os.path.join(BASE_DIR, 'results_us_embed.js')
RESULTS_JSON   = os.path.join(BASE_DIR, 'results_us.json')
CACHE_DIR      = os.path.join(BASE_DIR, '.cache_us')
CACHE_TTL_SEC  = 23 * 3600
QFIN_CACHE     = os.path.join(CACHE_DIR, '_qfin_cache_us.pkl')
QFIN_TTL       = 7 * 24 * 3600
QFIN_TTL_NULL  = 6 * 3600
LISTING_CACHE  = os.path.join(CACHE_DIR, '_listing_dates_us.pkl')
US_SYMBOLS_CACHE = os.path.join(CACHE_DIR, '_us_symbols.pkl')
US_SYMBOLS_TTL   = 7 * 24 * 3600   # refresh weekly

os.makedirs(CACHE_DIR, exist_ok=True)

# ── US SYMBOL LOADING ─────────────────────────────────────────────────────────

# Extra high-growth / Nasdaq stocks not always in S&P 500
_EXTRA_US_STOCKS = [
    # Mega-cap tech & growth
    'GOOGL','GOOG','META','TSLA','NVDA','AVGO','AMD','QCOM','INTC',
    # Cloud / SaaS
    'NOW','CRM','SNOW','DDOG','NET','ZS','PANW','CRWD','FTNT','OKTA','S',
    'HUBS','TTD','APP','BILL','TOST','ZI','ESTC','MDB','CFLT','GTLB',
    # Fintech
    'COIN','HOOD','SQ','AFRM','SOFI','NU','MELI','SE',
    # Consumer / E-commerce
    'SHOP','ABNB','DASH','RBLX','GRAB',
    # Biotech / Health
    'DXCM','PODD','INSP','IRTC','ALGN','SWAV','ELF','CELH','DUOL',
    # Semiconductors (smaller)
    'SMCI','ONTO','ACLS','ENTG','WOLF','AMBA','MPWR','MCHP','LRCX',
    # Energy / Clean tech
    'ENPH','FSLR','CEG','VST','ETN','PWR',
    # Defence / Space
    'AXON','PLTR','SOUN','AI','RKLB','RDW','ASTS','LUNR',
    # ETFs used as market gauges (excluded from individual stock logic)
]

def fetch_us_symbols():
    """
    Fetch S&P 500 from GitHub-hosted CSV (reliable, no auth needed) + extra growth stocks.
    Falls back to hardcoded list if all network sources fail.
    Cache is refreshed weekly.
    """
    if os.path.exists(US_SYMBOLS_CACHE):
        age = time.time() - os.path.getmtime(US_SYMBOLS_CACHE)
        if age < US_SYMBOLS_TTL:
            with open(US_SYMBOLS_CACHE, 'rb') as f:
                syms = pickle.load(f)
            print(f"  US symbols: {len(syms)} stocks (cached)", flush=True)
            return syms

    print("  Fetching US symbol lists...", flush=True)
    syms = set()

    # Primary: GitHub-hosted S&P 500 CSV (doesn't block like Wikipedia)
    sp500_sources = [
        'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv',
        'https://raw.githubusercontent.com/lukelukem/sp500/main/sp500.csv',
    ]
    for url in sp500_sources:
        try:
            df_sp = pd.read_csv(url)
            for col in ['Symbol', 'symbol', 'Ticker', 'ticker']:
                if col in df_sp.columns:
                    tickers = [str(t).strip().replace('.', '-') for t in df_sp[col].dropna()
                               if str(t).strip() and len(str(t).strip()) <= 6]
                    syms.update(tickers)
                    print(f"    S&P 500 (GitHub): {len(tickers)} symbols", flush=True)
                    break
            if syms:
                break
        except Exception as e:
            print(f"    [WARN] {url}: {e}", flush=True)

    # Secondary: Wikipedia (may work sometimes)
    if len(syms) < 400:
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            tickers = [str(t).strip().replace('.', '-') for t in tables[0]['Symbol'].dropna()]
            syms.update(tickers)
            print(f"    S&P 500 (Wikipedia): {len(tickers)} symbols", flush=True)
        except Exception:
            pass

    # Always add extra high-growth stocks
    syms.update(_EXTRA_US_STOCKS)

    if len(syms) < 100:
        # Last-resort hardcoded core list
        print("  Using hardcoded fallback list", flush=True)
        syms = set(['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','JPM','LLY',
                    'V','MA','UNH','XOM','JNJ','WMT','HD','BAC','PG','COST','CVX',
                    'ABBV','CRM','ACN','NFLX','AMD','TMO','PEP','MCD','ORCL','ABT','GE',
                    'NOW','ISRG','INTU','QCOM','IBM','GS','TXN','UBER','CAT','BA','HON',
                    'PANW','CRWD','SNOW','DDOG','NET','ZS','COIN','SHOP','MELI','ABNB',
                    'PLTR','APP','AXON','TTD','HUBS','SMCI','ENPH','FSLR','ETN','PWR'])
        syms.update(_EXTRA_US_STOCKS)

    syms = sorted(s for s in syms if s and len(s) <= 6 and not s.startswith('^'))
    with open(US_SYMBOLS_CACHE, 'wb') as f:
        pickle.dump(syms, f)
    print(f"  US symbols: {len(syms)} stocks total", flush=True)
    return syms


# ── PRICE DATA LOADING ─────────────────────────────────────────────────────────
def cache_path(sym):
    return os.path.join(CACHE_DIR, sym.replace('-', '_').replace('.', '_') + '.pkl')

def load_ticker(sym):
    cp = cache_path(sym)
    if os.path.exists(cp):
        if time.time() - os.path.getmtime(cp) < CACHE_TTL_SEC:
            try:
                with open(cp, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
    try:
        df = yf.download(sym, period='2y', interval='1d', progress=False,
                         auto_adjust=True, timeout=15)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or len(df) < 30:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 30:
            return None
        with open(cp, 'wb') as f:
            pickle.dump(df, f)
        return df
    except Exception:
        return None


def load_all_data(symbols, max_workers=20):
    print(f"  Downloading/loading price data for {len(symbols)} stocks...", flush=True)
    data = {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(load_ticker, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            sym = futures[fut]
            df = fut.result()
            if df is not None and len(df) >= 50:
                data[sym] = df
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(symbols)}...", flush=True)
    print(f"  Loaded {len(data)} stocks with sufficient data.", flush=True)
    return data


# ── REUSE CORE LOGIC FROM scanner.py ─────────────────────────────────────────
# Import shared functions directly
sys.path.insert(0, BASE_DIR)
from scanner import (
    calc_rs_ranks, evaluate, calc_cci, calc_cci_streak_start,
    detect_vcp, detect_htf, detect_hhhl, detect_primary_base,
    find_last_vcp_date, CRITERIA_META, QFIN_FIELDS,
    _fetch_one_qfin, get_listing_dates,
)

def build_entry_us(sym, ev, extra=None):
    """Same as build_entry but uses NASDAQ/NYSE TradingView prefix."""
    e = {
        'ticker':        sym,
        'tv_symbol':     'NASDAQ:' + sym,   # dashboard will fix to correct exchange
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


# ── QUARTERLY FINANCIALS ──────────────────────────────────────────────────────
def fetch_quarterly_financials_us(syms):
    """Same as India version but without .NS suffix."""
    cache = {}
    if os.path.exists(QFIN_CACHE):
        try:
            with open(QFIN_CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    now = time.time()
    to_fetch = []
    for sym in syms:
        cached = cache.get(sym)
        if cached is None:
            to_fetch.append(sym)
        elif cached.get('data') is None and now - cached.get('ts', 0) < QFIN_TTL_NULL:
            pass
        elif now - cached.get('ts', 0) > QFIN_TTL:
            to_fetch.append(sym)

    if to_fetch:
        print(f"  Fetching quarterly results for {len(to_fetch)} US stocks...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_fetch_one_qfin, sym): sym for sym in to_fetch}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                orig_sym = futures[fut]
                _, result = fut.result()
                cache[orig_sym] = {'data': result, 'ts': time.time()}
                done += 1

        with open(QFIN_CACHE, 'wb') as f:
            pickle.dump(cache, f)

    result = {}
    for sym in syms:
        cached = cache.get(sym)
        if cached and cached.get('data'):
            result[sym] = cached['data']
    return result


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────
def run_scan(max_syms=None):
    print(f"\n{'='*55}")
    print("  US MARKET SCANNER — Minervini Trend Template")
    print(f"{'='*55}\n")

    symbols = fetch_us_symbols()
    if max_syms:
        symbols = symbols[:max_syms]

    data = load_all_data(symbols)

    print("Calculating RS ranks...", flush=True)
    # calc_rs_ranks expects {sym: df} — works same for US
    rs_ranks = calc_rs_ranks(data)

    print("Running screens...\n", flush=True)

    evaluations = {}
    for sym, df in data.items():
        rs = rs_ranks.get(sym, 0)
        ev = evaluate(df, rs)
        if ev:
            evaluations[sym] = (ev, df)

    today_str = datetime.now().strftime('%Y-%m-%d')

    screens = {
        'c1': {'label': 'C1 — Price above MA150 & MA200', 'desc': 'Current price above both the 150-day and 200-day MAs.', 'group': 'individual', 'stocks': []},
        'c2': {'label': 'C2 — MA150 above MA200',         'desc': '150-day MA above 200-day MA.',                         'group': 'individual', 'stocks': []},
        'c3': {'label': 'C3 — MA200 trending up (1m)',     'desc': '200-day MA higher than ~21 days ago.',                  'group': 'individual', 'stocks': []},
        'c4': {'label': 'C4 — MA50 above MA150 & MA200',  'desc': '50-day MA above 150-day and 200-day MAs.',              'group': 'individual', 'stocks': []},
        'c5': {'label': 'C5 — Price above MA50',          'desc': 'Price above 50-day MA.',                                'group': 'individual', 'stocks': []},
        'c6': {'label': 'C6 — 25%+ above 52-week low',    'desc': 'Price at least 25% above its 52-week low.',             'group': 'individual', 'stocks': []},
        'c7': {'label': 'C7 — Within 25% of 52-week high','desc': 'Price within 25% of 52-week high.',                     'group': 'individual', 'stocks': []},
        'c8': {'label': 'C8 — RS Rank >= 70',             'desc': 'Relative Strength rank ≥ 70 within US universe.',       'group': 'individual', 'stocks': []},
        'full_template':  {'label': 'Full Trend Template (8/8)',          'desc': 'All 8 Minervini criteria met.',       'group': 'composite', 'stocks': []},
        'near_breakout':  {'label': 'Near Breakout (within 5% of high)',  'desc': 'Within 5% of 52-week high.',          'group': 'composite', 'stocks': []},
        'rs_leaders':     {'label': 'RS Leaders (RS >= 85)',              'desc': 'RS rank ≥ 85.',                       'group': 'composite', 'stocks': []},
        'vcp_setup':      {'label': 'VCP Setup',                         'desc': 'Volatility Contraction Pattern.',      'group': 'composite', 'stocks': []},
        'new_highs':      {'label': 'New 52-Week Highs (within 2%)',      'desc': 'Within 2% of 52-week high.',          'group': 'composite', 'stocks': []},
        'watch_list':     {'label': 'Watch List — 7/8 Criteria',          'desc': '7 of 8 criteria met.',                'group': 'composite', 'stocks': []},
        'c1_to_c6':      {'label': 'C1–C6 — Stage 2 Uptrend Structure',  'desc': 'All 6 trend structure criteria met — confirmed Stage 2 uptrend.', 'group': 'composite', 'stocks': []},
        'htf_setup':      {'label': 'High Tight Flag',                    'desc': 'Pole ≥90% in ≤40 days, tight flag.', 'group': 'htf',       'stocks': []},
        'htf_potential':  {'label': 'Potential High Tight Flag',          'desc': 'Pole ≥50%, near peak.',               'group': 'htf',       'stocks': []},
        'ma_pullback':    {'label': 'MA Pullback — Trend Rider',          'desc': 'Above MA50/150/200, pulling back to key MA.', 'group': 'htf', 'stocks': []},
        'hhhl_pullback':  {'label': 'HH/HL Pullback',                     'desc': 'Higher highs and higher lows, near recent HL.', 'group': 'htf', 'stocks': []},
        'cci34_daily_100':       {'label': 'CCI34 — Daily ≥ 100',               'desc': 'Daily CCI34 ≥ 100.',                  'group': 'cci34', 'stocks': []},
        'cci34_weekly_100':      {'label': 'CCI34 — Weekly ≥ 100',              'desc': 'Weekly CCI34 ≥ 100.',                 'group': 'cci34', 'stocks': []},
        'cci34_daily_cross_100': {'label': 'CCI34 — Daily just crossed 100',    'desc': 'Daily CCI34 just crossed above 100.', 'group': 'cci34', 'stocks': []},
        'cci34_weekly_cross_100':{'label': 'CCI34 — Weekly just crossed 100',   'desc': 'Weekly CCI34 just crossed 100.',      'group': 'cci34', 'stocks': []},
        'cci34_best_setups':     {'label': 'CCI34 — Best Setups',               'desc': 'Full Template + CCI34 bullish.',      'group': 'cci34', 'stocks': []},
        'primary_base_new':  {'label': 'Primary Base — Recent IPO (≤3 yrs)',   'desc': 'Recent IPO in first tight base near ATH.', 'group': 'htf', 'stocks': []},
        'primary_base_10yr': {'label': 'Primary Base — Young Company (≤10 yrs)','desc': 'Young company in primary base.', 'group': 'htf', 'stocks': []},
    }

    # ── LISTING DATES ─────────────────────────────────────────────────────────
    _listing_dates = {}
    if os.path.exists(LISTING_CACHE):
        try:
            with open(LISTING_CACHE, 'rb') as f:
                _listing_dates = pickle.load(f)
        except Exception:
            _listing_dates = {}
    to_fetch_ld = [s for s in evaluations if s not in _listing_dates]
    if to_fetch_ld:
        print(f"  Listing dates: fetching {len(to_fetch_ld)} new symbols...", flush=True)
        for sym in to_fetch_ld:
            try:
                df_max = yf.download(sym, period='max', interval='1d', progress=False,
                                     auto_adjust=True, timeout=15)
                if isinstance(df_max.columns, pd.MultiIndex):
                    df_max.columns = df_max.columns.get_level_values(0)
                _listing_dates[sym] = df_max.index[0].strftime('%Y-%m-%d') if len(df_max) > 0 else None
            except Exception:
                _listing_dates[sym] = None
        with open(LISTING_CACHE, 'wb') as f:
            pickle.dump(_listing_dates, f)

    # ── CRITERIA SCREENS ──────────────────────────────────────────────────────
    INDIV_CRITERIA = {
        'c1': lambda cr: cr['c1'], 'c2': lambda cr: cr['c2'],
        'c3': lambda cr: cr['c3'], 'c4': lambda cr: cr['c4'],
        'c5': lambda cr: cr['c5'], 'c6': lambda cr: cr['c6'],
        'c7': lambda cr: cr['c7'], 'c8': lambda cr: cr['c8'],
    }

    for sym, (ev, df) in evaluations.items():
        cr     = ev['criteria']
        passed = ev['passed']

        for key, fn in INDIV_CRITERIA.items():
            if fn(cr):
                screens[key]['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))

        if passed == 8:
            screens['full_template']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))
        if passed >= 7:
            screens['watch_list']['stocks'].append(build_entry_us(sym, ev, {'failed_criteria': [k for k,v in cr.items() if not v], 'added_date': today_str}))
        if cr['c1'] and cr['c2'] and cr['c3'] and cr['c4'] and cr['c5'] and cr['c6']:
            screens['c1_to_c6']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))
        if passed == 8 and ev['pct_from_high'] >= -5:
            screens['near_breakout']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))
        if ev['rs_rank'] >= 85:
            screens['rs_leaders']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))
        if ev['pct_from_high'] >= -2:
            screens['new_highs']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str}))

        # VCP
        if detect_vcp(df):
            vcp_d = find_last_vcp_date(df, weekly=False)
            vcp_w = find_last_vcp_date(df, weekly=True)
            screens['vcp_setup']['stocks'].append(build_entry_us(sym, ev, {'vcp': True, 'added_date': today_str, 'vcp_last_daily': vcp_d, 'vcp_last_weekly': vcp_w}))

        # HTF
        _htf_full, _htf_pot, _htf_det = detect_htf(df)
        if _htf_full and _htf_det:
            screens['htf_setup']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, **_htf_det}))
        elif _htf_pot and _htf_det:
            screens['htf_potential']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, **_htf_det}))

        # MA Pullback
        try:
            _c    = df['Close'].squeeze()
            _p    = float(_c.iloc[-1])
            _e10  = float(_c.ewm(span=10,  adjust=False).mean().iloc[-1])
            _e21  = float(_c.ewm(span=21,  adjust=False).mean().iloc[-1])
            _ma50 = float(_c.rolling(50).mean().iloc[-1])
            _ma150= float(_c.rolling(150).mean().iloc[-1]) if len(_c)>=150 else None
            _ma200= float(_c.rolling(200).mean().iloc[-1]) if len(_c)>=200 else None
            _above_mas  = _p>_ma50>0 and (_ma150 is None or _p>_ma150) and (_ma200 is None or _p>_ma200)
            _ma_stacked = _e10>_e21>_ma50>0
            _near10  = abs(_p-_e10)/_e10   <= 0.03
            _near21  = abs(_p-_e21)/_e21   <= 0.04
            _near50  = abs(_p-_ma50)/_ma50 <= 0.05
            _hi52_ma = float(df['High'].iloc[-252:].max()) if len(df)>=252 else float(df['High'].max())
            _near_high = _p >= _hi52_ma*0.75
            if _above_mas and _ma_stacked and (_near10 or _near21 or _near50) and _near_high:
                _which = 'EMA10' if _near10 else ('EMA21' if _near21 else 'MA50')
                screens['ma_pullback']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, 'ma_pullback_near': _which}))
        except Exception:
            pass

        # HH/HL
        hhhl_ok, hhhl_det = detect_hhhl(df)
        if hhhl_ok:
            screens['hhhl_pullback']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, **(hhhl_det or {})}))

        # Primary Base
        _ld = _listing_dates.get(sym)
        _pb_new_ok,  _pb_new_det  = detect_primary_base(df, _ld, max_years=3)
        _pb_10yr_ok, _pb_10yr_det = detect_primary_base(df, _ld, max_years=10)
        if _pb_new_ok:
            screens['primary_base_new']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, **(_pb_new_det or {})}))
        elif _pb_10yr_ok:
            screens['primary_base_10yr']['stocks'].append(build_entry_us(sym, ev, {'added_date': today_str, **(_pb_10yr_det or {})}))

        # CCI
        cci_d_now, cci_d_prev = calc_cci(df, period=34, weekly=False)
        cci_w_now, cci_w_prev = calc_cci(df, period=34, weekly=True)

        def _cci_entry_us(cci_now, cci_prev):
            _c2   = df['Close'].squeeze()
            _p2   = float(_c2.iloc[-1])
            _e10v = float(_c2.ewm(span=10, adjust=False).mean().iloc[-1])
            _e21v = float(_c2.ewm(span=21, adjust=False).mean().iloc[-1])
            _m50v = float(_c2.rolling(50).mean().iloc[-1])
            _hi2  = float(df['High'].iloc[-252:].max()) if len(df)>=252 else float(df['High'].max())
            _vol  = float(df['Volume'].iloc[-1])
            _avgv = float(df['Volume'].iloc[-50:].mean()) if len(df)>=50 else _vol
            return {
                'ticker': sym, 'tv_symbol': 'NASDAQ:'+sym,
                'price': round(_p2,2), 'rs_rank': int(ev['rs_rank']),
                'passed': ev['passed'], 'criteria': ev['criteria'],
                'cci_now': round(cci_now,1), 'cci_prev': round(cci_prev,1),
                'pct_from_high': round((_p2/_hi2-1)*100, 1),
                'vol_ratio': round(_vol/_avgv, 2) if _avgv else None,
                'pct_from_ma50':  round((_p2-_m50v)/_m50v*100, 1) if _m50v else None,
                'pct_from_ema21': round((_p2-_e21v)/_e21v*100, 1) if _e21v else None,
                'pct_from_ema10': round((_p2-_e10v)/_e10v*100, 1) if _e10v else None,
                'pivot_high': None, 'pivot_bars_ago': None, 'pivot_crossed': False,
                'pct_from_pivot': None, 'pivot_high_w': None, 'pivot_bars_ago_w': None,
                'pivot_crossed_w': False, 'pct_from_pivot_w': None,
                'vcp_last_daily': ev.get('vcp_last_daily'),
                'vcp_last_weekly': ev.get('vcp_last_weekly'),
                'added_date': today_str,
            }

        if cci_d_now is not None and cci_d_prev is not None:
            if cci_d_now >= 100 and cci_d_prev < 100:
                screens['cci34_daily_cross_100']['stocks'].append(_cci_entry_us(cci_d_now, cci_d_prev))
            if cci_d_now >= 100:
                _de = _cci_entry_us(cci_d_now, cci_d_prev)
                _d_streak = calc_cci_streak_start(df, period=34, weekly=False) or today_str
                _de['added_date'] = _d_streak
                _de['is_new'] = (_d_streak == today_str)
                screens['cci34_daily_100']['stocks'].append(_de)

        if cci_w_now is not None and cci_w_prev is not None:
            if cci_w_now >= 100 and cci_w_prev < 100:
                screens['cci34_weekly_cross_100']['stocks'].append(_cci_entry_us(cci_w_now, cci_w_prev))
            if cci_w_now >= 100:
                _we = _cci_entry_us(cci_w_now, cci_w_prev)
                _w_streak = calc_cci_streak_start(df, period=34, weekly=True) or today_str
                _we['added_date'] = _w_streak
                _we['is_new'] = (_w_streak == today_str)
                screens['cci34_weekly_100']['stocks'].append(_we)

    # ── CCI BEST SETUPS ───────────────────────────────────────────────────────
    _d100_map = {e['ticker']: e for e in screens['cci34_daily_100']['stocks']}
    _w100_map = {e['ticker']: e for e in screens['cci34_weekly_100']['stocks']}
    _full_set  = {e['ticker'] for e in screens['full_template']['stocks']}
    _near_set  = {e['ticker'] for e in screens['near_breakout']['stocks']}
    _minervini = {**{t: next(e for e in screens['full_template']['stocks'] if e['ticker']==t) for t in _full_set},
                  **{t: next(e for e in screens['near_breakout']['stocks'] if e['ticker']==t) for t in _near_set if t not in _full_set}}
    for ticker, mentry in _minervini.items():
        d_entry = _d100_map.get(ticker)
        w_entry = _w100_map.get(ticker)
        if d_entry is None and w_entry is None:
            continue
        screens['cci34_best_setups']['stocks'].append({
            **mentry,
            'cci_daily':  round(d_entry['cci_now'],1) if d_entry else None,
            'cci_weekly': round(w_entry['cci_now'],1) if w_entry else None,
            'cci_signals': ', '.join(filter(None, [
                'D-cross' if ticker in {e['ticker'] for e in screens['cci34_daily_cross_100']['stocks']} else None,
                'W-cross' if ticker in {e['ticker'] for e in screens['cci34_weekly_cross_100']['stocks']} else None,
                'D≥100' if d_entry else None,
                'W≥100' if w_entry else None,
            ])),
        })

    # ── SORT ──────────────────────────────────────────────────────────────────
    for scr in screens.values():
        scr['stocks'].sort(key=lambda x: x.get('rs_rank', 0), reverse=True)

    # ── QUARTERLY FINANCIALS ──────────────────────────────────────────────────
    _QFIN_SCREENS = {'full_template','near_breakout','rs_leaders','vcp_setup',
                     'new_highs','watch_list','cci34_daily_100','cci34_weekly_100',
                     'cci34_best_setups','htf_setup','htf_potential','ma_pullback',
                     'hhhl_pullback','primary_base_new','primary_base_10yr'}
    _qfin_syms = set()
    for key, scr in screens.items():
        if key in _QFIN_SCREENS:
            for e in scr['stocks']:
                _qfin_syms.add(e['ticker'])

    _qfin = fetch_quarterly_financials_us(list(_qfin_syms))
    for scr in screens.values():
        for e in scr['stocks']:
            qf = _qfin.get(e['ticker']) or {}
            for fld in QFIN_FIELDS:
                e[fld] = qf.get(fld)

    # ── WRITE OUTPUT ──────────────────────────────────────────────────────────
    result = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market':       'us',
        'scanned':      len(evaluations),
        'criteria_meta': CRITERIA_META,
        'screens':      screens,
    }

    def _sanitize(s):
        s = re.sub(r'\bNaN\b', 'null', s)
        s = re.sub(r'\bInfinity\b', 'null', s)
        s = re.sub(r'\b-Infinity\b', 'null', s)
        return s

    tmp_json = RESULTS_JSON + '.tmp'
    tmp_js   = RESULTS_JS   + '.tmp'
    with open(tmp_json, 'w') as f:
        f.write(_sanitize(json.dumps(result, indent=2)))
    os.replace(tmp_json, RESULTS_JSON)
    with open(tmp_js, 'w') as f:
        f.write('window.SCAN_DATA_US = ')
        f.write(_sanitize(json.dumps(result)))
        f.write(';')
    os.replace(tmp_js, RESULTS_JS)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"US Scan complete — {result['generated_at']}")
    print(f"Universe scanned: {len(evaluations)} stocks")
    print(f"{'='*55}")
    print(f"\n  COMPOSITE SCREENS:")
    for key in ['full_template','near_breakout','rs_leaders','vcp_setup','new_highs','watch_list']:
        print(f"    {screens[key]['label']:<45} {len(screens[key]['stocks'])} stocks")
    print(f"\n  CCI34:")
    for key in ['cci34_daily_100','cci34_weekly_100','cci34_daily_cross_100','cci34_weekly_cross_100','cci34_best_setups']:
        print(f"    {screens[key]['label']:<45} {len(screens[key]['stocks'])} stocks")
    print(f"\n  HTF & TREND:")
    for key in ['htf_setup','htf_potential','ma_pullback','hhhl_pullback']:
        print(f"    {screens[key]['label']:<45} {len(screens[key]['stocks'])} stocks")
    print(f"\n  Results saved: {RESULTS_JS}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--symbols', type=int, default=None)
    args = parser.parse_args()
    max_s = 100 if args.fast else args.symbols
    run_scan(max_syms=max_s)
