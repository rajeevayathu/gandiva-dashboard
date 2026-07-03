"""
AlphaEdge AI Agent — Groq-powered agentic analysis layer for NSE Minervini Scanner.

The agent has 6 tools that read directly from results.json.
It runs an agentic loop: LLM → decide which tools to call → execute tools →
feed results back → LLM → more tools if needed → final answer.

No data is sent to external servers except the scan summary/stock data
sent to Groq for LLM inference.
"""
import json
import os
import re

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, 'results.json')

# ── Screen label map ──────────────────────────────────────────────────────────
SCREEN_LABELS = {
    'full_template':        'Full Template (all 8 criteria)',
    'near_breakout':        'Near Breakout (within 5% of high)',
    'vcp_setup':            'VCP Setup',
    'rs_leaders':           'RS Leaders (RS ≥ 85)',
    'watch_list':           'Watch List (7/8 criteria)',
    'c1_to_c6':             'C1–C6 Stage 2 Uptrend',
    'new_highs':            'New 52-Week Highs',
    'fo_momentum':          'F&O Stocks in Dashboard',
    'fo_strong_uptrend':    'F&O Strong Uptrend (C1–C6)',
    'ma_pullback':          'MA Pullback',
    'hhhl_pullback':        'HH/HL Pullback',
    'ck_daily_100':         'CCI Daily ≥ 100',
    'ck_weekly_100':        'CCI Weekly ≥ 100',
    'ck_mtf_100':           'CCI MTF (Daily + Weekly) ≥ 100',
    'cci34_best_setups':    'CCI Best Setups (Full Template + CCI)',
    'newly_added':          'New This Scan',
    'strong_earnings':      'Strong Earnings + CCI',
    'ipo_watch':            'IPO Watch',
    'minervini_backtest':   'Minervini Backtest Exact Picks',
    'primary_base_new':     'Primary Base — Recent IPO (≤3 yrs)',
    'primary_base_10yr':    'Primary Base — Young Company (≤10 yrs)',
}

# ── Data loader ───────────────────────────────────────────────────────────────
def _load_results():
    if not os.path.exists(RESULTS_JSON):
        return {}
    with open(RESULTS_JSON) as f:
        raw = f.read()
    raw = re.sub(r'\bNaN\b', 'null', raw)
    raw = re.sub(r'\bInfinity\b', 'null', raw)
    return json.loads(raw)

def _ticker(s):
    return (s.get('ticker') or '').replace('.NS', '').upper()

def _stock_summary(s):
    return {
        'ticker':           _ticker(s),
        'price':            s.get('price'),
        'rs_rank':          s.get('rs_rank'),
        'pct_from_high':    s.get('pct_from_high'),
        'vol_ratio':        s.get('vol_ratio'),
        'criteria_passed':  s.get('passed'),
        'vcp_daily':        s.get('vcp_last_daily'),
        'vcp_weekly':       s.get('vcp_last_weekly'),
    }

# ── TOOL FUNCTIONS ────────────────────────────────────────────────────────────

def get_scan_summary() -> dict:
    """High-level scan overview: counts per screen, regime, scan time."""
    data = _load_results()
    if not data:
        return {'error': 'No scan data. Run scanner.py first.'}
    screens = data.get('screens', {})
    regime  = (data.get('market_analysis') or {}).get('regime', 'Unknown')
    return {
        'generated_at':  data.get('generated_at'),
        'total_scanned': data.get('scanned'),
        'regime':        regime,
        'screen_counts': {k: len(v.get('stocks', [])) for k, v in screens.items()},
    }


def get_market_regime() -> dict:
    """Current market regime (Bull/Caution/Bear) with breadth context."""
    data = _load_results()
    ma   = data.get('market_analysis') or {}
    scr  = data.get('screens') or {}
    return {
        'regime':              ma.get('regime', 'Unknown'),
        'generated_at':        data.get('generated_at'),
        'full_template_count': len(scr.get('full_template', {}).get('stocks', [])),
        'near_breakout_count': len(scr.get('near_breakout', {}).get('stocks', [])),
        'vcp_count':           len(scr.get('vcp_setup', {}).get('stocks', [])),
        'new_this_scan':       len(scr.get('newly_added', {}).get('stocks', [])),
        'summary':             ma.get('summary', ''),
    }


def get_screen(screen_name: str) -> dict:
    """Get stocks in a specific screen (top 20 by RS rank)."""
    data    = _load_results()
    screens = data.get('screens', {})
    if screen_name not in screens:
        return {'error': f"Screen '{screen_name}' not found. Available: {list(SCREEN_LABELS.keys())}"}
    stocks = screens[screen_name].get('stocks', [])
    return {
        'screen':      screen_name,
        'label':       SCREEN_LABELS.get(screen_name, screen_name),
        'total_count': len(stocks),
        'stocks':      [_stock_summary(s) for s in stocks[:20]],
    }


def get_stock_details(ticker: str) -> dict:
    """Full analysis for one stock: all 8 criteria, RS, VCP, earnings, screens it appears in."""
    ticker  = ticker.upper().replace('.NS', '')
    data    = _load_results()
    screens = data.get('screens', {})

    best    = None
    appears = []
    priority = ['full_template', 'near_breakout', 'vcp_setup', 'c1_to_c6', 'watch_list',
                'fo_strong_uptrend', 'ma_pullback', 'ck_mtf_100', 'cci34_best_setups']

    for key, scr in screens.items():
        for s in scr.get('stocks', []):
            if _ticker(s) == ticker:
                appears.append(SCREEN_LABELS.get(key, key))
                if best is None or key in priority:
                    best = s

    if not best:
        return {'error': f'{ticker} not found in any screen.'}

    cr = best.get('criteria') or {}
    return {
        'ticker':           ticker,
        'price':            best.get('price'),
        'rs_rank':          best.get('rs_rank'),
        'pct_from_high':    best.get('pct_from_high'),
        'vol_ratio':        best.get('vol_ratio'),
        'pct_from_ma50':    best.get('pct_from_ma50'),
        'pct_from_ema21':   best.get('pct_from_ema21'),
        'pct_from_ema10':   best.get('pct_from_ema10'),
        'criteria_passed':  best.get('passed'),
        'criteria':         {f'c{i}': cr.get(f'c{i}') for i in range(1, 9)},
        'c1_to_c6':         all(cr.get(f'c{i}') for i in range(1, 7)),
        'vcp_last_daily':   best.get('vcp_last_daily'),
        'vcp_last_weekly':  best.get('vcp_last_weekly'),
        'cci_signals':      best.get('cci_signals') or best.get('ck_signals'),
        'q_label':          best.get('q_label'),
        'q_eps_yoy':        best.get('q_eps_yoy'),
        'q_rev_yoy':        best.get('q_rev_yoy'),
        'q_np_yoy':         best.get('q_np_yoy'),
        'fo_screens':       best.get('fo_screens'),
        'appears_in':       list(set(appears)),
    }


def filter_stocks(screen_name: str, min_rs: int = 0,
                  require_cci_daily: bool = False,
                  require_cci_weekly: bool = False,
                  require_c1_to_c6: bool = False,
                  require_vcp: bool = False,
                  max_results: int = 10) -> dict:
    """Filter stocks in a screen by RS rank, CCI momentum, C1–C6, or VCP pattern."""
    data    = _load_results()
    screens = data.get('screens', {})
    if screen_name not in screens:
        return {'error': f"Screen '{screen_name}' not found."}

    ck_daily  = {_ticker(s) for s in screens.get('ck_daily_100',  {}).get('stocks', [])}
    ck_weekly = {_ticker(s) for s in screens.get('ck_weekly_100', {}).get('stocks', [])}

    filtered = []
    for s in screens[screen_name].get('stocks', []):
        t  = _ticker(s)
        rs = s.get('rs_rank') or 0
        cr = s.get('criteria') or {}
        if rs < min_rs:                                                   continue
        if require_cci_daily  and t not in ck_daily:                     continue
        if require_cci_weekly and t not in ck_weekly:                    continue
        if require_c1_to_c6   and not all(cr.get(f'c{i}') for i in range(1, 7)): continue
        if require_vcp        and not s.get('vcp_last_daily') and not s.get('vcp_last_weekly'): continue
        entry = _stock_summary(s)
        entry['cci_daily']  = t in ck_daily
        entry['cci_weekly'] = t in ck_weekly
        filtered.append(entry)

    return {
        'screen':               screen_name,
        'total_before_filter':  len(screens[screen_name].get('stocks', [])),
        'total_after_filter':   len(filtered),
        'filters': {
            'min_rs': min_rs, 'require_cci_daily': require_cci_daily,
            'require_cci_weekly': require_cci_weekly,
            'require_c1_to_c6': require_c1_to_c6, 'require_vcp': require_vcp,
        },
        'stocks': filtered[:max_results],
    }


def get_new_this_scan() -> dict:
    """Stocks newly added to key screens since the previous scan."""
    data   = _load_results()
    newly  = (data.get('screens') or {}).get('newly_added', {}).get('stocks', [])
    by_scr = {}
    for s in newly:
        for scr in (s.get('new_in_screens') or []):
            by_scr.setdefault(scr, []).append({
                'ticker':  _ticker(s),
                'rs_rank': s.get('rs_rank'),
                'price':   s.get('price'),
            })
    return {
        'total_new':    len(newly),
        'generated_at': data.get('generated_at'),
        'by_screen':    by_scr,
        'top_by_rs':    [
            {'ticker': _ticker(s), 'rs_rank': s.get('rs_rank'), 'new_in': s.get('new_in_screens')}
            for s in sorted(newly, key=lambda x: -(x.get('rs_rank') or 0))[:20]
        ],
    }


def compare_screens(screen1: str, screen2: str) -> dict:
    """Find stocks appearing in BOTH screens — confluence setups."""
    data    = _load_results()
    screens = data.get('screens', {})
    if screen1 not in screens: return {'error': f"Screen '{screen1}' not found."}
    if screen2 not in screens: return {'error': f"Screen '{screen2}' not found."}

    map1 = {_ticker(s): s for s in screens[screen1].get('stocks', [])}
    map2 = {_ticker(s): s for s in screens[screen2].get('stocks', [])}
    common = sorted(set(map1) & set(map2), key=lambda t: -(map1[t].get('rs_rank') or 0))

    return {
        'screen1':            SCREEN_LABELS.get(screen1, screen1),
        'screen2':            SCREEN_LABELS.get(screen2, screen2),
        'screen1_total':      len(map1),
        'screen2_total':      len(map2),
        'intersection_count': len(common),
        'stocks': [_stock_summary(map1[t]) for t in common[:20]],
    }


# ── GROQ TOOL DEFINITIONS ────────────────────────────────────────────────────
_SCREEN_ENUM = list(SCREEN_LABELS.keys())

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_scan_summary',
            'description': 'Get a high-level overview of the latest scan — total stocks scanned, count in each screen, market regime, and scan timestamp. Call this first to orient yourself.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_market_regime',
            'description': 'Get current market regime (Bull, Caution, or Bear) and market breadth summary. Use this to set context for any trade recommendations.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_screen',
            'description': 'Get the stocks in a specific screen (top 20 by RS rank) with key metrics. Use this to see what stocks are in any screen.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'screen_name': {
                        'type': 'string',
                        'description': (
                            'Screen to retrieve. Valid values: full_template, near_breakout, vcp_setup, '
                            'rs_leaders, watch_list, c1_to_c6, new_highs, fo_momentum, fo_strong_uptrend, '
                            'ma_pullback, hhhl_pullback, ck_daily_100, ck_weekly_100, ck_mtf_100, '
                            'cci34_best_setups, newly_added, strong_earnings, ipo_watch'
                        ),
                    }
                },
                'required': ['screen_name'],
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_stock_details',
            'description': 'Get full analysis for a specific NSE stock: all 8 Minervini criteria, RS rank, VCP pattern dates, earnings (EPS/Rev YoY), MA proximity, and which screens it appears in.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'NSE ticker (e.g. HFCL, RELIANCE, BHARATSEATS). Do not include .NS suffix.',
                    }
                },
                'required': ['ticker'],
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'filter_stocks',
            'description': 'Filter stocks in a screen by criteria: minimum RS rank, CCI daily/weekly momentum, all C1–C6 confirmed, or VCP pattern present. Use this to find the highest-quality setups.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'screen_name':        {'type': 'string', 'description': 'Screen to filter (e.g. full_template, fo_strong_uptrend, c1_to_c6, ck_mtf_100, ma_pullback, ck_daily_100, ck_weekly_100)'},
                    'min_rs':             {'type': 'integer', 'description': 'Minimum RS rank (0–100)'},
                    'require_cci_daily':  {'type': 'boolean', 'description': 'Only stocks with Daily CCI34 ≥ 100'},
                    'require_cci_weekly': {'type': 'boolean', 'description': 'Only stocks with Weekly CCI34 ≥ 100'},
                    'require_c1_to_c6':   {'type': 'boolean', 'description': 'Only stocks with all C1–C6 Minervini structural criteria confirmed'},
                    'require_vcp':        {'type': 'boolean', 'description': 'Only stocks with a recent VCP pattern on daily or weekly chart'},
                    'max_results':        {'type': 'integer', 'description': 'Max results to return (default 10)'},
                },
                'required': ['screen_name'],
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_new_this_scan',
            'description': 'Get stocks newly added to key screens (MA Pullback, CCI Daily, CCI Weekly, C1–C6, Full Template) since the previous scan run. Use this to find fresh setups.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'compare_screens',
            'description': 'Find stocks appearing in BOTH of two screens — high-confluence setups. E.g. full_template + ck_weekly_100 finds stocks with perfect trend AND CCI momentum.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'screen1': {'type': 'string', 'description': 'First screen (e.g. full_template, ck_weekly_100)'},
                    'screen2': {'type': 'string', 'description': 'Second screen (e.g. fo_strong_uptrend, ck_daily_100)'},
                },
                'required': ['screen1', 'screen2'],
            }
        }
    },
]

# ── CONTEXT BUILDER ───────────────────────────────────────────────────────────
def _build_context(user_message: str) -> tuple[str, list[str]]:
    """
    Build a rich, focused context from live scan data.
    Returns (context_text, tools_used_labels).
    Selects data relevant to the user's question to keep prompt tight.
    """
    q = user_message.lower()
    data    = _load_results()
    screens = data.get('screens', {}) if data else {}
    ma      = data.get('market_analysis') or {}
    tools   = []
    parts   = []

    if not data:
        return 'No scan data available. Run scanner.py first.', []

    # ── Always include: scan meta + regime ───────────────────────────────────
    tools.append('get_scan_summary')
    parts.append(f"SCAN DATE: {data.get('generated_at', 'unknown')}")
    parts.append(f"TOTAL SCANNED: {data.get('scanned', 0)} stocks")
    regime = ma.get('regime', 'Unknown')
    parts.append(f"MARKET REGIME: {regime}")
    if ma.get('summary'):
        parts.append(f"REGIME NOTES: {ma.get('summary')}")

    # ── Screen counts ─────────────────────────────────────────────────────────
    tools.append('get_market_regime')
    key_screens = [
        ('full_template', 'Full Template (C1–C8)'),
        ('c1_to_c6',      'C1–C6 Structural Base'),
        ('near_breakout', 'Near Breakout (≤5% from high)'),
        ('vcp_setup',     'VCP Setup'),
        ('fo_strong_uptrend', 'F&O Strong Uptrend'),
        ('ck_daily_100',  'CCI Daily ≥100'),
        ('ck_weekly_100', 'CCI Weekly ≥100'),
        ('ck_mtf_100',    'CCI Multi-TF (Daily+Weekly)'),
        ('ma_pullback',   'MA Pullback'),
        ('newly_added',   'New This Scan'),
    ]
    counts = []
    for k, lbl in key_screens:
        n = len(screens.get(k, {}).get('stocks', []))
        counts.append(f"  {lbl}: {n}")
    parts.append("SCREEN COUNTS:\n" + '\n'.join(counts))

    # ── New This Scan ─────────────────────────────────────────────────────────
    if any(w in q for w in ['new', 'fresh', 'yesterday', 'added', 'since']):
        tools.append('get_new_this_scan')
        newly = screens.get('newly_added', {}).get('stocks', [])[:20]
        if newly:
            rows = []
            for s in newly:
                scrs = ', '.join(s.get('new_in_screens') or [])
                rows.append(f"  {_ticker(s):15} RS:{s.get('rs_rank','-'):>3}  new in: {scrs}")
            parts.append("NEW THIS SCAN (since previous scan):\n" + '\n'.join(rows))
        else:
            parts.append("NEW THIS SCAN: none (or no previous baseline)")

    # ── F&O / best setups ─────────────────────────────────────────────────────
    if any(w in q for w in ['fo', 'f&o', 'fno', 'best', 'setup', 'trade', 'buy', 'pick']):
        tools.append('filter_stocks')
        fo = screens.get('fo_strong_uptrend', {}).get('stocks', [])[:15]
        ck_daily  = {_ticker(s) for s in screens.get('ck_daily_100',  {}).get('stocks', [])}
        ck_weekly = {_ticker(s) for s in screens.get('ck_weekly_100', {}).get('stocks', [])}
        vcp_ft = screens.get('vcp_setup', {}).get('stocks', [])[:15]
        ft     = screens.get('full_template', {}).get('stocks', [])[:20]

        rows = []
        for s in fo:
            t  = _ticker(s)
            cd = '✓' if t in ck_daily  else '✗'
            cw = '✓' if t in ck_weekly else '✗'
            vd = '✓' if s.get('vcp_last_daily')  else '✗'
            rows.append(
                f"  {t:15} RS:{s.get('rs_rank','-'):>3}  "
                f"CCId:{cd} CCIw:{cw} VCP:{vd}  "
                f"price:{s.get('price','-')}"
            )
        parts.append("F&O STRONG UPTREND (top 15):\n" + ('\n'.join(rows) if rows else '  (none)'))

        ft_rows = []
        for s in ft[:10]:
            t  = _ticker(s)
            cd = '✓' if t in ck_daily  else '✗'
            cw = '✓' if t in ck_weekly else '✗'
            vd = '✓' if s.get('vcp_last_daily') else ('W' if s.get('vcp_last_weekly') else '✗')
            ft_rows.append(
                f"  {t:15} RS:{s.get('rs_rank','-'):>3}  "
                f"CCId:{cd} CCIw:{cw} VCP:{vd}  "
                f"%hi:{s.get('pct_from_high','-')}"
            )
        parts.append("FULL TEMPLATE TOP 10:\n" + ('\n'.join(ft_rows) if ft_rows else '  (none)'))

    # ── Specific ticker lookup ─────────────────────────────────────────────────
    import re as _re
    mentioned_tickers = _re.findall(r'\b([A-Z]{3,}[A-Z0-9]*)\b', user_message.upper())
    for tk in mentioned_tickers:
        if tk in ('THE', 'AND', 'FOR', 'WHAT', 'HOW', 'CCI', 'RSI', 'NSE', 'BSE',
                  'VCP', 'MTF', 'EMA', 'SMA', 'ATR', 'IPO', 'FNO'):
            continue
        tools.append('get_stock_details')
        detail = get_stock_details(tk)
        if 'error' not in detail:
            cr = detail.get('criteria', {})
            c_str = ' '.join(f"C{i}:{'✓' if cr.get(f'c{i}') else '✗'}" for i in range(1,9))
            parts.append(
                f"STOCK DETAIL — {tk}:\n"
                f"  Price: {detail.get('price')}  RS: {detail.get('rs_rank')}\n"
                f"  %from52w high: {detail.get('pct_from_high')}\n"
                f"  Criteria: {c_str}\n"
                f"  VCP daily: {detail.get('vcp_last_daily') or 'none'}  weekly: {detail.get('vcp_last_weekly') or 'none'}\n"
                f"  CCI signals: {detail.get('cci_signals') or 'none'}\n"
                f"  Appears in: {', '.join(detail.get('appears_in', []))}\n"
                f"  Earnings EPS yoy: {detail.get('q_eps_yoy')}  Rev yoy: {detail.get('q_rev_yoy')}"
            )

    # ── CCI momentum / compare ─────────────────────────────────────────────────
    if any(w in q for w in ['cci', 'momentum', 'mtf', 'multi']):
        tools.append('compare_screens')
        ck_mtf = screens.get('ck_mtf_100', {}).get('stocks', [])[:15]
        rows = [
            f"  {_ticker(s):15} RS:{s.get('rs_rank','-'):>3}  price:{s.get('price','-')}"
            for s in ck_mtf
        ]
        parts.append("CCI MULTI-TF (Daily+Weekly ≥100) TOP 15:\n" + ('\n'.join(rows) if rows else '  (none)'))

    # ── MA pullback ────────────────────────────────────────────────────────────
    if any(w in q for w in ['pullback', 'ma ', 'moving average', 'ma50', 'ema']):
        tools.append('get_screen')
        mp = screens.get('ma_pullback', {}).get('stocks', [])[:15]
        ck_daily  = {_ticker(s) for s in screens.get('ck_daily_100', {}).get('stocks', [])}
        rows = [
            f"  {_ticker(s):15} RS:{s.get('rs_rank','-'):>3}  CCId:{'✓' if _ticker(s) in ck_daily else '✗'}  price:{s.get('price','-')}"
            for s in mp
        ]
        parts.append("MA PULLBACK TOP 15:\n" + ('\n'.join(rows) if rows else '  (none)'))

    return '\n\n'.join(parts), list(dict.fromkeys(tools))  # deduplicated tools list


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AlphaEdge AI, an expert NSE India equity analyst embedded in the AlphaEdge Minervini Scanner dashboard.

Live scan data is provided below. Use it to give precise, data-backed answers.

Domain knowledge:
- Minervini SEPA: C1–C8 = 8 structural criteria. C1–C6 = Stage 2 base. Full Template = all 8 met.
- VCP (Volatility Contraction Pattern) = ideal entry trigger for swing trades. Daily VCP > Weekly VCP.
- CCI34 ≥ 100 daily + weekly (MTF) = strongest momentum confirmation signal.
- RS Rank = relative strength vs NSE universe. 85+ = leader, 70+ = solid, below 60 = avoid.
- F&O stocks: can trade futures/options. Prefer Full Template + CCI momentum + low %from-high.
- Regime: BULL → full risk; CAUTION → half size; BEAR → cash.
- %from high: closer to 0% = near breakout zone (best entry). Negative means below 52w high.

Response rules:
1. Always cite specific tickers from the data — never make up names.
2. For "best setups": rank by VCP ✓ > CCI daily+weekly ✓ > RS rank high > small %from high.
3. Keep answers concise: bullet list for stocks, 2–3 sentences max for explanations.
4. Mention current regime when recommending trades.
5. If a ticker isn't in the data, say so — never guess."""


# ── AGENTIC LOOP (context-stuffed, model-agnostic) ───────────────────────────
def run_agent(user_message: str, api_key: str) -> tuple:
    """
    Answer using pre-built scan context. Returns (answer: str, tools_used: list[str]).
    """
    from groq import Groq
    client = Groq(api_key=api_key)

    context, tools_used = _build_context(user_message)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + '\n\n--- LIVE SCAN DATA ---\n' + context},
        {'role': 'user',   'content': user_message},
    ]

    resp = client.chat.completions.create(
        model       = 'llama-3.3-70b-versatile',
        messages    = messages,
        max_tokens  = 1024,
        temperature = 0.2,
    )

    answer = (resp.choices[0].message.content or '').strip()
    return answer, tools_used
