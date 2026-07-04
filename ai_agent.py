"""
AlphaEdge AI Agent — Groq-powered agentic analysis layer for NSE Minervini Scanner.

Answers are grounded in TWO sources:
  1. Live scan data from results.json (real-time NSE screener output)
  2. RAG over "Trade Like a Stock Market Wizard" by Mark Minervini (local ChromaDB)

No data is sent to external servers except scan context sent to Groq for inference.
"""
import json
import os
import re

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, 'results.json')
RAG_DB       = os.path.join(BASE_DIR, 'rag_db')

# ── RAG retriever (lazy-loaded on first use) ──────────────────────────────────
_rag_collection  = None
_rag_embed_model = None

def _get_rag():
    global _rag_collection, _rag_embed_model
    if _rag_collection is not None:
        return _rag_collection, _rag_embed_model
    if not os.path.exists(RAG_DB):
        return None, None
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.PersistentClient(path=RAG_DB)
        _rag_collection  = client.get_collection('minervini_books')
        _rag_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        return None, None
    return _rag_collection, _rag_embed_model


def _query_rag(query: str, n: int = 4) -> list[dict]:
    """Return top-n relevant passages from the Minervini book."""
    collection, model = _get_rag()
    if collection is None or model is None:
        return []
    try:
        embedding = model.encode([query]).tolist()
        results   = collection.query(
            query_embeddings=embedding,
            n_results=n,
            include=['documents', 'metadatas', 'distances'],
        )
        passages = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        ):
            if dist < 0.7:   # cosine distance < 0.7 = meaningful relevance
                passages.append({
                    'text':  doc,
                    'book':  meta.get('book', ''),
                    'page':  meta.get('page', ''),
                    'score': round(1 - dist, 3),
                })
        return passages
    except Exception:
        return []

# ── Minervini-style VCP validator ─────────────────────────────────────────────
def _validate_vcp_minervini(stocks: list, screens: dict) -> list:
    """
    Apply Minervini's actual criteria to filter and score VCP candidates.

    A true Minervini VCP requires (from the book):
      ✓ Stage 2 uptrend — C1–C6 all confirmed (structural prerequisite)
      ✓ RS rank ≥ 80 — must be a relative strength leader
      ✓ Tight base — within 12% of 52-week high (base forming near highs)
      ✓ Volume contraction — vol_ratio < 1.0 ideally (volume drying up in base)
      ✓ Momentum — CCI daily ≥ 100 preferred (institutional accumulation)
      ✓ VCP flag from scanner — algorithmic contraction detection
      ✓ Near key MAs — price close to EMA21 or MA50 (not extended)

    Returns validated candidates with a quality score and reasoning.
    """
    ck_daily  = {_ticker(s) for s in screens.get('ck_daily_100',  {}).get('stocks', [])}
    ck_weekly = {_ticker(s) for s in screens.get('ck_weekly_100', {}).get('stocks', [])}
    full_tmpl = {_ticker(s) for s in screens.get('full_template', {}).get('stocks', [])}
    c1_to_c6  = {_ticker(s) for s in screens.get('c1_to_c6',     {}).get('stocks', [])}

    validated = []
    for s in stocks:
        t   = _ticker(s)
        cr  = s.get('criteria') or {}
        rs  = s.get('rs_rank') or 0
        pct_hi   = s.get('pct_from_high') or -999
        vol      = s.get('vol_ratio') or 1.0
        pct_ma50 = s.get('pct_from_ma50') or 999
        pct_e21  = s.get('pct_from_ema21') or 999
        vcp_d    = bool(s.get('vcp_last_daily'))
        vcp_w    = bool(s.get('vcp_last_weekly'))

        passes  = []
        fails   = []
        score   = 0

        # ── C1–C6: Stage 2 structural base (MANDATORY) ──────────────────────
        c1_c6_ok = all(cr.get(f'c{i}') for i in range(1, 7))
        if c1_c6_ok:
            passes.append('C1–C6 ✓ Stage 2 base confirmed')
            score += 30
        else:
            failed_cs = [f'C{i}' for i in range(1,7) if not cr.get(f'c{i}')]
            fails.append(f'Stage 2 incomplete — {", ".join(failed_cs)} not met (DISQUALIFIED)')
            # Hard disqualify — Minervini never buys outside Stage 2
            validated.append({'ticker': t, 'score': 0, 'grade': 'FAIL',
                              'passes': passes, 'fails': fails, 'data': s})
            continue

        # ── RS Rank ≥ 80 ────────────────────────────────────────────────────
        if rs >= 90:
            passes.append(f'RS {rs} — top-tier leader (≥90)')
            score += 20
        elif rs >= 80:
            passes.append(f'RS {rs} — strong leader (≥80)')
            score += 12
        elif rs >= 70:
            passes.append(f'RS {rs} — acceptable but not ideal (<80)')
            fails.append('RS below 80 — Minervini prefers 80+ for VCP entries')
            score += 5
        else:
            fails.append(f'RS {rs} — too weak; Minervini avoids RS < 70')
            score -= 10

        # ── Tightness of base (% from 52-week high) ─────────────────────────
        if pct_hi >= -5:
            passes.append(f'Tight base: {pct_hi:.1f}% from 52w high — breakout zone')
            score += 20
        elif pct_hi >= -10:
            passes.append(f'Base: {pct_hi:.1f}% from 52w high — acceptable')
            score += 10
        elif pct_hi >= -15:
            fails.append(f'Base loose: {pct_hi:.1f}% from high — getting extended')
            score += 3
        else:
            fails.append(f'Base too wide: {pct_hi:.1f}% from high — not a VCP (>15% is a correction, not contraction)')
            score -= 15

        # ── Volume contraction (drying up in base) ───────────────────────────
        if vol < 0.7:
            passes.append(f'Volume drying up: vol ratio {vol:.2f} — institutional sellers absent')
            score += 15
        elif vol < 1.0:
            passes.append(f'Volume subdued: vol ratio {vol:.2f} — base forming quietly')
            score += 8
        elif vol < 1.5:
            fails.append(f'Volume elevated: {vol:.2f}× avg — should be drying up in a VCP base')
        else:
            fails.append(f'Volume too high: {vol:.2f}× avg — distribution possible, not accumulation')
            score -= 10

        # ── CCI momentum (institutional footprint) ───────────────────────────
        if t in ck_daily and t in ck_weekly:
            passes.append('CCI daily + weekly ≥100 — multi-TF momentum confirmation')
            score += 15
        elif t in ck_daily:
            passes.append('CCI daily ≥100 — daily momentum present')
            score += 8
        elif t in ck_weekly:
            passes.append('CCI weekly ≥100 — weekly trend strong')
            score += 5
        else:
            fails.append('No CCI signal — momentum not confirmed on either timeframe')

        # ── VCP scanner flag ─────────────────────────────────────────────────
        if vcp_d and vcp_w:
            passes.append('VCP detected daily + weekly — multi-TF contraction pattern')
            score += 10
        elif vcp_d:
            passes.append('VCP detected on daily — contraction pattern on primary timeframe')
            score += 7
        elif vcp_w:
            passes.append('VCP detected on weekly — longer-term base')
            score += 4
        else:
            fails.append('Scanner did not flag VCP — may be false from raw screen')

        # ── Proximity to moving averages (not overextended) ──────────────────
        if abs(pct_e21) <= 5:
            passes.append(f'Price near EMA21 ({pct_e21:+.1f}%) — tight to trend')
            score += 5
        elif abs(pct_ma50) <= 8:
            passes.append(f'Price near MA50 ({pct_ma50:+.1f}%) — pulling back to support')
            score += 3
        elif pct_e21 > 15:
            fails.append(f'Extended {pct_e21:.1f}% above EMA21 — too extended for safe VCP entry')
            score -= 8

        # ── Grade ────────────────────────────────────────────────────────────
        if score >= 80:
            grade = 'A — Minervini-quality VCP'
        elif score >= 60:
            grade = 'B — Good setup, verify chart'
        elif score >= 40:
            grade = 'C — Marginal, needs confirmation'
        else:
            grade = 'D — Does not meet VCP criteria'

        validated.append({
            'ticker': t, 'score': score, 'grade': grade,
            'passes': passes, 'fails': fails,
            'rs': rs, 'pct_hi': pct_hi, 'vol': vol,
            'vcp_daily': vcp_d, 'vcp_weekly': vcp_w,
            'cci_daily': t in ck_daily, 'cci_weekly': t in ck_weekly,
            'data': s,
        })

    validated.sort(key=lambda x: -x['score'])
    return validated


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
    if any(w in q for w in ['fo', 'f&o', 'fno', 'best', 'setup', 'trade', 'buy', 'pick', 'vcp', 'breakout', 'entry']):
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

    # ── VCP setups — scan ALL screens, validate like Minervini ───────────────
    if any(w in q for w in ['vcp', 'volatility contraction', 'breakout', 'pattern',
                             'setup', 'best', 'pick', 'trade', 'buy', 'entry']):
        tools.append('get_screen')

        # Collect every unique stock across ALL screens — VCPs can appear anywhere
        _ALL_SCREENS = [
            'full_template', 'near_breakout', 'vcp_setup', 'c1_to_c6',
            'watch_list', 'fo_strong_uptrend', 'ma_pullback', 'hhhl_pullback',
            'ck_daily_100', 'ck_weekly_100', 'ck_mtf_100', 'cci34_best_setups',
            'rs_leaders', 'strong_earnings', 'new_highs', 'newly_added',
        ]
        seen, all_stocks = set(), []
        stock_screens = {}   # ticker → list of screens it appears in
        for scr_key in _ALL_SCREENS:
            scr_label = SCREEN_LABELS.get(scr_key, scr_key)
            for s in screens.get(scr_key, {}).get('stocks', []):
                t = _ticker(s)
                if not t:
                    continue
                stock_screens.setdefault(t, []).append(scr_label)
                if t not in seen:
                    seen.add(t)
                    all_stocks.append(s)

        # Run every stock through Minervini's VCP validator
        validated = _validate_vcp_minervini(all_stocks, screens)

        # Attach which screens each stock appears in
        for v in validated:
            v['appears_in'] = stock_screens.get(v['ticker'], [])

        grade_a = [v for v in validated if v['score'] >= 80]
        grade_b = [v for v in validated if 60 <= v['score'] < 80]
        disq    = [v for v in validated if 'DISQUALIFIED' in ' '.join(v.get('fails', []))]

        rows = []
        rows.append(f"  Scanned {len(all_stocks)} unique stocks across {len(_ALL_SCREENS)} screens.")
        rows.append(f"  Result: {len(grade_a)} Grade-A | {len(grade_b)} Grade-B | {len(disq)} disqualified (Stage 2 failed)\n")

        for v in grade_a[:10]:
            scrs = ', '.join(dict.fromkeys(v['appears_in']))   # deduplicated screen list
            rows.append(
                f"  ★ {v['ticker']:15} RS:{v['rs']:>3}  "
                f"Base:{v['pct_hi']:+.1f}%  Vol:{v['vol']:.2f}x  "
                f"CCId:{'✓' if v['cci_daily'] else '✗'} CCIw:{'✓' if v['cci_weekly'] else '✗'}  "
                f"VCP:{'D' if v['vcp_daily'] else ''}{'W' if v['vcp_weekly'] else ''}  "
                f"Score:{v['score']}  [{v['grade']}]"
            )
            rows.append(f"     Found in: {scrs}")
            rows.append(f"     PASSES: {' | '.join(v['passes'])}")
            if v['fails']:
                rows.append(f"     NOTES:  {' | '.join(v['fails'])}")

        for v in grade_b[:5]:
            scrs = ', '.join(dict.fromkeys(v['appears_in']))
            rows.append(
                f"  ◐ {v['ticker']:15} RS:{v['rs']:>3}  "
                f"Base:{v['pct_hi']:+.1f}%  Score:{v['score']}  [{v['grade']}]  "
                f"Found in: {scrs}"
            )
            if v['fails']:
                rows.append(f"     CAUTION: {' | '.join(v['fails'])}")

        parts.append("MINERVINI VCP VALIDATION (full dashboard scan):\n" + '\n'.join(rows))

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
SYSTEM_PROMPT = """You are AlphaEdge AI — you think and analyse exactly like Mark Minervini himself.

You have TWO sources to work from:
  1. MINERVINI BOOK EXCERPTS — exact passages from "Trade Like a Stock Market Wizard"
  2. MINERVINI VCP VALIDATION — pre-scored candidates using Minervini's actual criteria

Your job is NOT to list stocks from a screen. Your job is to REASON like Minervini:

CRITICAL RULES:
1. Only recommend Grade-A stocks (score ≥ 80) for actual trades. Be strict — Minervini himself says he waits for the best setups.
2. Explain WHY each stock qualifies using specific Minervini concepts from the book (cite page numbers).
3. Call out what's MISSING for lower-grade stocks — e.g. "Volume has not dried up enough per page 218's criteria."
4. NEVER recommend a stock that failed Stage 2 (C1–C6 not met) — Minervini never buys outside Stage 2.
5. Be honest about limitations: "I can see quantitative data but cannot see the actual price chart — you must visually verify the contraction symmetry before entering."
6. Always state the current regime and what it means for position sizing.
7. If no Grade-A stocks exist today, say so clearly — "There are no stocks meeting Minervini's full VCP criteria today. Wait for better setups."

MINERVINI'S VCP CHECKLIST (from the book):
- Stage 2 uptrend: price above rising MA200, MA200 > MA150 > MA50 (C1–C6 all met)
- RS leader: RS rank ≥ 80 — only buy the strongest stocks in the market
- Tight base: price within 10–12% of 52-week high — base forming near highs, not in a hole
- Volume contraction: volume must DRY UP in the base — fewer sellers each contraction
- Contraction symmetry: each swing smaller than the last (2–4 contractions typical)
- Pivot point: clear breakout level with volume expansion expected on breakout
- CCI momentum: institutional accumulation footprint visible in CCI34 ≥ 100"""


# ── AGENTIC LOOP (RAG + context-stuffed) ─────────────────────────────────────
def run_agent(user_message: str, api_key: str) -> tuple:
    """
    Answer using Minervini RAG + live scan context.
    Returns (answer: str, tools_used: list[str]).
    """
    from groq import Groq
    client = Groq(api_key=api_key)

    # 1. Retrieve relevant Minervini book passages
    passages   = _query_rag(user_message, n=4)
    tools_used = []

    rag_section = ''
    if passages:
        tools_used.append('query_minervini_book')
        lines = []
        for p in passages:
            lines.append(
                f'[Page {p["page"]} | relevance {p["score"]}]\n"{p["text"]}"'
            )
        rag_section = '--- MINERVINI BOOK EXCERPTS ---\n' + '\n\n'.join(lines)
    else:
        rag_section = '--- MINERVINI BOOK EXCERPTS ---\n(RAG index not found — run build_rag.py)'

    # 2. Build live scan context
    scan_context, scan_tools = _build_context(user_message)
    tools_used.extend(scan_tools)

    # 3. Compose prompt: book excerpts first, then live data
    full_context = rag_section + '\n\n' + '--- LIVE SCAN DATA ---\n' + scan_context

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + '\n\n' + full_context},
        {'role': 'user',   'content': user_message},
    ]

    resp = client.chat.completions.create(
        model       = 'llama-3.3-70b-versatile',
        messages    = messages,
        max_tokens  = 1024,
        temperature = 0.2,
    )

    answer = (resp.choices[0].message.content or '').strip()
    return answer, list(dict.fromkeys(tools_used))
