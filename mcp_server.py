"""
Gandiva MCP Server — Model Context Protocol interface for the NSE Scanner.

Exposes scanner tools to any MCP-compatible AI client:
  - Claude Desktop
  - Cursor IDE
  - Any future MCP client

Tools exposed:
  1. get_market_regime      — current regime + breadth
  2. get_screen             — stocks from any of the 16 screens
  3. get_top_setups         — Grade-A VCP candidates across all screens
  4. validate_stock         — Minervini VCP score for a specific ticker
  5. query_minervini_book   — semantic search over "Trade Like a Stock Market Wizard"
  6. get_scan_summary       — full overview: counts per screen, regime, last scan time

Usage:
  python3 mcp_server.py          # runs as stdio MCP server (for Claude Desktop)

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "gandiva": {
        "command": "/opt/homebrew/bin/python3.10",
        "args": ["/path/to/mcp_server.py"]
      }
    }
  }
"""
import json
import os
import sys

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, 'results.json')

sys.path.insert(0, BASE_DIR)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name='gandiva',
    instructions=(
        'Gandiva is an NSE India stock scanner based on Mark Minervini\'s SEPA methodology. '
        'Use get_top_setups to find Grade-A VCP candidates. '
        'Use get_market_regime to understand current market conditions before sizing positions. '
        'Always validate setups against Minervini\'s book using query_minervini_book.'
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_results() -> dict:
    if not os.path.exists(RESULTS_JSON):
        return {}
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _get_screen_stocks(results: dict, screen_key: str) -> list:
    """Handle both old flat format and new nested screens format."""
    screens = results.get('screens', {})
    screen  = screens.get(screen_key, {})
    if isinstance(screen, dict):
        return screen.get('stocks', [])
    if isinstance(screen, list):
        return screen
    # Fallback: top-level key
    top = results.get(screen_key, [])
    return top if isinstance(top, list) else []


ALL_SCREENS = [
    'full_template', 'near_breakout', 'vcp_setup', 'c1_to_c6',
    'watch_list', 'fo_strong_uptrend', 'ma_pullback', 'hhhl_pullback',
    'ck_daily_100', 'ck_weekly_100', 'ck_mtf_100', 'cci34_best_setups',
    'rs_leaders', 'strong_earnings', 'new_highs', 'newly_added',
]


# ── Tool 1: Market Regime ─────────────────────────────────────────────────────

@mcp.tool()
def get_market_regime() -> str:
    """
    Returns the current NSE market regime and breadth data.
    Use this first to determine position sizing and risk appetite.
    """
    results = _load_results()
    ma      = results.get('market_analysis', {})
    if not ma:
        return 'No scan data available. Run the scanner first.'

    regime  = ma.get('regime', 'Unknown')
    breadth = ma.get('breadth', {})
    lines   = [
        f'Market Regime: {regime}',
        f'Generated at: {results.get("generated_at", "unknown")}',
        f'Stocks scanned: {results.get("scanned", 0)}',
    ]
    if breadth:
        lines += [
            f'% above MA50:  {breadth.get("pct_above_ma50", "N/A")}%',
            f'% above MA200: {breadth.get("pct_above_ma200", "N/A")}%',
            f'Breadth label: {breadth.get("label", "N/A")}',
        ]
    return '\n'.join(lines)


# ── Tool 2: Get Screen ────────────────────────────────────────────────────────

@mcp.tool()
def get_screen(screen_name: str, limit: int = 20) -> str:
    """
    Returns stocks from a specific scanner screen.

    Available screens: full_template, near_breakout, vcp_setup, rs_leaders,
    new_highs, watch_list, ma_pullback, hhhl_pullback, fo_strong_uptrend,
    cci34_best_setups, strong_earnings, newly_added, c1_to_c6.

    Args:
        screen_name: Name of the screen (e.g. 'vcp_setup', 'near_breakout')
        limit: Max stocks to return (default 20)
    """
    results = _load_results()
    stocks  = _get_screen_stocks(results, screen_name)
    if not stocks:
        return f'No stocks found in screen "{screen_name}". Check screen name or run scanner.'

    lines = [f'Screen: {screen_name} — {len(stocks)} stocks (showing top {min(limit, len(stocks))})']
    for s in stocks[:limit]:
        ticker   = s.get('ticker', '?')
        price    = s.get('price', 'N/A')
        rs       = s.get('rs_rank', s.get('rs_rating', 'N/A'))
        vol      = s.get('vol_ratio', 'N/A')
        hi_dist  = s.get('pct_from_high', 'N/A')
        lines.append(f'  {ticker:12s} price={price:>8}  RS={rs:>3}  vol_ratio={vol}  from_52wHigh={hi_dist}%')

    return '\n'.join(lines)


# ── Tool 3: Top Setups ────────────────────────────────────────────────────────

@mcp.tool()
def get_top_setups(min_rs: int = 80, limit: int = 10) -> str:
    """
    Returns the best VCP candidates across ALL 16 screens, ranked by Minervini score.
    Stocks appearing in multiple screens are ranked higher (high-confluence setups).

    Args:
        min_rs: Minimum RS rank to include (default 80 — top 20% leaders only)
        limit:  Max setups to return (default 10)
    """
    results  = _load_results()
    seen: dict = {}

    for key in ALL_SCREENS:
        for s in _get_screen_stocks(results, key):
            ticker = s.get('ticker', '')
            if not ticker:
                continue
            rs = s.get('rs_rank', s.get('rs_rating', 0)) or 0
            if rs < min_rs:
                continue
            if ticker not in seen:
                seen[ticker] = {'stock': s, 'screens': [], 'rs': rs}
            seen[ticker]['screens'].append(key)

    if not seen:
        return f'No stocks found with RS ≥ {min_rs}. Try lowering min_rs.'

    # Rank: screen count desc, then RS desc
    ranked = sorted(seen.values(), key=lambda x: (len(x['screens']), x['rs']), reverse=True)

    lines = [f'Top VCP candidates (RS ≥ {min_rs}) across all screens:\n']
    for item in ranked[:limit]:
        s        = item['stock']
        ticker   = s.get('ticker', '?')
        price    = s.get('price', 'N/A')
        rs       = item['rs']
        vol      = s.get('vol_ratio', 'N/A')
        hi_dist  = s.get('pct_from_high', 'N/A')
        screens  = ', '.join(item['screens'][:4])
        n        = len(item['screens'])
        lines.append(
            f'{ticker} (RS={rs}, price={price}, vol_ratio={vol}, from_high={hi_dist}%)\n'
            f'  → {n} screens: {screens}'
        )

    return '\n'.join(lines)


# ── Tool 4: Validate Stock ────────────────────────────────────────────────────

@mcp.tool()
def validate_stock(ticker: str) -> str:
    """
    Runs Minervini's VCP validation on a specific stock ticker.
    Returns a detailed score breakdown with passes/failures on each criterion.

    Args:
        ticker: NSE stock symbol (e.g. 'DIXON', 'HFCL', 'BAJAJCON')
    """
    results  = _load_results()
    found    = None

    for key in ALL_SCREENS:
        for s in _get_screen_stocks(results, key):
            if s.get('ticker', '').upper() == ticker.upper():
                found = s
                break
        if found:
            break

    if not found:
        return f'{ticker} not found in any screen. Either not scanned or below all criteria.'

    # Basic Minervini criteria check
    criteria = found.get('criteria', {})
    price    = found.get('price', 0)
    rs       = found.get('rs_rank', found.get('rs_rating', 0))
    hi52     = found.get('hi52', 0)
    lo52     = found.get('lo52', 0)
    hi_dist  = found.get('pct_from_high', 0)
    vol      = found.get('vol_ratio', 1)
    passed   = found.get('passed', 0)

    score = 0
    checks = []

    # C1-C6 (Stage 2)
    c_pass = sum(1 for v in criteria.values() if v) if criteria else passed
    score += c_pass * 8
    checks.append(f'Stage 2 criteria: {c_pass}/8 passed ({c_pass*8} pts)')

    # RS leader
    if rs >= 90:
        score += 20
        checks.append(f'RS rank {rs} ≥ 90 — elite leader (+20 pts)')
    elif rs >= 80:
        score += 10
        checks.append(f'RS rank {rs} ≥ 80 — leader (+10 pts)')
    else:
        checks.append(f'RS rank {rs} < 80 — not a leader (0 pts)')

    # Near 52-week high
    if hi_dist is not None and hi_dist >= -5:
        score += 15
        checks.append(f'{hi_dist}% from 52wk high — tight base (+15 pts)')
    elif hi_dist is not None and hi_dist >= -10:
        score += 8
        checks.append(f'{hi_dist}% from 52wk high — acceptable (+8 pts)')
    else:
        checks.append(f'{hi_dist}% from 52wk high — too far from high (0 pts)')

    # Volume drying up
    if vol is not None and vol <= 0.6:
        score += 15
        checks.append(f'Vol ratio {vol} ≤ 0.6 — volume drying up (+15 pts)')
    elif vol is not None and vol <= 0.9:
        score += 8
        checks.append(f'Vol ratio {vol} — moderate contraction (+8 pts)')
    else:
        checks.append(f'Vol ratio {vol} — volume NOT drying up (0 pts)')

    grade = 'A' if score >= 80 else 'B' if score >= 60 else 'FAIL'

    lines = [
        f'=== Minervini VCP Validation: {ticker} ===',
        f'Price: ₹{price}  |  RS: {rs}  |  Grade: {grade}  |  Score: {score}/100',
        '',
        'Criterion breakdown:',
    ]
    for c in checks:
        lines.append(f'  {"✅" if "pts)" in c and "(0 pts)" not in c else "❌"} {c}')

    if grade == 'A':
        entry_low  = price
        entry_high = round(price * 1.02, 2)
        stop       = round(price * 0.92, 2)
        target     = round(price * 1.20, 2)
        lines += [
            '',
            f'Trade Setup (Grade-A):',
            f'  Entry zone: ₹{entry_low} – ₹{entry_high}',
            f'  Stop loss:  ₹{stop} (-8%)',
            f'  Target:     ₹{target} (+20%)',
            f'  R:R ratio:  2.5x',
            '',
            '⚠️  Visually verify contraction symmetry on chart before entering.',
        ]
    else:
        lines.append(f'\n{ticker} is Grade-{grade} — wait for a better setup.')

    return '\n'.join(lines)


# ── Tool 5: Query Book ────────────────────────────────────────────────────────

@mcp.tool()
def query_minervini_book(question: str) -> str:
    """
    Searches "Trade Like a Stock Market Wizard" by Mark Minervini using semantic search.
    Returns the most relevant passages from the book for the given question.

    Args:
        question: Any question about Minervini's methodology, VCP, SEPA, position sizing, etc.
    """
    try:
        from ai_agent import _query_rag
        passages = _query_rag(question, n=3)
        if not passages:
            return 'Book RAG index not found. Run build_rag.py first.'
        lines = [f'Relevant passages from "Trade Like a Stock Market Wizard":\n']
        for p in passages:
            lines.append(f'--- Page {p["page"]} (relevance: {p["score"]:.2f}) ---')
            lines.append(p['text'])
            lines.append('')
        return '\n'.join(lines)
    except Exception as e:
        return f'Error querying book: {e}'


# ── Tool 6: Scan Summary ──────────────────────────────────────────────────────

@mcp.tool()
def get_scan_summary() -> str:
    """
    Returns a full summary of the last scan: stock counts per screen,
    market regime, and scan timestamp. Good starting point for any analysis.
    """
    results = _load_results()
    if not results:
        return 'No scan data. Run scanner first (click Scan Now in dashboard).'

    lines = [
        f'Last scan: {results.get("generated_at", "unknown")}',
        f'Total stocks scanned: {results.get("scanned", 0)}',
        '',
        'Screen counts:',
    ]

    screens = results.get('screens', {})
    for key in ALL_SCREENS:
        screen = screens.get(key, {})
        if isinstance(screen, dict):
            count = len(screen.get('stocks', []))
            label = screen.get('label', key)
        elif isinstance(screen, list):
            count = len(screen)
            label = key
        else:
            continue
        if count > 0:
            lines.append(f'  {label:35s} {count:>4} stocks')

    ma = results.get('market_analysis', {})
    if ma:
        lines += ['', f'Regime: {ma.get("regime", "Unknown")}']

    return '\n'.join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mcp.run(transport='stdio')
