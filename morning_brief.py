"""
Gandiva Morning Brief — Agentic Pipeline.

Flow (ReAct-style):
  1. Load live scan data (all 16 screens)
  2. Run Minervini VCP validator on all candidates
  3. Query book RAG for setup-specific guidance
  4. Query history RAG for trend context
  5. Ask LLM to write a structured brief
  6. Format as HTML (save to exports/) + Markdown (send to Telegram)

Usage:
  python3 morning_brief.py              # generate + send Telegram alert
  python3 morning_brief.py --no-send   # generate HTML only, skip Telegram
"""
import json, os, sys, datetime, re

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(BASE_DIR, 'results.json')
EXPORTS_DIR  = os.path.join(BASE_DIR, 'exports', 'Briefs')
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env():
    env_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
_load_env()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# ── All 16 screen keys ────────────────────────────────────────────────────────
ALL_SCREENS = [
    'full_template', 'near_breakout', 'vcp_setup', 'c1_to_c6',
    'watch_list', 'fo_strong_uptrend', 'ma_pullback', 'hhhl_pullback',
    'ck_daily_100', 'ck_weekly_100', 'ck_mtf_100', 'cci34_best_setups',
    'rs_leaders', 'strong_earnings', 'new_highs', 'newly_added',
]

SCREEN_LABELS = {
    'full_template':    'Full Minervini Template',
    'near_breakout':    'Near Breakout',
    'vcp_setup':        'VCP Setup',
    'c1_to_c6':         'C1-C6 Criteria',
    'watch_list':       'Watch List',
    'fo_strong_uptrend':'F&O Strong Uptrend',
    'ma_pullback':      'MA Pullback',
    'hhhl_pullback':    'Higher-High Higher-Low',
    'ck_daily_100':     'CCI Daily 100',
    'ck_weekly_100':    'CCI Weekly 100',
    'ck_mtf_100':       'CCI Multi-TF 100',
    'cci34_best_setups':'CCI Best Setups',
    'rs_leaders':       'RS Leaders',
    'strong_earnings':  'Strong Earnings',
    'new_highs':        'New Highs',
    'newly_added':      'Newly Added',
}


def _load_results():
    if not os.path.exists(RESULTS_JSON):
        return {}
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _collect_candidates(results):
    """Collect unique stocks appearing across all screens, annotated with which screens."""
    seen   = {}
    for key in ALL_SCREENS:
        stocks = results.get(key, [])
        if not isinstance(stocks, list):
            continue
        label = SCREEN_LABELS.get(key, key)
        for s in stocks:
            if not isinstance(s, dict):
                continue
            ticker = s.get('ticker') or s.get('symbol', '')
            if not ticker:
                continue
            if ticker not in seen:
                seen[ticker] = {'stock': s, 'screens': []}
            seen[ticker]['screens'].append(label)
    return seen


def _regime_summary(results):
    regime = results.get('regime', {})
    if not regime:
        return 'Regime data not available.'
    label   = regime.get('label', 'Unknown')
    nifty50 = regime.get('nifty50_pct_above_200ma', 'N/A')
    breadth = regime.get('breadth_label', '')
    return (f'Market Regime: *{label}*  |  '
            f'Nifty50 stocks above 200MA: {nifty50}%  |  '
            f'Breadth: {breadth}')


def _run_agent_brief(candidates, regime_text, results):
    """Use ai_agent context stuffing + LLM to write the morning brief."""
    from ai_agent import run_agent
    # Build a rich context message for the agent
    screen_counts = results.get('meta', {})
    top_stocks = []
    for ticker, info in list(candidates.items())[:20]:
        s = info['stock']
        screens = ', '.join(info['screens'][:3])
        rs  = s.get('rs_rating', 'N/A')
        vol = s.get('rel_vol', s.get('volume_ratio', 'N/A'))
        top_stocks.append(f'{ticker} (RS={rs}, RelVol={vol}, screens: {screens})')

    stock_list = '\n'.join(f'- {x}' for x in top_stocks) if top_stocks else 'No candidates found.'

    message = f"""Generate a Minervini-style morning brief for today {datetime.date.today()}.

{regime_text}

Top candidates appearing across multiple screens:
{stock_list}

Total stocks scanned across all 16 screens: {len(candidates)} unique candidates.

Please:
1. State the market regime and what it means for position sizing
2. Identify the top 3-5 Grade-A VCP setups with entry zones, stops, and targets
3. Call out any stocks appearing in 3+ screens (high-confluence setups)
4. Give a risk management reminder based on the regime
5. End with a one-line Minervini quote from the book relevant to today's conditions

Keep the response concise, actionable, and formatted for a trader reading it at 9 AM."""

    answer, tools_used, _ = run_agent(message, GROQ_API_KEY)
    return answer, tools_used


def _format_telegram(brief_text, regime_text, date_str):
    """Format brief as Telegram Markdown."""
    header = (
        f'\U0001F3F9 *GANDIVA MORNING BRIEF* \U0001F3F9\n'
        f'_{date_str}_\n'
        f'{"─" * 30}\n\n'
        f'{regime_text}\n\n'
        f'{"─" * 30}\n\n'
    )
    return header + brief_text


def _format_html(brief_text, regime_text, date_str, candidates, results):
    """Format brief as standalone HTML file."""
    escaped = brief_text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    # Convert **bold** to <strong> and *italic* to <em>
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         escaped)
    escaped = re.sub(r'\n', '<br>', escaped)

    # Build screen coverage table
    screen_rows = ''
    for ticker, info in sorted(candidates.items(), key=lambda x: -len(x[1]['screens'])):
        screens_html = ', '.join(info['screens'][:4])
        count        = len(info['screens'])
        badge        = f'<span class="badge b{min(count,3)}">{count} screens</span>'
        screen_rows += f'<tr><td><strong>{ticker}</strong></td><td>{badge}</td><td>{screens_html}</td></tr>\n'

    top_rows = screen_rows[:5000]  # limit table size

    regime_raw = regime_text.replace('*','')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gandiva Morning Brief — {date_str}</title>
<style>
  body {{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
  h1 {{color:#d2a679;font-size:1.6em;margin-bottom:4px}}
  .subtitle {{color:#8b949e;font-size:.9em;margin-bottom:20px}}
  .regime-bar {{background:#161b22;border-left:4px solid #3fb950;padding:12px 16px;border-radius:6px;margin-bottom:20px;font-size:.95em}}
  .brief-box {{background:#161b22;padding:20px;border-radius:8px;line-height:1.7;font-size:.95em;margin-bottom:24px}}
  table {{width:100%;border-collapse:collapse;font-size:.87em}}
  th {{background:#21262d;color:#8b949e;padding:8px 12px;text-align:left;border-bottom:1px solid #30363d}}
  td {{padding:7px 12px;border-bottom:1px solid #21262d}}
  tr:hover td {{background:#161b22}}
  .badge {{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.8em;font-weight:600}}
  .b1 {{background:#1f3d2e;color:#3fb950}} .b2 {{background:#2d2a1a;color:#d29922}} .b3 {{background:#3d1f1f;color:#f85149}}
  .footer {{margin-top:30px;color:#484f58;font-size:.8em;text-align:center}}
</style>
</head>
<body>
<h1>\U0001F3F9 Gandiva Morning Brief</h1>
<div class="subtitle">{date_str} &nbsp;|&nbsp; Minervini SEPA + VCP Scanner</div>

<div class="regime-bar">{regime_raw}</div>

<div class="brief-box">{escaped}</div>

<h2 style="color:#8b949e;font-size:1em;margin-bottom:10px">SCREEN COVERAGE — {len(candidates)} candidates</h2>
<table>
<thead><tr><th>Ticker</th><th>Coverage</th><th>Screens</th></tr></thead>
<tbody>{screen_rows}</tbody>
</table>

<div class="footer">Generated by Gandiva Scanner &nbsp;|&nbsp; Not financial advice &nbsp;|&nbsp; Always verify charts</div>
</body>
</html>"""


def run_morning_brief(send_telegram=True):
    """Main entry point. Returns (html_path, telegram_sent)."""
    print('\n  \U0001F3F9 Gandiva Morning Brief starting...')

    results    = _load_results()
    candidates = _collect_candidates(results)
    regime_text = _regime_summary(results)
    date_str    = datetime.date.today().strftime('%A, %d %B %Y')

    print(f'  Loaded {len(candidates)} unique candidates across all screens')
    print(f'  {regime_text.replace("*","")}')
    print('  Calling AI agent for analysis...')

    brief_text, tools_used = _run_agent_brief(candidates, regime_text, results)
    print(f'  Agent tools used: {tools_used}')

    # Save HTML
    fname    = f'brief_{datetime.date.today().isoformat()}.html'
    html_path = os.path.join(EXPORTS_DIR, fname)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(_format_html(brief_text, regime_text, date_str, candidates, results))
    print(f'  HTML brief saved → {html_path}')

    telegram_sent = False
    if send_telegram:
        from telegram_bot import load_chat_id, send_message
        chat_id = load_chat_id()
        if chat_id:
            tg_text = _format_telegram(brief_text, regime_text, date_str)
            # Telegram has 4096 char limit — split if needed
            chunks = [tg_text[i:i+4000] for i in range(0, len(tg_text), 4000)]
            for chunk in chunks:
                send_message(chat_id, chunk)
            telegram_sent = True
            print(f'  Telegram alert sent to chat_id={chat_id}')
        else:
            print('  No chat_id registered — run: python3 telegram_bot.py')

    return html_path, telegram_sent


if __name__ == '__main__':
    send = '--no-send' not in sys.argv
    html_path, tg_sent = run_morning_brief(send_telegram=send)
    print(f'\n  Done.')
    print(f'  Brief: {html_path}')
    print(f'  Telegram: {"sent" if tg_sent else "skipped"}')
