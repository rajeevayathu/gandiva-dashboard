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
    screens_data = results.get('screens', results)  # support both old flat and new nested format
    for key in ALL_SCREENS:
        raw = screens_data.get(key, [])
        # new format: {'label':..., 'stocks': [...]}
        stocks = raw.get('stocks', []) if isinstance(raw, dict) else raw
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
    ma = results.get('market_analysis', {})
    if not ma:
        return 'Regime data not available.'
    label       = ma.get('regime', 'Unknown')
    nifty_close = ma.get('nifty_close', 'N/A')
    trend       = ma.get('trend', '')
    deploy      = ma.get('deploy_pct', '')
    action_note = ma.get('action', '')[:120] if ma.get('action') else ''
    return (f'Market Regime: *{label}*  |  '
            f'Nifty: {nifty_close}  |  '
            f'Trend: {trend}  |  '
            f'Deploy: {deploy}%\n_{action_note}_')


def _run_agent_brief(candidates, regime_text, results):
    """Write the morning brief via a direct, lightweight LLM call (no RAG pipeline)."""
    ma = results.get('market_analysis', {})
    screens_data = results.get('screens', {})

    # Grade-A candidates: in VCP + full_template, sorted by multi-screen count
    vcp_tickers = {
        s.get('ticker') for s in
        screens_data.get('vcp_setup', {}).get('stocks', [])
    }
    ranked = sorted(candidates.items(), key=lambda x: -len(x[1]['screens']))

    grade_a, other = [], []
    for ticker, info in ranked:
        s = info['stock']
        rs   = s.get('rs_rank', '?')
        price = s.get('price', '?')
        hi_pct = s.get('pct_from_high')
        vol  = s.get('vol_ratio')
        pivot = s.get('pivot_high')
        vcp_d = s.get('vcp_last_daily', '')
        scr  = ', '.join(info['screens'][:4])
        hi_str  = f"{hi_pct:+.1f}%" if hi_pct is not None else '?'
        vol_str = f"{vol:.2f}x"     if vol    is not None else '?'
        line = (f"{ticker} | ₹{price} | RS {rs} | {hi_str} from 52wk hi | "
                f"vol {vol_str} | pivot ₹{pivot} | VCP:{vcp_d or 'none'} | screens: {scr}")
        if ticker in vcp_tickers and len(info['screens']) >= 2:
            grade_a.append(line)
        elif len(other) < 10:
            other.append(line)

    multi_screen = [(t, len(i['screens'])) for t, i in ranked if len(i['screens']) >= 3][:8]
    multi_str = '\n'.join(f"  {t}: {n} screens" for t, n in multi_screen) or '  None'

    prompt = f"""You are Gandiva, a Minervini-style stock scanner AI. Write a concise morning brief.

DATE: {datetime.date.today().strftime('%A, %d %B %Y')}
REGIME: {ma.get('regime','?')} | Nifty {ma.get('nifty_close','?')} | Trend: {ma.get('trend','?')}
DEPLOY SIGNAL: {ma.get('deploy_pct','?')}% allocation | {(ma.get('action',''))[:200]}

GRADE-A VCP SETUPS (VCP + Full Template, 2+ screens):
{chr(10).join(grade_a[:6]) or 'None today'}

HIGH-CONFLUENCE (3+ screens):
{multi_str}

TOTAL CANDIDATES: {len(candidates)} unique across all screens

Write the brief in this format — keep it under 350 words total:
1. MARKET REGIME — what it means for sizing (2 sentences)
2. TOP SETUPS — for each Grade-A: entry zone, stop (8% below entry), target, R:R (3-5 stocks)
3. HIGH-CONFLUENCE — stocks in 3+ screens deserve priority (1-2 sentences)
4. RISK NOTE — one practical rule for today's regime
5. MINERVINI QUOTE — one relevant line from "Trade Like a Stock Market Wizard"

Be specific with numbers. No generic filler."""

    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=600,
                temperature=0.4,
            )
            return resp.choices[0].message.content, ['direct_llm']
        except Exception as e:
            print(f'  Groq error: {e}')

    # Ollama fallback
    try:
        from openai import OpenAI
        client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        resp = client.chat.completions.create(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=600,
            temperature=0.4,
        )
        return resp.choices[0].message.content, ['ollama_fallback']
    except Exception as e:
        return f'LLM unavailable: {e}', []


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
