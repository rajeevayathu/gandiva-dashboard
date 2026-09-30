# GANDIVA — NSE Stock Scanner (CLAUDE.md)

## What This Is

**AlphaEdge / GANDIVA** — an AI-powered NSE India stock scanner built on Mark Minervini's
SEPA methodology. Scans 1,312 stocks daily, validates setups with a RAG-grounded AI agent,
and delivers trade briefs via Telegram. Runs entirely locally at zero cost.

## Key Facts

- **Python**: `/opt/homebrew/bin/python3.10`
- **Port**: 8765 (dashboard server)
- **Cache**: `/tmp/bt_cache_30yr/` (India), `/tmp/bt_cache_us/` (US)
- **Branch**: `feature/ai-agent` (default branch on GitHub)
- **Repo**: https://github.com/rajeevayathu/nse-minervini-scanner

---

## File Map

```
scanner.py              ← NSE scan engine: 16 screens, 1,312 stocks, Minervini criteria
scanner_us.py           ← US market variant (S&P 500)
server.py               ← Local HTTP server + all API routes (port 8765)
dashboard.html          ← Full web UI: screens, filters, AI chat, trade cards
ai_agent.py             ← LLM agent: dual RAG, trade card extraction, streaming
mcp_server.py           ← MCP server: 6 tools exposed to Claude Desktop
morning_brief.py        ← Agentic pipeline: scan → validate → LLM → Telegram
telegram_bot.py         ← Telegram delivery (@GandivaScannerBot)
daily_run.py            ← Scheduler orchestrator (called by launchd at 8 PM)
build_rag.py            ← Builds book RAG index (Minervini book PDF → ChromaDB)
build_history_rag.py    ← Builds scan history RAG index (scan_history.jsonl → ChromaDB)
langfuse/               ← Docker Compose for self-hosted Langfuse (port 3000)
results.json            ← Latest scan output (gitignored)
scan_history.jsonl      ← Full scan timeline (gitignored)
rag_db/                 ← ChromaDB vector stores: book + history (gitignored)
.env                    ← All secrets (gitignored)
DOCUMENTATION.md        ← 1,700-line deep technical reference (zero → interview-ready)
README.md               ← Public GitHub README
```

---

## How to Run

```bash
# Dashboard
/opt/homebrew/bin/python3.10 server.py
open http://localhost:8765

# Manual scan
/opt/homebrew/bin/python3.10 scanner.py

# Morning brief
/opt/homebrew/bin/python3.10 morning_brief.py

# Daily run (scanner + brief)
/opt/homebrew/bin/python3.10 daily_run.py

# Langfuse observability
cd langfuse && ./start.sh
open http://localhost:3000

# MCP server (normally auto-started by Claude Desktop)
/opt/homebrew/bin/python3.10 mcp_server.py
```

---

## Architecture

```
scanner.py → results.json → dashboard.html (via server.py)
                          → ai_agent.py (RAG + LLM + trade cards)
                          → morning_brief.py → telegram_bot.py
                          → mcp_server.py (Claude Desktop tools)

ai_agent.py uses:
  - rag_db/book_collection    ← Minervini book (1,203 chunks)
  - rag_db/history_collection ← scan history
  - Groq llama-3.3-70b (primary) OR Ollama llama3.2 (fallback)
  - Langfuse for tracing (optional)
```

---

## Critical Rules (DO NOT CHANGE)

These caused regressions when changed — learned through backtesting:

1. **No liquidity gate in scan loop** — adding one breaks 2014 WR (76% → 46%)
2. **RS as score only, no hard gate** — RS gates remove valid VCP breakout winners
3. **DD halt is permanent** — smart reset caused MaxDD to blow out to -30%
4. **SUPERPERF disabled for India** — causes fast breakeven exits, WR drops 53% → 46%
5. **Kelly quality_mult disabled for India** — dilutes performance with marginal trades
6. **`ipo_mode`** — defined as `n_bars < IPO_MAX_DAYS or pd.isna(s200)` — do not add h52/l52 NaN checks

---

## Environment Variables (.env)

```
GROQ_API_KEY=gsk_...
TELEGRAM_BOT_TOKEN=...          # @GandivaScannerBot
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

---

## MCP Tools (mcp_server.py)

6 tools exposed to Claude Desktop via stdio transport:
1. `get_market_regime` — NSE regime + breadth
2. `get_screen(name, limit)` — stocks from any of 16 screens
3. `get_top_setups(min_rs, limit)` — Grade-A VCPs ranked by confluence
4. `validate_stock(ticker)` — full Minervini score breakdown
5. `query_minervini_book(question)` — semantic search over book RAG
6. `get_scan_summary` — counts per screen + regime

Claude Desktop config: `~/Library/Application Support/Claude/claude_desktop_config.json`

---

## Scheduler (launchd)

Plist: `~/Library/LaunchAgents/com.gandiva.scanner.plist`
Fires: 8 PM daily. If Mac was off, runs once on wake.

```bash
launchctl list | grep gandiva          # check status
tail -f logs/daily_run.log             # live logs
```

---

## Backtesting Results (canslim_sepa script)

| Metric | Result |
|--------|--------|
| CAGR | 35.07% |
| Win Rate | 52.5% |
| Profit Factor | 3.57 |
| Max Drawdown | -16.9% |

Backtest scripts live in `~/Downloads/` — separate from this repo (not pushed to GitHub).

---

## Related Project: Portfolio Site

Located at: `../portfolio-site/`
The portfolio site features GANDIVA as Rajeev's personal AI engineering project.
Has its own CLAUDE.md.
