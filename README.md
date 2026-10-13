# AlphaEdge — NSE India Stock Scanner with AI Agent

> Scans 1,300+ NSE stocks daily using Mark Minervini's SEPA methodology, validates setups with a RAG-powered AI agent grounded in his book, and delivers trade briefs to your Telegram at 8 PM — all running locally at zero cost.

---

## What It Does

| Layer | What happens |
|-------|-------------|
| **Scanner** | Downloads live OHLCV data for 1,312 NSE stocks, applies Minervini's 8 Stage 2 criteria, detects VCP patterns, pulls CCI signals from Chartink |
| **Dashboard** | 16 screens (Full Template, VCP, MA Pullback, RS Leaders, New Highs, F&O, CCI, and more), "New This Scan" diff, market breadth & regime |
| **AI Agent** | Chat panel powered by Groq (Llama 3.3 70B) — validates stocks against Minervini's book using RAG, cites page numbers, outputs structured trade cards |
| **Morning Brief** | Agentic pipeline runs every evening: scan → validate → LLM writes brief → sends to Telegram |
| **MCP Server** | 6 tools exposed via Model Context Protocol — Claude Desktop can query live NSE data with natural language |
| **Observability** | Every AI call traced in self-hosted Langfuse (Docker) with latency, token usage, retrieval scores |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AlphaEdge                            │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐ │
│  │ scanner  │───▶│results.json│───▶│   dashboard.html     │ │
│  │  .py     │    │           │    │   (16 screens)        │ │
│  └──────────┘    └─────┬─────┘    └──────────────────────┘ │
│                        │                                     │
│              ┌─────────▼──────────┐                         │
│              │     ai_agent.py    │                         │
│              │                    │                         │
│              │  ┌──────────────┐  │   ┌──────────────────┐ │
│              │  │  Book RAG    │  │   │  Groq 70B cloud  │ │
│              │  │  (ChromaDB)  │  │──▶│  OR              │ │
│              │  ├──────────────┤  │   │  Ollama 3B local │ │
│              │  │ History RAG  │  │   └──────────────────┘ │
│              │  │  (JSONL)     │  │                         │
│              │  └──────────────┘  │                         │
│              └────────┬───────────┘                         │
│                       │                                     │
│         ┌─────────────┼──────────────┐                      │
│         ▼             ▼              ▼                      │
│  ┌────────────┐ ┌──────────┐ ┌────────────┐                │
│  │ Telegram   │ │ Langfuse │ │    MCP     │                │
│  │    Bot     │ │ (Docker) │ │   Server   │                │
│  │ @Gandiva   │ │localhost │ │ 6 tools    │                │
│  │ ScannerBot │ │  :3000   │ │ for Claude │                │
│  └────────────┘ └──────────┘ └────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 16 Scanner Screens
- Full Minervini Template (all 8 Stage 2 criteria)
- VCP Setup (Volatility Contraction Pattern)
- Near Breakout (within 3% of pivot)
- RS Leaders (top 20% relative strength)
- MA Pullback, HHHL Pullback
- CCI34 Daily / Weekly / MTF
- F&O Strong Uptrend
- New Highs, Strong Earnings
- "New This Scan" — only stocks that appeared since the last run

### AI Agent — Grounded, Not Hallucinating
- RAG over *"Trade Like a Stock Market Wizard"* — answers cite exact page numbers
- RAG over scan history — answers temporal queries ("when did DIXON first appear?")
- Structured trade cards: entry zone, stop loss, target, R:R — extracted from LLM output and rendered visually
- SSE streaming — word-by-word, JSON block suppressed until fully parsed

### MCP Server — Talk to Your Scanner from Claude Desktop
```
You: "What are the best VCP setups right now?"

Claude autonomously:
  1. get_market_regime()       → Strong Bull, 1,358 stocks
  2. get_top_setups(min_rs=80) → HFCL, BAJAJCON, GRWRHITECH...
  3. validate_stock("HFCL")    → entry ₹214, stop ₹197, target ₹257
  4. query_minervini_book()    → "Stage 2 requires price above rising 200-day MA (p.87)"
```

### Automated Daily Pipeline
```
8:00 PM (launchd) → scanner.py → morning_brief.py → Telegram
```
If Mac was off, runs once on wake and catches up. Fully offline if needed (Ollama fallback).

### Observability
Every AI call traced in self-hosted Langfuse:
- Input/output token counts
- RAG retrieval relevance scores
- Latency per span (RAG fetch, LLM generation)
- Full prompt/response logged for debugging

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Data | yfinance + Chartink scrape | Free real-time NSE data |
| Vector DB | ChromaDB (local) | No server, HNSW index, free |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, runs on CPU |
| LLM (primary) | Groq / Llama 3.3 70B | Free tier, fast |
| LLM (fallback) | Ollama / Llama 3.2 3B | Fully offline, zero cost |
| Streaming | SSE (Server-Sent Events) | Browser-native, no WebSocket |
| Observability | Langfuse v2 (Docker) | Self-hosted, free forever |
| Scheduler | macOS launchd | Survives reboots and sleep |
| AI Protocol | MCP (Model Context Protocol) | Claude Desktop tool integration |
| Alerts | Telegram Bot API | Free, instant delivery |
| UI | Vanilla HTML/JS | Zero framework dependencies |

**Monthly cost: ₹0**

---

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop (for Langfuse observability — optional)
- Ollama (for local LLM fallback — optional)

### 1. Clone and install
```bash
git clone <repo-url>
cd minervini-trend-template-scanner
pip install -r requirements.txt
pip install chromadb sentence-transformers yfinance groq
```

### 2. Configure secrets
Create a `.env` file:
```
GROQ_API_KEY=gsk_...              # free at console.groq.com
TELEGRAM_BOT_TOKEN=...            # optional: @BotFather on Telegram
LANGFUSE_PUBLIC_KEY=pk-lf-...     # optional: only if running Langfuse
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### 3. Build RAG indexes (one-time)
```bash
# Index Minervini's book (add PDF to rag_db/ first)
python3 build_rag.py

# Index scan history (run after each scan automatically)
python3 build_history_rag.py
```

### 4. Run
```bash
# Start dashboard
python3 server.py
open http://localhost:8765

# Run scanner manually
python3 scanner.py

# Morning brief
python3 morning_brief.py

# MCP server (auto-started by Claude Desktop)
python3 mcp_server.py
```

### 5. Enable daily 8 PM scan (macOS)
```bash
cp com.gandiva.scanner.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.gandiva.scanner.plist
```

### 6. Langfuse observability (optional)
```bash
cd langfuse && ./start.sh
open http://localhost:3000
```

---

## MCP — Connect Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "gandiva": {
      "command": "/opt/homebrew/bin/python3.10",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. The "gandiva" connector will appear. Claude can now query your scanner directly.

**Available tools:** `get_market_regime`, `get_screen`, `get_top_setups`, `validate_stock`, `query_minervini_book`, `get_scan_summary`

---

## Project Structure

```
.
├── scanner.py              # NSE scan engine — 16 screens, 1,312 stocks
├── scanner_us.py           # US market variant
├── server.py               # Local HTTP server + API routes
├── dashboard.html          # Full-featured web UI
├── ai_agent.py             # LLM agent with dual RAG + trade cards
├── mcp_server.py           # MCP server — 6 tools for Claude Desktop
├── morning_brief.py        # Agentic daily brief pipeline
├── telegram_bot.py         # Telegram delivery
├── daily_run.py            # Orchestrator for launchd scheduler
├── build_rag.py            # Book RAG index builder
├── build_history_rag.py    # Scan history RAG index builder
├── langfuse/               # Docker Compose for self-hosted Langfuse
│   ├── docker-compose.yml
│   ├── start.sh
│   └── stop.sh
├── rag_db/                 # ChromaDB vector store (book + history)
├── results.json            # Latest scan output
├── scan_history.jsonl      # Full scan timeline
├── requirements.txt
└── DOCUMENTATION.md        # Deep technical docs (zero to interview-ready)
```

---

## Backtesting Results (NSE India, 2004–2024)

The entry/exit logic is validated against 20 years of NSE data:

| Metric | Result |
|--------|--------|
| CAGR | 35.07% |
| Win Rate | 52.5% |
| Profit Factor | 3.57 |
| Max Drawdown | -16.9% |
| Nifty 50 CAGR (same period) | ~13% |

---

## Limitations

- Requires Mac to be on for launchd scheduling (or migrate to GitHub Actions / VPS)
- Book RAG requires you to provide the PDF (copyright — not included in repo)
- MCP server uses local stdio transport — works on same machine as Claude Desktop only
- US market backtest CAGR ~18% (India secular bull 2014–2024 is the ideal condition for this strategy)

---

## Full Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete technical depth — architecture decisions, RAG internals, Langfuse setup, MCP protocol, Ollama integration, backtesting methodology, and interview Q&As for every component.

---

*Built with Mark Minervini's SEPA methodology as the domain foundation. Not financial advice.*
