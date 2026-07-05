# AlphaEdge — Complete Technical Documentation

**Project:** NSE Minervini Scanner with AI Agent Layer  
**Author:** Rajeev Sharma  
**Stack:** Python · ChromaDB · Groq · Llama 3.3 70B · Sentence Transformers · ChromaDB  
**Purpose:** This document explains every technical decision, concept, and component built in this project — from zero knowledge to interview-ready depth.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Scanner — How It Works](#2-the-scanner--how-it-works)
3. [The Dashboard](#3-the-dashboard)
4. [What is AI / LLM?](#4-what-is-ai--llm)
5. [What is RAG?](#5-what-is-rag)
6. [What is an Embedding?](#6-what-is-an-embedding)
7. [What is ChromaDB?](#7-what-is-chromadb)
8. [Book RAG — Minervini Knowledge Base](#8-book-rag--minervini-knowledge-base)
9. [History RAG — Scan Timeline Memory](#9-history-rag--scan-timeline-memory)
10. [The AI Agent — How It Thinks](#10-the-ai-agent--how-it-thinks)
11. [The Complete Flow — Step by Step](#11-the-complete-flow--step-by-step)
12. [Minervini VCP Validator — How the AI Validates Like Minervini](#12-minervini-vcp-validator--how-the-ai-validates-like-minervini)
13. [Security — What Is Private, What Leaves Your Mac](#13-security--what-is-private-what-leaves-your-mac)
14. [File Structure — What Every File Does](#14-file-structure--what-every-file-does)
15. [Interview Questions & Answers](#15-interview-questions--answers)
16. [Agentic Workflow — Gandiva Morning Brief & Telegram Bot](#16-agentic-workflow--gandiva-morning-brief--telegram-bot)
17. [AI Observability — Langfuse](#17-ai-observability--langfuse)
18. [Daily Scheduler — How Automation Works](#18-daily-scheduler--how-automation-works)
19. [MCP Server — Model Context Protocol](#19-mcp-server--model-context-protocol)
20. [Ollama — Local LLM Fallback](#20-ollama--local-llm-fallback)
21. [Structured Trade Cards](#21-structured-trade-cards)
22. [How to Run — Complete Reference](#22-how-to-run--complete-reference)
23. [Portfolio Pitch — How to Present This Project](#23-portfolio-pitch--how-to-present-this-project)
24. [Features Built — Status Tracker](#24-features-built--status-tracker)

---

## 1. What This Project Is

### The Problem It Solves

There are 1,300+ stocks listed on NSE India. Every day, each stock moves. A trader following Mark Minervini's methodology needs to find stocks that are:

- In a proper Stage 2 uptrend (8 specific structural criteria)
- Showing momentum (RS rank, CCI signals)
- Forming a Volatility Contraction Pattern (VCP) — Minervini's ideal entry
- Backed by strong earnings

Doing this manually for 1,300 stocks every day would take 8–10 hours. This project does it in under 10 minutes.

### What We Built

```
Layer 1: SCANNER (scanner.py)
  Downloads live price data for 1,312 NSE stocks
  Applies Minervini's 8 criteria to every stock
  Pulls live CCI signals from Chartink
  Outputs: results.json (the master data file)

Layer 2: DASHBOARD (dashboard.html + server.py)
  Visual interface to explore all screens
  25+ screens: Full Template, VCP, MA Pullback, CCI, F&O, etc.
  "New This Scan" — shows what changed since yesterday
  Market breadth, regime detection

Layer 3: AI AGENT (ai_agent.py)
  Floating chat panel in the dashboard
  Powered by Groq (free) + Llama 3.3 70B
  Reads from TWO knowledge bases:
    - Minervini's book (RAG)
    - Your scan history (RAG)
  Validates stocks using Minervini's actual criteria
  Cites exact page numbers from the book
```

---

## 2. The Scanner — How It Works

### Data Sources

| Source | What It Provides | How |
|--------|-----------------|-----|
| Yahoo Finance (yfinance) | Daily OHLCV price data, 30 years | Python library, free |
| Chartink | Live CCI34 signals, weekly data | Web scraping |
| NSE India | FII/DII data, F&O list | API |
| Screener.in | Quarterly earnings (EPS, Revenue) | Web scraping |

### Minervini's 8 Criteria (C1–C8)

These are the exact criteria from "Trade Like a Stock Market Wizard":

| Criterion | What It Checks | Why It Matters |
|-----------|---------------|----------------|
| C1 | Price > 150-day MA AND > 200-day MA | Stock is above long-term trend |
| C2 | 150-day MA > 200-day MA | Long-term trend is rising |
| C3 | 200-day MA trending up for ≥1 month | Uptrend is established, not new |
| C4 | 50-day MA > 150-day MA AND > 200-day MA | Medium-term trend aligns with long-term |
| C5 | Price > 50-day MA | Stock is above medium-term trend |
| C6 | Price ≥ 30% above 52-week low | Stock has recovered strongly from its low |
| C7 | Price within 25% of 52-week high | Stock is near highs, not in a hole |
| C8 | RS rank ≥ 70 | Stock is outperforming most of the market |

**C1–C6 together = Stage 2 uptrend** (structural base confirmed)  
**C1–C8 all met = Full Template** (complete Minervini setup)

### What Happens Each Scan

```
1. Load cached price data (or download if stale)
2. For each of 1,312 stocks:
   a. Calculate all moving averages (MA50, MA150, MA200, EMA21, EMA10)
   b. Check each of the 8 criteria → True/False
   c. Calculate RS rank vs all other stocks
   d. Detect VCP pattern (volatility contraction algorithm)
   e. Fetch latest earnings data
3. Pull live Chartink data (CCI34 ≥ 100 daily/weekly)
4. Combine into screens
5. Save to results.json
6. Auto-snapshot to scan_history.jsonl
7. Re-index history RAG
```

### The VCP Detection Algorithm

A VCP (Volatility Contraction Pattern) is Minervini's signature entry pattern. The scanner detects it by looking for:

1. A series of price swings where each swing is **smaller** than the previous (contracting volatility)
2. Volume **decreasing** on each contraction (supply drying up)
3. Stock in a proper Stage 2 uptrend (C1–C6 met)
4. Price forming a tight base near 52-week highs

The scanner flags potential VCPs but the AI then applies a second validation layer using Minervini's actual book criteria (see Section 12).

---

## 3. The Dashboard

### What It Is

A single HTML file (`dashboard.html`) that loads `results.json` and renders everything visually. It runs in your browser.

### Two Modes

**File mode:** Open `dashboard.html` directly in browser — reads from embedded JS data files  
**Server mode:** Run `python3 server.py` → open `localhost:8765` — reads live from `results.json`, enables Scan Now button and AI chat

### The "New This Scan" Screen

One of the more technically interesting features:

**The problem:** How do you know what changed since yesterday?

**The solution:**
1. At the START of each scan, before overwriting anything, read the old `results.json`
2. Extract which stocks were in each tracked screen yesterday
3. Save that as `prev_screens.json` (only if it's a different day — same-day re-runs keep yesterday's baseline)
4. After the scan, compare today's screen members vs `prev_screens.json`
5. Any stock in today's screen that wasn't in yesterday's = "New This Scan"

This means: run the scanner every day, and "New This Scan" always shows you exactly what entered each screen overnight.

---

## 4. What is AI / LLM?

### LLM = Large Language Model

A Large Language Model is a neural network trained on vast amounts of text (books, websites, code, articles) to predict the next word in a sequence. Through this training it develops an understanding of language, facts, reasoning, and concepts.

**Examples:** GPT-4 (OpenAI), Claude (Anthropic), Llama (Meta), Gemini (Google)

**We use:** Llama 3.3 70B — Meta's open-source model with 70 billion parameters, run on Groq's hardware.

### What "70 billion parameters" means

Parameters are the numbers inside the neural network that were adjusted during training. More parameters = more capacity to learn complex patterns. 70B is large enough to reason well about complex topics like Minervini's trading methodology.

### What Groq Is

Groq is a company that built custom hardware (LPU — Language Processing Unit) specifically optimised to run LLMs very fast. They offer an API:

- **Free tier:** ~14,400 requests/day, 131,072 tokens/day
- **Speed:** Much faster than most providers
- **We use:** `llama-3.3-70b-versatile` model

### What a Token Is

Text is broken into tokens before being processed. Roughly:
- 1 word ≈ 1.3 tokens
- "VCP" = 1 token
- "Volatility Contraction Pattern" = 4 tokens
- Our typical question + context = ~3,000–5,000 tokens

The free tier's 131,072 daily tokens ≈ 26–40 questions per day.

### What a Prompt Is

Everything sent to the LLM in one request. It includes:

```
System prompt:     "You are AlphaEdge AI. You think like Minervini..."
Retrieved book:    "Page 214: VCP forms 2-4 contractions..."
Live scan data:    "Today: HFCL RS:99, VCP daily, base -3.6%..."
User question:     "Which stocks show a VCP today?"
```

The LLM reads all of this and generates a response. It has no memory between questions — each question is a fresh prompt.

### Why We Don't Use Tool Calling

Groq's Llama model was generating XML-format function calls (`<function=name/>`) instead of JSON tool calls — a known bug with certain Groq model versions. We solved this with **context stuffing**: we pre-load all relevant scan data into the prompt based on the query's intent, instead of asking the LLM to call functions. This is more reliable and faster (one API call instead of multiple).

---

## 5. What is RAG?

### The Problem

The LLM knows general facts from its training data. But it has never:
- Read your specific Minervini book
- Seen your live scan data
- Seen your historical scans

If you ask "What does Minervini say about VCP on page 214?" it will guess — and may be wrong.

### The Solution: RAG

**RAG = Retrieval Augmented Generation**

Three steps:
1. **Retrieval** — Search your private knowledge base for content relevant to the question
2. **Augmented** — Add that content to the LLM's prompt
3. **Generation** — LLM generates an answer grounded in your actual data

### The Analogy

You're in an exam. Without notes, you guess. With notes, you look up the right page and answer accurately. 

RAG = giving the AI the right pages at the right moment. The AI doesn't memorise the book — it reads the relevant pages on demand for each question.

### RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|--|-----|-------------|
| Cost | Near zero | $10,000–$1M+ |
| Updates | Instant (re-index) | Re-train the whole model |
| Accuracy | Cites exact sources | Can still hallucinate |
| Data privacy | Your data stays local | Data sent to training provider |
| Best for | Private docs, changing data | Teaching style or behaviour |

For our use case (a book + daily scan data that changes every day), RAG is clearly correct.

---

## 6. What is an Embedding?

### The Core Idea

Computers only understand numbers. An **embedding model** converts text into a list of numbers that represents the **meaning** of that text.

```
"VCP forms 2-4 contractions"      → [0.23, -0.45, 0.12, 0.67, ...] (384 numbers)
"Volatility contraction pattern"   → [0.21, -0.43, 0.14, 0.65, ...] (384 numbers)
"Stop loss at 7-8 percent"         → [0.89,  0.11, -0.34, 0.22, ...] (384 numbers)
```

Key observation: VCP and "Volatility contraction pattern" produce **very similar numbers** because they mean the same thing. "Stop loss" produces very different numbers.

This is called **semantic similarity** — meaning closeness expressed as mathematical closeness.

### The Model We Use: all-MiniLM-L6-v2

- Size: 90MB (downloads once, cached on your Mac)
- Output: 384-dimensional vector per text chunk
- Free, runs locally — no API needed
- Trained on hundreds of millions of sentence pairs to understand meaning

### What 384-Dimensional Means

Each piece of text is represented as a point in 384-dimensional space. Texts with similar meanings are close together in that space. Finding the most relevant chunks = finding the nearest points to your question's point.

### Cosine Similarity

The standard way to measure how similar two vectors are. It measures the angle between them:

```
Same direction (angle = 0°)    → similarity = 1.0  → identical meaning
Right angle   (angle = 90°)    → similarity = 0.0  → unrelated
Opposite      (angle = 180°)   → similarity = -1.0 → opposite meaning
```

We use **cosine distance** = 1 - cosine similarity. We only return chunks with distance < 0.7 (meaningfully related). Chunks with distance > 0.7 are too loosely related and discarded.

---

## 7. What is ChromaDB?

### What It Is

ChromaDB is an **open-source vector database** that:
1. Stores text chunks alongside their embedding vectors
2. Builds a fast search index (HNSW) over those vectors
3. Lets you find the most semantically similar chunks to any query — in milliseconds

### Regular Database vs Vector Database

```
Regular Database (SQL):
  Question: "Tell me about VCP"
  SELECT * FROM chunks WHERE text LIKE '%VCP%'
  → Finds exact word match only
  → Misses: "volatility contraction", "price contractions", "tightening base"

Vector Database (ChromaDB):
  Question: "Tell me about VCP"
  → Converts question to numbers
  → Finds chunks whose numbers are closest
  → Returns: "volatility contraction", "price contractions", 
             "supply being absorbed" — all semantically related
```

### HNSW Index — Why Search Is Fast

HNSW = Hierarchical Navigable Small World. It's a graph structure where:
- Similar vectors are connected to each other
- The graph has multiple layers (hierarchy)
- Search navigates the graph rather than checking every vector

Instead of comparing your query to all 1,203 book chunks one by one (brute force, slow), HNSW navigates the graph and finds nearest neighbours in O(log n) time — effectively instant even at millions of records.

### Our Two ChromaDB Collections

```
rag_db/
├── minervini_books     ← Book RAG
│   1,203 chunks from "Trade Like a Stock Market Wizard"
│   Each chunk: ~600 chars of book text + page number
│   Used for: conceptual questions, Minervini methodology
│
└── scan_history        ← History RAG  
    1,235 documents (1,234 stocks + 1 overview for today)
    Grows by ~1,235 docs every time you run the scanner
    Used for: HFCL trend, stocks appearing consistently, etc.
```

### Where ChromaDB Lives

Entirely on your Mac at:
```
minervini-trend-template-scanner/rag_db/
├── chroma.sqlite3              ← Main index, metadata, text content
├── d033bc24-.../               ← Book RAG binary files
│   ├── data_level0.bin         ← The actual embedding vectors
│   ├── index_metadata.pickle   ← HNSW graph structure
│   └── ...
└── b255b737-.../               ← History RAG binary files
```

Total size: ~15MB. Nothing in the cloud.

---

## 8. Book RAG — Minervini Knowledge Base

### What Was Indexed

**Source:** "Trade Like a Stock Market Wizard" by Mark Minervini (353 pages)  
**Content extracted:** 328 pages with actual text content (rest were blank/cover pages)  
**Chunks created:** 1,203  
**Chunk size:** ~600 characters with 100-character overlap between chunks

### Topics Covered (Every Chapter)

| Topic | Pages in Index |
|-------|---------------|
| VCP Pattern (contractions, volume, pivot) | 214, 216, 218, 220 |
| Stage 2 Uptrend / SEPA Criteria | 54, 79, 83, 94 |
| Stop Loss & Defensive Sell Rules | 305, 306, 310, 316, 317 |
| Position Sizing & Risk Per Trade | 211, 322, 326 |
| Relative Strength — Identifying Leaders | 180, 182, 200 |
| Earnings, EPS Growth, Revenue | 112, 146, 160 |
| When to Sell Winners | 314, 317 |
| Market Timing, Bull vs Bear | 81, 200 |
| Industry Groups & Catalysts | 97, 112, 126, 180 |
| Superperformance — What Creates 100%+ Gains | 38, 51 |
| Chart Reading — Pivot Points, Breakouts | 238, 244, 258, 264 |
| Volume Analysis — Accumulation vs Distribution | 218, 241 |
| IPO Stocks & New Issues | 52, 275 |
| Psychology, Discipline, Emotion | 305, 307 |
| Risk Management | 306, 308 |
| Entry Timing — Exact Buy Point | 244, 258 |
| Market Leaders, Institutional Buying | 111, 210 |
| Fundamental Analysis | 146, 160 |

### How the Chunking Works

```
Raw page text (page 214):
"When sellers become scarcer, the price correction will not be as 
dramatic, and volatility will decrease. Typically, most VCP setups 
will be formed by two to four contractions, although sometimes there 
can be as many as five or six. This action will produce a pattern..."

Chunk 1 (chars 0-600):
"When sellers become scarcer... two to four contractions..."

Chunk 2 (chars 500-1100):  ← 100 char overlap with chunk 1
"...two to four contractions... This action will produce a pattern..."

Why overlap? Without it, a key sentence split across two chunks 
would lose meaning in both. Overlap preserves context at boundaries.
```

### How It's Built

Run once (or when you add a new book):
```bash
python3 build_rag.py
```

What happens:
1. `pdfplumber` reads every page of the PDF
2. Pages with < 50 characters are skipped (blank/image pages)
3. Each page's text is split into overlapping chunks
4. `SentenceTransformer.encode()` converts each chunk to 384 numbers
5. Chunks are added to ChromaDB in batches of 64
6. ChromaDB builds HNSW index automatically
7. `rag_meta.json` records what was indexed

---

## 9. History RAG — Scan Timeline Memory

### The Problem It Solves

The book RAG answers "what is a VCP?" The history RAG answers "has HFCL been forming a VCP base over the past 2 weeks?" — temporal, stock-specific questions that require memory across scan dates.

### What Gets Stored Per Scan

Every time `scanner.py` finishes, it automatically:

1. Reads `results.json` (today's fresh scan)
2. Extracts for every stock: ticker, price, RS rank, % from high, volume ratio, VCP flags, criteria passed, earnings YoY, which screens it appeared in
3. Appends one JSON line to `scan_history.jsonl`
4. Re-indexes the entire history into ChromaDB `scan_history` collection

### The scan_history.jsonl Format

```json
{
  "date": "2026-07-04",
  "regime": "Bull",
  "total_scanned": 1312,
  "screen_counts": {"full_template": 165, "vcp_setup": 27, ...},
  "stocks": [
    {
      "ticker": "HFCL",
      "price": 212.09,
      "rs_rank": 99,
      "pct_from_high": -3.6,
      "vol_ratio": 0.27,
      "vcp_daily": true,
      "vcp_weekly": false,
      "passed": 8,
      "screens": ["Full Template", "Near Breakout", "VCP Setup", "CCI Daily", ...]
    },
    ...
  ]
}
```

### What Each History Document Looks Like (in ChromaDB)

```
"On 2026-07-04 (Bull market): HFCL appeared in Full Template, 
Near Breakout, VCP Setup, CCI Daily, CCI Weekly, MTF. 
RS rank: 99, Price: ₹212.09, % from 52w high: -3.6%, 
Volume: 0.27x avg (drying up), VCP detected on daily, 
Criteria passed: 8/8, EPS YoY: +316%."
```

This natural language format is what gets embedded and searched semantically.

### Queries It Enables

| Question | How It's Answered |
|----------|------------------|
| "Show me HFCL's trend over last 6 scans" | `_history_stock_trend('HFCL')` reads JSONL directly, returns date-by-date table |
| "Which stocks appeared in Full Template 3+ times?" | `_history_screen_frequency('Full Template', min_count=3)` counts across all dates |
| "Which stocks have been in CCI Daily consistently?" | Same frequency function for 'CCI Daily' |
| "What's changed this week?" | Semantic search over history + "New This Scan" data |

### Why It Gets Better Over Time

```
Day 1:  1 scan date  → basic snapshot, limited trend analysis
Day 7:  7 scan dates → can identify stocks consistently in screens for a week
Day 30: 30 dates     → meaningful patterns emerge
Day 90: 90 dates     → can spot multi-week base formations before breakouts
```

---

## 10. The AI Agent — How It Thinks

### Architecture: Context Stuffing (Not Tool Calling)

We originally built a tool-calling agent (LLM decides which tools to call, calls them, feeds results back). This failed because Groq's Llama model was generating XML-format function calls instead of JSON — a known bug.

We switched to **context stuffing**: intelligently pre-load all relevant data into the prompt based on query intent, then send one request to the LLM. This is:

- More reliable (no tool calling format dependency)
- Faster (one API call instead of 3–5)
- Cheaper (fewer tokens on failed attempts)
- Produces equally good results for our data scale

### How Query Intent is Detected

```python
q = user_message.lower()

if 'vcp' in q or 'breakout' in q:
    → Load VCP validation data for all 16 screens

if 'new' in q or 'yesterday' in q:
    → Load "New This Scan" data

if 'hfcl' in q (ticker detected):
    → Load HFCL's full detail + history trend

if 'cci' in q or 'momentum' in q:
    → Load CCI MTF screen data

if 'history' in q or 'trend' in q or 'last' in q:
    → Query scan history RAG
```

### The Prompt Structure

Every question sends this to the LLM:

```
[SYSTEM PROMPT]
You are AlphaEdge AI. You think like Minervini.
Only recommend Grade-A stocks. Cite page numbers.
Never make up tickers. Warn about visual chart verification.

[BOOK EXCERPTS — from ChromaDB]
Page 214: "VCP forms 2-4 contractions..."
Page 218: "Volume dries up at the tightest point..."

[LIVE SCAN DATA — from results.json]
MARKET REGIME: Bull
SCAN DATE: 2026-07-04
SCREEN COUNTS: Full Template: 165, VCP: 27...
MINERVINI VCP VALIDATION: 116 Grade-A from 1,234 stocks...
★ SBCL: RS:94, base:-5.0%, vol:0.57x, Score:115 [Grade-A]

[USER QUESTION]
"Which stocks show a real VCP today?"
```

### Why the LLM Never Touches ChromaDB Directly

This is a common misconception. The LLM has no database connection. It simply reads its input prompt. ChromaDB runs separately in Python, retrieves the relevant text, and that text is inserted into the prompt as plain string content. The LLM reads it the same way it reads any text.

```
ChromaDB → retrieves text → Python inserts into prompt string → LLM reads prompt → answer
```

---

## 11. The Complete Flow — Step by Step

### When You Ask a Question in the Chat Panel

```
STEP 1: You type in the chat panel
  "Which stocks show a real VCP today?"
  → Browser sends POST /api/ai/chat to server.py

STEP 2: server.py receives the request
  → Reads GROQ_API_KEY from .env file
  → Calls run_agent(message, api_key) in ai_agent.py

STEP 3: ai_agent.py — Query the book RAG
  → Converts "Which stocks show a real VCP today?" to 384 numbers
  → ChromaDB searches minervini_books collection
  → Returns 4 most relevant book passages (Page 214, 218, 220...)
  → These passages will be cited in the answer

STEP 4: ai_agent.py — Build scan context
  → Detects 'vcp' in question → triggers VCP validation
  → Reads results.json → loads all 1,234 unique stocks
  → Runs _validate_vcp_minervini() on every stock across all 16 screens
  → Scores each stock: C1-C6? RS≥80? Base tight? Volume drying? VCP flag?
  → Produces: 116 Grade-A, 89 Grade-B, X disqualified
  → Formats top 10 Grade-A with passes/fails/screens

STEP 5: ai_agent.py — Assemble the full prompt
  System prompt (Minervini reasoning rules)
  + Book excerpts (4 retrieved passages)
  + Scan context (regime, counts, Grade-A stocks with analysis)
  = ~3,000–5,000 tokens total

STEP 6: Send to Groq
  POST https://api.groq.com/openai/v1/chat/completions
  model: llama-3.3-70b-versatile
  → Groq runs the LLM on its hardware
  → ~2–4 seconds

STEP 7: LLM generates answer
  Reads book passages → understands what a real VCP requires
  Reads Grade-A stocks → sees SBCL score 115 with all criteria met
  Writes: "As Minervini explains on page 214... SBCL qualifies because
          volume ratio 0.57x is drying up, base is only -5% from high..."

STEP 8: Answer returned
  server.py → JSON response to browser
  Browser renders answer + shows tool chips (📖 Minervini book, 📅 Scan history)
```

---

## 12. Minervini VCP Validator — How the AI Validates Like Minervini

### The Problem with Just Using the VCP Screen

The scanner has a VCP screen — but it's a **starting filter**, not a final judgement. It detects price contraction patterns algorithmically but doesn't apply all of Minervini's qualitative criteria.

The AI used to just list whatever stocks were in the VCP screen. That's wrong — a stock can be flagged by the scanner but not actually qualify by Minervini's standards.

### The Validator: _validate_vcp_minervini()

This function applies Minervini's actual criteria from the book to every stock across all 16 dashboard screens (not just the VCP screen — VCPs can form in any screen):

```
For each stock, score it on 6 criteria:

1. C1–C6 Stage 2 (MANDATORY — hard disqualify if failed)
   All 6 structural criteria must be met
   Minervini never buys outside Stage 2
   +30 points if passed / DISQUALIFIED if failed

2. RS Rank
   ≥ 90: top-tier leader       +20 points
   ≥ 80: strong leader         +12 points
   ≥ 70: acceptable            +5 points
   < 70: too weak              -10 points

3. Base Tightness (% from 52-week high)
   Within 5%: breakout zone    +20 points
   Within 10%: acceptable      +10 points
   Within 15%: getting loose   +3 points
   Beyond 15%: not a VCP base  -15 points

4. Volume Contraction
   < 0.7× avg: drying up       +15 points (ideal)
   < 1.0× avg: subdued         +8 points
   1.0–1.5× avg: elevated      0 points (warning)
   > 1.5× avg: too high        -10 points (possible distribution)

5. CCI34 Momentum
   Daily + Weekly ≥ 100: MTF   +15 points (strongest signal)
   Daily ≥ 100 only            +8 points
   Weekly ≥ 100 only           +5 points
   Neither                     0 points

6. VCP Scanner Flag
   Daily + Weekly flagged      +10 points
   Daily only                  +7 points
   Weekly only                 +4 points
   No flag                     0 points (warning)

7. MA Proximity (not overextended)
   Within 5% of EMA21          +5 points (tight to trend)
   Within 8% of MA50           +3 points (pulling to support)
   More than 15% above EMA21   -8 points (overextended, risky entry)

Grade:
  Score ≥ 80: Grade-A — Minervini-quality VCP ★
  Score ≥ 60: Grade-B — Good setup, verify chart ◐
  Score ≥ 40: Grade-C — Marginal
  Score < 40: Grade-D — Does not meet criteria
```

### Why We Scan All 16 Screens

A stock forming a genuine VCP might appear in:
- Full Template (all 8 criteria met)
- MA Pullback (pulling back to a moving average while forming VCP)
- CCI Daily/Weekly (momentum confirmed)
- Near Breakout (within 5% of highs)
- F&O Uptrend (tradeable via futures)

If we only look at the VCP screen, we miss Grade-A setups sitting in other screens that happen to have perfect VCP characteristics.

Result: **1,234 unique stocks scanned → 116 Grade-A candidates identified** vs previous flat list of 15 from just the VCP screen.

### Honest Limitation

The validator uses quantitative data from `results.json`. It cannot see the actual price chart. **Minervini's full VCP analysis requires visual verification of:**
- Contraction symmetry (each swing visually smaller)
- The exact pivot point level
- Whether the base is a proper rectangle or irregular

The AI always states this limitation: *"You must visually verify the contraction symmetry before entering."*

---

## 13. Security — What Is Private, What Leaves Your Mac

### What Stays on Your Mac

| Item | Location | Never Leaves? |
|------|----------|---------------|
| Minervini book PDF | `~/Downloads/` | ✅ Never |
| Book text chunks | `rag_db/` | ✅ Never |
| Embedding vectors | `rag_db/` | ✅ Never |
| Scan data (results.json) | Project folder | ✅ Never |
| Scan history (JSONL) | Project folder | ✅ Never |
| Groq API key | `.env` file (gitignored) | ✅ Never committed |

### What Leaves Your Mac (to Groq)

Only the prompt for each question:
- Your question text
- ~4 paragraphs of book text (the retrieved passages)
- A summary of scan data (stock names, RS ranks, scores)
- Market regime

The book PDF itself never leaves. Only small excerpts relevant to your question.

### What's on GitHub

GitHub contains only the CODE (scanner logic, AI agent, dashboard). It does NOT contain:
- `.env` (API keys) — gitignored
- `results.json` (scan data) — gitignored
- `rag_db/` (vector database) — gitignored
- `scan_history.jsonl` — gitignored
- PDF books — never added

---

## 14. File Structure — What Every File Does

```
minervini-trend-template-scanner/
│
├── scanner.py              Main scanner — downloads data, applies criteria,
│                           builds all screens, saves results.json
│
├── scanner_us.py           Same for US market (S&P 500 stocks)
│
├── server.py               Local HTTP server — serves dashboard, handles
│                           POST /api/ai/chat, triggers scanner, serves results
│
├── ai_agent.py             AI brain — RAG queries, VCP validation,
│                           context building, Groq API call
│
├── build_rag.py            One-time book indexer — reads PDF, chunks,
│                           embeds, stores in ChromaDB minervini_books
│
├── build_history_rag.py    Scan history indexer — reads results.json,
│                           appends to JSONL, re-indexes ChromaDB scan_history
│                           (called automatically after every scan)
│
├── dashboard.html          Single-page app — all screens, charts, filters,
│                           floating AI chat panel (5,400+ lines)
│
├── requirements.txt        Python dependencies
│
├── .env                    GROQ_API_KEY=gsk_... (gitignored, never committed)
│
├── .gitignore              Excludes: .env, results.json, rag_db/, scan_history.jsonl
│
├── results.json            Today's scan output (1,312 stocks × all criteria)
│                           Overwritten on every scan run
│
├── scan_history.jsonl      Historical scan snapshots — one JSON line per day
│                           Grows automatically, never overwritten
│
├── prev_screens.json       Yesterday's screen membership (for "New This Scan")
│                           Updated at the START of each scan
│
├── breadth.json            Market breadth data (advance/decline, above MA200 etc.)
│
├── cci_history.json        CCI34 historical readings per stock
│
├── screen_history.json     Legacy file — currently unused
│
├── rag_meta.json           Records what's been indexed in book RAG
│
├── rag_db/                 ChromaDB vector database (15MB, gitignored)
│   ├── chroma.sqlite3      Index + metadata + text content
│   ├── d033bc24.../        minervini_books collection vectors
│   └── b255b737.../        scan_history collection vectors
│
├── exports/                CSV downloads from dashboard (gitignored)
│
└── .cache/                 Cached price data from Yahoo Finance (gitignored)
```

---

## 15. Interview Questions & Answers

### RAG & Embeddings

**Q: What is RAG and why is it better than fine-tuning for this use case?**

RAG (Retrieval Augmented Generation) retrieves relevant content from a private knowledge base at query time and injects it into the LLM's prompt. It's better than fine-tuning here because: (1) our data changes daily — the scan history grows every day and fine-tuning can't be updated in real time; (2) fine-tuning costs tens of thousands of dollars and requires curated training data; (3) RAG produces answers grounded in exact source text with citations, reducing hallucination; (4) RAG is instantly updatable — add a new book or yesterday's scan without touching the model.

---

**Q: What is an embedding and how is it created?**

An embedding is a dense vector (list of floating-point numbers) that represents the semantic meaning of text. We use `all-MiniLM-L6-v2`, a 90MB neural network trained on sentence pairs. It maps text to 384-dimensional space where semantically similar texts are geometrically close (small cosine distance). The model was trained so that "VCP" and "Volatility Contraction Pattern" produce nearby vectors even though the words are different.

---

**Q: What is cosine similarity and why use it over Euclidean distance?**

Cosine similarity measures the angle between two vectors, ranging from -1 to 1. We use cosine distance (1 - cosine similarity) as our relevance score. Cosine similarity is preferred over Euclidean distance for text embeddings because it's invariant to vector magnitude — a long document and a short document about the same topic will have similar directions even if their magnitudes differ. This makes semantic comparison more accurate than raw Euclidean distance.

---

**Q: What is chunking and why does chunk size matter?**

Chunking splits long documents into pieces that fit in the LLM's context window and are granular enough for precise retrieval. Too large: each chunk covers multiple topics — retrieved chunk is noisy, contains irrelevant content. Too small: chunks lose context, individual sentences without surrounding text lose meaning. We use ~600 characters with 100-character overlap. Overlap ensures sentences at chunk boundaries retain their surrounding context in at least one chunk.

---

**Q: What is HNSW and why does ChromaDB use it?**

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbour algorithm. It organises vectors into a multi-layer graph where similar vectors are connected. Search navigates this graph in O(log n) time rather than comparing against every stored vector (O(n) brute force). For 1,203 book chunks, brute force is fine. For millions of records, HNSW is essential. ChromaDB builds HNSW automatically when you add documents.

---

**Q: What are the failure modes of RAG?**

| Failure | Cause | Fix |
|---------|-------|-----|
| Wrong chunks retrieved | Poor chunking, weak embedding model | Better chunk strategy, stronger model |
| Retrieved but irrelevant | Chunk covers too many topics | Smaller, more focused chunks |
| LLM ignores context | Weak prompt instructions | Stronger system prompt, lower temperature |
| Hallucination despite RAG | LLM fills in gaps | Stricter prompt, require citations |
| Query-document mismatch | Different vocabulary | HyDE (generate hypothetical answer, embed that) |
| Stale information | Old documents in index | Automatic re-indexing after each scan (what we built) |

---

**Q: What is HyDE?**

Hypothetical Document Embeddings. Instead of embedding the user's raw question, ask the LLM to generate a hypothetical answer, then embed that. A hypothetical answer uses the same vocabulary and style as the actual documents, producing better retrieval. For example: query "what is a VCP?" might not match well. A hypothetical answer "A VCP consists of 2-4 contractions with volume drying up..." matches perfectly. Not implemented in our system but a meaningful upgrade.

---

### Architecture & Design

**Q: Why did you use context stuffing instead of tool calling?**

Groq's Llama model was generating XML-format function calls (`<function=name/>`) instead of the expected JSON tool call format — a bug in the model's function calling implementation. Context stuffing is also architecturally simpler: detect query intent in Python, pre-load relevant data into the prompt, one LLM call. This is faster (one round trip vs 3-5), cheaper, and more reliable. The tradeoff is that context stuffing sends more tokens per request, but for our data scale that's acceptable.

---

**Q: Why ChromaDB over Pinecone or Weaviate?**

ChromaDB runs locally with zero configuration and zero cost. Our data volume (1,203 book chunks + ~1,200 stock records growing daily) doesn't require a distributed cloud solution. Privacy is important — the book PDF and scan data never leave the machine. Pinecone would make sense at millions of records or when multiple servers need to share the index. Weaviate offers more features (multi-modal, GraphQL) but adds operational complexity we don't need.

---

**Q: How would you evaluate whether your RAG is working well?**

Three key metrics:
1. **Retrieval precision**: Are retrieved chunks actually relevant? Measure by human rating of retrieved passages for each query type
2. **Answer faithfulness**: Does the answer stay grounded in retrieved context? Can every claim be traced back to a retrieved chunk?
3. **Answer relevance**: Does the answer address what was asked?

Tools: RAGAS framework automates these metrics. Langfuse provides logging + traceability to see exactly which chunks were retrieved for each query. (Langfuse is our next feature to build.)

---

**Q: How does your system handle stocks not found in scan data?**

The VCP validator processes every stock across all 16 screens. If a specific ticker is mentioned in a question, `_history_stock_trend()` searches `scan_history.jsonl` directly by ticker name. If the stock hasn't appeared in any scan, the AI is instructed to say so clearly rather than guessing — "XYZ not found in any screen in today's scan."

---

**Q: What happens if the book PDF is updated or a new book is added?**

Re-run `build_rag.py`. It deletes the existing `minervini_books` collection and rebuilds from scratch. Add the new book path to the `BOOKS` list in `build_rag.py`. The process takes ~60 seconds. The `rag_meta.json` is updated to reflect the new index.

---

**Q: How does the "New This Scan" feature work technically?**

At the START of each scan (before any new data is written), the scanner reads the old `results.json`, extracts stock lists for each tracked screen, and saves them to `prev_screens.json` — but ONLY if the date in `results.json` is different from today. Same-day re-runs preserve the previous day's baseline. After the scan completes, for each tracked screen it computes `current_set - previous_set` = new entrants. These are collected, enriched with full stock data, and stored as the `newly_added` screen in `results.json`.

---

### LLM & AI

**Q: What is a token in the context of LLMs?**

Tokens are the basic units of text that LLMs process. Roughly 1 word ≈ 1.3 tokens in English. LLMs have a context window (max tokens per request) and a cost model based on tokens processed. Our typical question + book excerpts + scan data = 3,000–5,000 tokens. The Groq free tier allows 131,072 tokens/day, supporting ~26–40 questions per day at this rate.

---

**Q: What is temperature in LLM generation?**

Temperature controls randomness in the LLM's output. Temperature = 0: always picks the most likely next token (deterministic, factual). Temperature = 1: samples proportionally (more creative, more variable). We use temperature = 0.2 — low enough for consistent, factual answers about stock setups, high enough to avoid mechanical repetition.

---

**Q: Why Groq instead of OpenAI or Anthropic?**

Free tier with no credit card required. For a personal portfolio project running ~5–10 questions per day, the free tier (131K tokens/day) is more than sufficient. OpenAI charges per token from first use. Anthropic's Claude API also requires payment. Groq's speed (LPU hardware) also means faster responses. The tradeoff: Groq's Llama model is slightly less capable than GPT-4o or Claude Opus for complex reasoning, but for structured stock analysis with pre-loaded context it performs well.

---

## 16. Agentic Workflow — Gandiva Morning Brief & Telegram Bot

### What Is an Agentic Workflow?

A regular AI answers a question when you ask it. An **agentic workflow** is an AI that acts autonomously — it decides what data to gather, runs multiple steps in sequence, produces a structured output, and delivers it somewhere useful — all without you doing anything.

Our agentic pipeline runs every evening and delivers a trading brief to your phone.

---

### The Gandiva Bot

**Name origin:** Gandiva is Arjuna's divine bow from the Mahabharata — it never missed its target. Named to represent precision, speed, and power in identifying stock setups.

- **Telegram handle:** @GandivaScannerBot
- **Bot ID:** 8861126690
- **Platform:** Telegram Bot API (free, no server required)

---

### What is Telegram Bot API?

Telegram provides a free HTTP API that lets any program send messages to any Telegram user. You don't need to host a server — you just make HTTP POST requests to:

```
https://api.telegram.org/bot{TOKEN}/sendMessage
```

with `chat_id` (who to send to) and `text` (what to send). Telegram delivers it instantly to the user's phone.

**Two things needed:**
1. `BOT_TOKEN` — given by BotFather when you create the bot
2. `CHAT_ID` — the user's unique Telegram ID, captured by polling for a `/start` message

---

### Registration Flow (One-Time Setup)

```
python3 telegram_bot.py
    ↓
Script polls getUpdates API every 2 seconds
    ↓
User sends /start to @GandivaScannerBot in Telegram
    ↓
Script captures chat_id from the update
    ↓
Saves to telegram_chat_id.json  (permanent, never expires)
    ↓
Sends confirmation message to user
```

After this one-time step, the bot knows where to send every future alert.

---

### Morning Brief Pipeline — Step by Step

**File:** `morning_brief.py`

```
Step 1: Load results.json
        → collect all unique stocks across all 16 screens
        → note which screens each stock appears in

Step 2: Load regime data
        → market label (BULL/CAUTION/BEAR)
        → % stocks above 200MA
        → breadth reading

Step 3: Build context for AI
        → top 20 candidates with RS rating, relative volume, screen count
        → regime summary
        → date and trading context

Step 4: Call run_agent()
        → context stuffing kicks in (book RAG + history RAG + live data)
        → LLM writes structured brief:
            - Regime interpretation + position sizing guidance
            - Top 3-5 Grade-A VCP setups with entry/stop/target
            - High-confluence stocks (appearing in 3+ screens)
            - Risk management reminder
            - Minervini quote relevant to current conditions

Step 5: Save HTML brief
        → exports/Briefs/brief_YYYY-MM-DD.html
        → dark-themed, table of all candidates, screen coverage

Step 6: Send Telegram alert
        → split into 4,000-char chunks (Telegram limit)
        → sends each chunk sequentially
        → falls back to plain text if Markdown formatting fails
```

---

### High-Confluence Stocks — Why They Matter

A stock appearing in 3+ screens simultaneously is a strong signal:

| Screens | Meaning |
|---------|---------|
| Full Template | Passes all 8 Minervini criteria |
| VCP Setup | Volatility contraction pattern detected |
| RS Leaders | Top relative strength in the market |
| Near Breakout | Price within 5% of pivot |

If DIXON appears in all four → it's passing every filter independently. Each screen is a separate algorithm. Agreement = high conviction.

---

### Telegram Markdown — Why Fallback Exists

Telegram supports a limited subset of Markdown. Characters like `_`, `*`, `[`, `]`, `.` have special meaning. If the AI generates text with unmatched or misplaced symbols, Telegram returns HTTP 400 Bad Request.

**Fix:** `send_message()` catches the 400 error, strips all Markdown symbols from the text, and retries as plain text. The message always arrives — formatting is best-effort.

---

### Files Involved

| File | Role |
|------|------|
| `telegram_bot.py` | Bot API wrapper — send_message, register_via_polling, load/save chat_id |
| `morning_brief.py` | Full agentic pipeline — scan → analyze → format → deliver |
| `telegram_chat_id.json` | Persisted chat_id after one-time /start registration |
| `exports/Briefs/` | HTML brief archive, one file per day |
| `.env` | Stores TELEGRAM_BOT_TOKEN (never committed to git) |

---

### Interview Questions

**Q: What is an agentic workflow?**

An agentic workflow is a pipeline where an AI autonomously decides what tools to use, sequences multiple steps, and produces a structured output without human intervention at each step. Ours follows a ReAct-style pattern: gather context (scan data + book RAG + history RAG) → reason (LLM analysis) → act (send Telegram). The key distinction from a chatbot is autonomy — it runs on a schedule, not on demand.

**Q: What is the Telegram Bot API and how does it work?**

Telegram's Bot API is a free HTTP interface that lets programs send messages to Telegram users. You create a bot via @BotFather (gets you a token), then make POST requests to `api.telegram.org/bot{token}/sendMessage` with the recipient's chat_id. No server hosting required — Telegram's infrastructure handles delivery. The main limitation is a 4,096 character message limit and restricted Markdown syntax.

**Q: How do you capture a user's chat_id without them knowing their own ID?**

Poll the `getUpdates` endpoint after asking the user to send `/start` to the bot. The update object contains the sender's `chat` field with their `id`. This is a standard bot onboarding pattern — you register once, persist the chat_id, and all future sends use that stored value.

**Q: What happens if the Telegram message fails?**

Our `send_message()` function has a two-level fallback: first attempt with Markdown formatting, catch any HTTP error, strip all Markdown symbols, retry as plain text. This ensures the message always arrives even if formatting is broken. The HTML brief is also saved locally as a permanent record regardless of whether Telegram delivery succeeded.

---

## 17. AI Observability — Langfuse

### What is Observability?

In software engineering, observability means being able to see inside a running system — what it did, how long it took, where it failed. For a web server, that's request logs. For an AI system, it means tracing every LLM call from input to output.

Without observability, you're flying blind:
- Is the RAG retrieving relevant chunks or random noise?
- Is the LLM staying grounded in the book or hallucinating?
- Which questions use the most tokens (cost)?
- Is latency getting worse over time?

**Langfuse** is the leading open-source LLM observability platform. Used in production by teams at major AI companies.

---

### What We Track Per Query

Every time someone asks Gandiva a question, Langfuse records a full **trace** with nested **spans**:

```
Trace: "gandiva-query"  (the full request)
├── Span: "book-rag"
│     Input:  query text
│     Output: 4 chunks found, pages [89, 134, 218, 301], scores [0.42, 0.51, 0.63, 0.71]
│     Duration: 180ms
│
├── Generation: "llm-response"
│     Input:  full messages array (system prompt + context + user question)
│     Output: clean answer text
│     Model:  llama-3.3-70b-versatile
│     Tokens: 3,840 input / 420 output
│     Duration: 1,240ms
│
└── Trace output:
      answer_length: 1,820 chars
      tools_used: [query_minervini_book, get_scan_summary, filter_stocks]
      trade_cards: 2
      latency_sec: 1.42
```

---

### How It Works — Architecture

```
User asks question in dashboard
    ↓
run_agent_stream() starts
    ↓
_get_langfuse() → checks .env for LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
    ↓ (if keys present)
lf.trace() → creates trace object in memory
    ↓
RAG runs → trace.span("book-rag") records what was retrieved
    ↓
LLM runs → trace.generation("llm-response") records prompt + output + token usage
    ↓
Trade cards extracted → recorded in trace output
    ↓
lf.flush() → sends all trace data to Langfuse cloud (async, non-blocking)
    ↓
Langfuse web dashboard → you see the full trace
```

**Graceful degradation:** If `LANGFUSE_PUBLIC_KEY` is not in `.env`, `_get_langfuse()` returns `None` and every `if trace:` block is skipped. Zero performance impact, zero errors.

---

### Key Concepts

**Trace:** One complete request/response cycle — the top-level container.

**Span:** A sub-step within a trace with its own start/end time — used for RAG retrieval.

**Generation:** A special span specifically for LLM calls — Langfuse understands token counts, model names, and cost estimation automatically.

**Flush:** Langfuse batches data in memory and sends it asynchronously. `lf.flush()` forces an immediate send — important to call after each query so data isn't lost if the server restarts.

---

### What You See in the Langfuse Dashboard

- **Traces list**: every question asked, when, how long it took
- **Trace detail**: drill into any query to see exactly what RAG retrieved and what the LLM responded
- **Token usage**: total tokens per day/week, broken down by model
- **Latency charts**: p50/p95 response times over time
- **Score tracking**: you can manually rate responses as good/bad → builds a quality dataset
- **Cost estimation**: Langfuse estimates API cost based on token usage and model pricing

---

### Setup (One-Time)

1. Go to **cloud.langfuse.com** → sign up free (no credit card)
2. Create a project called "Gandiva"
3. Go to Settings → API Keys → copy Public Key + Secret Key
4. Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
5. Restart `server.py` — tracing activates automatically

---

### Interview Questions

**Q: What is LLM observability and why does it matter?**

LLM observability is the ability to inspect what an AI system did at every step of a request — what data it retrieved, what prompt it sent, what the model responded, how many tokens it used, and how long it took. It matters because LLMs are non-deterministic black boxes: the same question can get different answers depending on what RAG retrieved. Without observability, you can't tell if the system is working correctly, degrading over time, or hallucinating. It's the difference between a hobby project and a production system.

**Q: What is the difference between a trace, span, and generation in Langfuse?**

A trace is the top-level container for one complete user request. Spans are nested steps within that trace — each with their own start/end time, input, and output. A generation is a special type of span specifically for LLM calls — Langfuse gives it extra treatment: it understands token counts, cost estimation, and model versioning. Our trace has one RAG span (what the book search returned) and one generation (what the LLM said).

**Q: How did you make Langfuse optional without breaking the existing system?**

The `_get_langfuse()` function checks for API keys in the environment. If they're not present, it returns `None`. Every tracing call is guarded by `if trace:` — so if `_get_langfuse()` returns `None`, every tracing block is a no-op. The `run_agent` and `run_agent_stream` functions work identically with or without Langfuse. This is called graceful degradation — the feature enhances the system without creating a hard dependency.

**Q: How would you use Langfuse to evaluate RAG quality?**

Retrieve the traces for a sample of questions. For each trace, look at the RAG span: what chunks were retrieved and what were their similarity scores? Then look at the LLM generation: does the answer cite information from the retrieved chunks, or is it making things up? Langfuse's scoring feature lets you annotate traces as good/bad. Over time, you build a labelled dataset you can use to measure retrieval precision and answer faithfulness — the two key RAG quality metrics. You can also run automated evals using RAGAS framework and log the scores back to Langfuse.

---

## 18. Daily Scheduler — How Automation Works

### The Problem

Python scripts don't run themselves. For Gandiva to send a brief every evening at 8 PM, something needs to trigger `daily_run.py` at that exact time — whether the user is at their desk or not.

### macOS launchd — The Right Tool for Mac

**launchd** is macOS's system daemon manager — the same infrastructure that starts Bluetooth, Wi-Fi, and every background Apple service. It is more reliable than `cron` on Mac because:
- Survives reboots (automatically re-registers on login)
- Handles sleep/wake correctly (fires as soon as Mac wakes if it missed the scheduled time)
- Integrated with the OS — Apple's recommended scheduler

**cron** on Mac has known issues: it doesn't run reliably when the Mac is sleeping, requires full disk access permissions separately, and can be killed silently by macOS security policies.

---

### How launchd Works

A **plist** (Property List) file defines the job — it's an XML config file that tells launchd:
- What program to run
- When to run it
- Where to write logs

```xml
<key>StartCalendarInterval</key>
<dict>
  <key>Hour</key>   <integer>20</integer>   <!-- 8 PM -->
  <key>Minute</key> <integer>0</integer>
</dict>
```

The plist lives at `~/Library/LaunchAgents/com.gandiva.scanner.plist`. LaunchAgents run as the current user (not root) — appropriate for user-level tasks like our scanner.

---

### Registration Commands

```bash
# Register the job (run once)
launchctl load ~/Library/LaunchAgents/com.gandiva.scanner.plist

# Check it's registered
launchctl list | grep gandiva

# Unregister (stop scheduling)
launchctl unload ~/Library/LaunchAgents/com.gandiva.scanner.plist

# Trigger manually for testing
launchctl start com.gandiva.scanner
```

---

### daily_run.py — What It Does

```
8:00 PM — launchd wakes and calls daily_run.py
    ↓
Step 1: Run scanner.py (subprocess, 30 min max timeout)
        → fetches live NSE data
        → updates all 16 screens
        → updates prev_screens.json (resets "New This Scan" baseline)
        → updates scan_history.jsonl (history RAG grows)
        → writes results.json
    ↓
Step 2: (only if scanner succeeded) Run morning_brief.py
        → AI analysis of today's results
        → HTML brief saved to exports/Briefs/
        → Telegram alert sent to @GandivaScannerBot
    ↓
All output logged to logs/daily_run.log with timestamps
```

---

### What "New This Scan" Shows After Daily Scheduling

Since the scanner runs every day at 8 PM, `prev_screens.json` is updated daily. This means:

- **"New This Scan" = "New Since Yesterday 8 PM"** — exactly a 24-hour window
- If your Mac was off for 3 days and you open it: launchd fires once, scanner runs, compares against whatever `prev_screens.json` contains (from 3 days ago) — so you see everything new in the last 3 days

**The one blind spot:** stocks that briefly entered and exited a screen during the gap (e.g., appeared Monday, disappeared Tuesday) are invisible because the scanner only captures the current snapshot vs the last snapshot. This is inherent to point-in-time comparison, not a bug.

---

### Option A vs Option B — Scheduling Approaches

**Option A (current implementation — local launchd):**
- Runs on your Mac at 8 PM
- Requires Mac to be on/awake
- "New This Scan" works perfectly — same machine, same files
- If Mac sleeps: runs when Mac wakes up
- If Mac off 3 days: runs once on open, covers full gap

**Option B (cloud — GitHub Actions):**
- Runs on GitHub's servers at 8 PM regardless of Mac state
- Telegram alert is 100% reliable
- "New This Scan" on local dashboard breaks (results.json is on GitHub, not Mac)
- Fix: dashboard fetches results.json from GitHub URL on load (auto-sync)
- Best for: reliable daily Telegram alert even if Mac is rarely on

**Current choice: Option A** — simpler, everything stays local, dashboard always in sync. Upgrade to Option B if reliable Telegram delivery becomes the priority.

---

### Log Files

| File | Contents |
|------|----------|
| `logs/daily_run.log` | Timestamped log of every step — scanner output, brief generation, Telegram status |
| `logs/launchd_out.log` | Raw stdout from launchd (mirrors daily_run.log) |
| `logs/launchd_err.log` | Any stderr/crashes from the scheduled run |

---

### Interview Questions

**Q: How did you schedule the pipeline to run automatically?**

Used macOS launchd — the OS-level daemon manager — rather than cron, because launchd is more reliable on Mac: it survives reboots, handles sleep/wake gracefully, and is the Apple-recommended approach. Defined a plist file under `~/Library/LaunchAgents/` with `StartCalendarInterval` set to 8 PM. The plist calls `daily_run.py` which runs scanner.py first and morning_brief.py second, with full logging to timestamped log files.

**Q: What happens if the scanner crashes — does the brief still run?**

No. `daily_run.py` checks the return code of the scanner subprocess. If it fails (non-zero exit code or timeout), it logs the error and skips the brief entirely. Sending an AI brief based on stale data would be misleading. The brief only runs when the scanner succeeds and `results.json` is fresh.

**Q: Why not just use cron?**

cron on macOS has known reliability issues: it doesn't reliably wake the machine from sleep, requires separate Full Disk Access permissions in System Preferences, and can be silently disabled by macOS security policies (Gatekeeper, SIP). launchd is the OS's own scheduler and doesn't have these problems. On Linux servers, cron is perfectly fine — but on Mac desktop, launchd is the correct tool.

**Q: How would you move this to production for 1,000 users?**

Replace launchd with a cloud scheduler (GitHub Actions, AWS EventBridge, or a cron job on a Linux VPS). Store results.json in S3 or a database instead of local file. The dashboard fetches from an API endpoint instead of localhost. Each user has their own chat_id stored in a database. The morning_brief pipeline becomes a cloud function (AWS Lambda or similar) triggered by the scheduler. The architecture is the same — the hosting layer changes.

---

## 19. MCP Server — Model Context Protocol

### What is MCP?

MCP (Model Context Protocol) is an open standard released by Anthropic in late 2024. It defines a universal way for AI models to connect to external tools and data sources — like a USB port for AI. Instead of every AI app building its own custom integrations, MCP provides one standard protocol any AI client can speak.

### What We Built

**File:** `mcp_server.py`

Six tools exposed to any MCP-compatible AI client:

| Tool | What it does |
|------|-------------|
| `get_market_regime` | Current NSE regime, breadth %, stocks above MA200 |
| `get_screen` | All stocks from any of 16 scanner screens |
| `get_top_setups` | Grade-A VCPs ranked by multi-screen confluence |
| `validate_stock` | Full Minervini score breakdown for any ticker |
| `query_minervini_book` | Semantic search over "Trade Like a Stock Market Wizard" |
| `get_scan_summary` | Full overview: counts per screen, regime, scan time |

### How It Works

```
Claude Desktop (or any MCP client)
    ↓ speaks MCP protocol over stdio
mcp_server.py (running on your Mac)
    ↓ reads
results.json + rag_db/ + scan_history.jsonl
    ↓ returns structured data
Claude Desktop formats and presents the answer
```

The MCP server runs as a background process managed by Claude Desktop. When you open Claude Desktop, it starts `mcp_server.py` automatically. When you close Claude Desktop, it stops.

### Real Example — What Happened

User asked: *"What are the best VCP setups in my NSE scanner right now?"*

Claude Desktop autonomously:
1. Called `get_market_regime()` → Strong Bull, 1,358 stocks
2. Called `get_top_setups(min_rs=80)` → ranked HFCL, BAJAJCON, GRWRHITECH...
3. Called `validate_stock("HFCL")` → entry ₹214.48, stop ₹197.32, target ₹257.38
4. Called `validate_stock("BAJAJCON")` → entry ₹616.20, stop ₹566.90, target ₹739.44
5. Applied user's risk rule (0.75% of ₹30L = ₹22,500 risk → ₹2.8L position size)

**4 tool calls. One question. Zero dashboard interactions.**

### Claude Desktop Configuration

File: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "gandiva": {
      "command": "/opt/homebrew/bin/python3.10",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

### Limitations (Current)

- Mac must be on for MCP to work (local stdio transport)
- Remote MCP (always-on cloud) requires hosting on a VPS
- Data is only as fresh as the last scan run

### Interview Questions

**Q: What is MCP and why does it matter?**

MCP is Anthropic's open standard for connecting AI models to external tools and data. Before MCP, every AI application built its own proprietary tool integration — Claude had one format, GPT had another, each IDE extension was different. MCP defines one standard protocol so any AI client can connect to any MCP server without custom integration work. It's architecturally significant because it decouples the AI (client) from the tools (server) — you can swap the AI model without rewriting the tools, and your tools work across any MCP-compatible AI.

**Q: What's the difference between MCP tools and regular API calls?**

In a regular API, you write code that explicitly calls specific endpoints. With MCP, the AI decides autonomously which tools to call and in what order based on the user's question. You don't write `if user_says_setup then call get_top_setups()` — the AI figures that out from the tool descriptions. This is the distinction between scripted automation and genuine agentic behaviour.

**Q: How did you expose your scanner as an MCP server?**

Used the FastMCP framework from the MCP Python SDK. Decorated each function with `@mcp.tool()` — the decorator registers the function as an MCP tool, uses the docstring as the tool description (what the AI reads to decide when to call it), and handles all the protocol serialization automatically. The server runs via `mcp.run(transport='stdio')` — Claude Desktop starts it as a subprocess and communicates over stdin/stdout.

**Q: How would you scale MCP to serve multiple users?**

Switch from stdio transport to HTTP/SSE transport (MCP supports both). Host the server on a VPS with a proper web framework (FastAPI). Add authentication — each user gets an API key. Store results.json per-user in a database instead of a local file. The tool logic stays identical — only the transport and storage layer changes. This is the production path for any MCP server.

---

## 20. Ollama — Local LLM Fallback

### What is Ollama?

Ollama is an open-source tool that lets you run large language models entirely on your own hardware — no internet, no API key, no token limits, no cost. It downloads model weights to your Mac and runs inference on the Apple Silicon chip.

### How We Integrated It

**The priority chain in `_get_client(api_key)`:**

```python
if api_key:           → use Groq cloud (llama-3.3-70b, fast, powerful)
else:                 → use Ollama local (llama3.2, slower, free forever)
```

This is automatic — no code change needed. Remove the Groq key from `.env` and every AI call in the system (dashboard chat, morning brief, MCP tools) switches to Ollama instantly.

### Models Installed

| Model | Size | Speed | Best for |
|-------|------|-------|---------|
| llama3.2 (3B) | 2GB | ~15-30s/query | Fallback, simple questions |

### Why Groq for Now, Ollama for Insurance

The 70B vs 3B model size difference is significant for our use case:
- Trade card JSON output requires precise instruction following → 70B does this reliably
- Minervini VCP validation with 7 criteria → 70B reasons correctly
- Book RAG with page citations → 70B cites accurately

The 3B model handles simple regime questions well but can struggle with structured JSON output. For production-quality trade analysis, Groq's 70B is the right choice while it remains free.

**Ollama activates when:**
- Groq key is removed from `.env`
- Groq API is unreachable
- You're fully offline (travelling, no internet)

### Useful Commands

```bash
# Start Ollama service
brew services start ollama

# List installed models
ollama list

# Pull a better model (8B — better quality, needs 5GB RAM)
ollama pull llama3.1:8b

# Test directly
ollama run llama3.2 "What is a VCP setup?"

# Check Ollama API
curl http://localhost:11434/api/tags
```

### Interview Questions

**Q: Why did you use the OpenAI SDK to talk to Ollama?**

Ollama exposes an OpenAI-compatible REST API at `localhost:11434/v1`. By using `OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')`, we get identical code for both cloud and local inference — the same `client.chat.completions.create()` call works for both Groq and Ollama. This is the standard pattern for LLM provider abstraction: program to the OpenAI interface, swap the base_url for different backends.

**Q: What are the trade-offs between a 3B and 70B model for this use case?**

Parameter count roughly correlates with reasoning capability and instruction following. Our system asks the model to: extract structured JSON trade cards, apply 7 Minervini criteria consistently, cite page numbers from retrieved book passages, and reason about market regimes. A 70B model handles all of these reliably. A 3B model handles simple factual questions but often deviates from the JSON schema or misses nuanced criteria. For a trading system where a wrong stop loss calculation has real financial consequences, model quality is not a place to cut corners.

---

## 21. Structured Trade Cards

### What They Are

When the AI recommends a stock, it outputs two things simultaneously:
1. **Prose analysis** — reasoning in Minervini's style, book citations, risk warnings
2. **JSON trade card** — machine-readable structured data with exact entry/stop/target

The JSON card is extracted, stripped from the visible chat, and rendered as a visual card below the AI response.

### JSON Schema

```json
{
  "ticker": "HFCL",
  "grade": "A",
  "pattern": "VCP",
  "entry_low": 214.48,
  "entry_high": 218.77,
  "stop_loss": 197.32,
  "stop_pct": 8.0,
  "target": 257.38,
  "target_pct": 20.0,
  "risk_reward": 2.5,
  "timeframe": "4-8 weeks",
  "reasoning": "Elite RS 99, vol ratio 0.34, -2.5% from 52wk high, all 8 criteria passed"
}
```

### How Prices Are Computed

The scanner stores each stock's `price` field (last traded price). The system prompt instructs the LLM:
- `entry_low` = current price
- `entry_high` = price × 1.02 (2% above for confirmation)
- `stop_loss` = entry_low × (1 − stop_pct/100)
- `target` = entry_low × (1 + target_pct/100)
- `risk_reward` = target_pct / stop_pct

### How Streaming Handles the JSON Block

The JSON block appears at the end of the LLM response. During streaming:
1. Text chunks are forwarded to the browser as they arrive
2. When `\`\`\`json` is detected in the accumulated text, chunk forwarding stops
3. The rest of the stream (JSON content) is buffered silently
4. After streaming ends, `_extract_trade_cards()` parses the JSON
5. A `{'type': 'cards', 'cards': [...]}` event is emitted
6. The browser renders the visual trade cards

Result: user sees clean prose appearing word by word, then trade cards appear at the end. No raw JSON ever shows in the chat.

---

## 22. How to Run — Complete Reference

### Starting the System

```bash
# 1. Start the dashboard server
/opt/homebrew/bin/python3.10 server.py

# 2. Open dashboard
open http://localhost:8765

# 3. Start Langfuse observability (optional)
cd langfuse && docker compose up -d
open http://localhost:3000
```

### Daily Scan (Automated)
Runs automatically at 8 PM via launchd. To trigger manually:
```bash
/opt/homebrew/bin/python3.10 daily_run.py
```

### Morning Brief (Manual)
```bash
/opt/homebrew/bin/python3.10 morning_brief.py
# or click "🏹 Brief" button in dashboard
```

### Telegram Registration (One-Time)
```bash
/opt/homebrew/bin/python3.10 telegram_bot.py
# Send /start to @GandivaScannerBot in Telegram within 2 minutes
```

### Build/Rebuild RAG Indexes
```bash
# Book RAG (run once, or after adding new books)
/opt/homebrew/bin/python3.10 build_rag.py

# History RAG (auto-runs after each scan — manual rebuild if needed)
/opt/homebrew/bin/python3.10 build_history_rag.py
```

### MCP Server (Auto-managed by Claude Desktop)
```bash
# Test tools directly
/opt/homebrew/bin/python3.10 mcp_server.py

# Claude Desktop config: ~/Library/Application Support/Claude/claude_desktop_config.json
```

### Scheduler Management
```bash
# Check scheduler status
launchctl list | grep gandiva

# Stop auto-scan
launchctl unload ~/Library/LaunchAgents/com.gandiva.scanner.plist

# Re-enable
launchctl load ~/Library/LaunchAgents/com.gandiva.scanner.plist

# View logs
tail -f logs/daily_run.log
```

### Environment Variables (.env)
```
GROQ_API_KEY=gsk_...           # Groq cloud LLM (free tier)
TELEGRAM_BOT_TOKEN=...         # @GandivaScannerBot token
LANGFUSE_PUBLIC_KEY=pk-lf-...  # Langfuse observability
LANGFUSE_SECRET_KEY=sk-lf-...  # Langfuse observability
LANGFUSE_HOST=http://localhost:3000  # Self-hosted Langfuse
```

---

## 23. Portfolio Pitch — How to Present This Project

### One-Line Summary
> *"An AI-powered NSE stock scanner that identifies Minervini VCP setups, validates them against his published methodology using RAG, delivers daily trade briefs via Telegram, and exposes live market data to any AI client via MCP — running entirely on a local Mac with zero ongoing cost."*

### What Makes It Technically Impressive

1. **RAG over a domain-specific book** — not just generic LLM, but grounded in Minervini's actual methodology with page citations
2. **Self-hosted vector database** — ChromaDB running locally, 1,203 book chunks + growing scan history
3. **Agentic pipeline** — morning brief chains 5 steps autonomously: scan → validate → analyse → format → deliver
4. **MCP server** — scanner exposed as a standardised AI tool, live-demonstrated with Claude Desktop calling real NSE data
5. **Production observability** — every AI call traced in self-hosted Langfuse with latency, token usage, RAG retrieval quality
6. **LLM provider abstraction** — Groq cloud primary, Ollama local fallback, zero code change to switch
7. **Streaming UI** — SSE-based word-by-word streaming with structured trade cards appearing after prose

### Costs
- **Monthly cost: ₹0** (Groq free tier, self-hosted Langfuse, local Ollama fallback)
- **Worst case if Groq charges: ~₹600/year** (can eliminate entirely with Ollama)

### Tech Stack (Interview Cheat Sheet)

| Layer | Technology | Why |
|-------|-----------|-----|
| Data | yfinance + NSE | Free real-time Indian market data |
| Storage | JSON files + JSONL | Simple, zero infrastructure |
| Vector DB | ChromaDB (local) | Free, no server needed, HNSW index |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, runs on CPU, 90MB |
| LLM (cloud) | Groq / Llama 3.3 70B | Free tier, fast LPU hardware |
| LLM (local) | Ollama / Llama 3.2 3B | Offline fallback, zero cost |
| Streaming | SSE (Server-Sent Events) | Browser-native, no WebSocket needed |
| Observability | Langfuse v2 (Docker) | Open-source, self-hosted, free |
| Automation | macOS launchd | Reliable Mac scheduler, survives reboots |
| AI Protocol | MCP (Model Context Protocol) | Anthropic's 2024 open standard |
| Alerts | Telegram Bot API | Free, instant, no server needed |
| UI | Vanilla HTML/JS | Zero framework dependencies |

---

## 24. Features Built — Status Tracker

| # | Feature | Status | Files |
|---|---------|--------|-------|
| ✅ | NSE Scanner (25+ screens) | Complete | scanner.py |
| ✅ | Dashboard (all screens, filters, CSV export) | Complete | dashboard.html |
| ✅ | Local server with Scan Now | Complete | server.py |
| ✅ | New This Scan (daily diff) | Complete | scanner.py, dashboard.html |
| ✅ | AI Chat Panel (floating) | Complete | dashboard.html, server.py |
| ✅ | Book RAG (Trade Like a Stock Market Wizard) | Complete | build_rag.py, ai_agent.py |
| ✅ | Scan History RAG (temporal queries) | Complete | build_history_rag.py, ai_agent.py |
| ✅ | Minervini VCP Validator (all 16 screens) | Complete | ai_agent.py |
| ✅ | Streaming Responses (SSE, word by word) | Complete | ai_agent.py, server.py, dashboard.html |
| ✅ | Agentic Morning Brief (ReAct pipeline) | Complete | morning_brief.py |
| ✅ | Telegram Bot (Gandiva @GandivaScannerBot) | Complete | telegram_bot.py |
| ✅ | Daily Scheduler (launchd, 8 PM) | Complete | daily_run.py, com.gandiva.scanner.plist |
| ✅ | Structured Trade Cards (JSON output) | Complete | ai_agent.py, dashboard.html |
| ✅ | AI Observability (self-hosted Langfuse + Docker) | Complete | ai_agent.py, langfuse/ |
| ✅ | MCP Server (Model Context Protocol, 6 tools) | Complete | mcp_server.py |
| ✅ | Local LLM Fallback (Ollama, zero cost) | Complete | ai_agent.py |
| ⏳ | Multimodal Chart Vision | Pending | needs GPT-4o |

---

*Last updated: 2026-07-05*
