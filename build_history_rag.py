"""
build_history_rag.py — Index scan history into ChromaDB for temporal queries.

Run after every scan (or manually):
  /opt/homebrew/bin/python3.10 build_history_rag.py

Reads:  scan_history.jsonl  (one JSON line per scan date)
Writes: rag_db/  (ChromaDB collection 'scan_history')

Enables AI queries like:
  - "Show me HFCL's trend over the last 6 scans"
  - "Which stocks appeared in Full Template 3+ times this month?"
  - "What stocks have been in CCI daily for 5+ consecutive days?"
  - "Which stocks improved their RS rank the most this week?"
"""
import json, os, sys
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_JSONL = os.path.join(BASE_DIR, 'scan_history.jsonl')
RAG_DB        = os.path.join(BASE_DIR, 'rag_db')
RESULTS_JSON  = os.path.join(BASE_DIR, 'results.json')

# Screens to track in history (compact — not all 25+)
TRACKED_SCREENS = [
    'full_template', 'near_breakout', 'vcp_setup', 'c1_to_c6',
    'watch_list', 'fo_strong_uptrend', 'ma_pullback', 'hhhl_pullback',
    'ck_daily_100', 'ck_weekly_100', 'ck_mtf_100', 'cci34_best_setups',
    'rs_leaders', 'strong_earnings', 'new_highs', 'newly_added',
]

SCREEN_SHORT = {
    'full_template':      'Full Template',
    'near_breakout':      'Near Breakout',
    'vcp_setup':          'VCP Setup',
    'c1_to_c6':           'C1-C6',
    'watch_list':         'Watch List',
    'fo_strong_uptrend':  'F&O Uptrend',
    'ma_pullback':        'MA Pullback',
    'hhhl_pullback':      'HH/HL Pullback',
    'ck_daily_100':       'CCI Daily',
    'ck_weekly_100':      'CCI Weekly',
    'ck_mtf_100':         'CCI MTF',
    'cci34_best_setups':  'CCI Best',
    'rs_leaders':         'RS Leaders',
    'strong_earnings':    'Strong Earnings',
    'new_highs':          'New Highs',
    'newly_added':        'New This Scan',
}


# ── Snapshot builder ──────────────────────────────────────────────────────────
def take_snapshot(results_path: str = RESULTS_JSON) -> dict | None:
    """Read results.json and return a compact scan snapshot dict."""
    if not os.path.exists(results_path):
        return None
    import re
    with open(results_path) as f:
        raw = f.read()
    raw = re.sub(r'\bNaN\b', 'null', raw)
    raw = re.sub(r'\bInfinity\b', 'null', raw)
    data    = json.loads(raw)
    screens = data.get('screens', {})
    ma      = data.get('market_analysis') or {}
    date    = (data.get('generated_at') or '')[:10]
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    # Build per-stock screen membership
    stock_data: dict[str, dict] = {}
    for scr_key in TRACKED_SCREENS:
        for s in screens.get(scr_key, {}).get('stocks', []):
            tk = (s.get('ticker') or '').replace('.NS', '').upper()
            if not tk:
                continue
            if tk not in stock_data:
                stock_data[tk] = {
                    'ticker':        tk,
                    'price':         s.get('price'),
                    'rs_rank':       s.get('rs_rank'),
                    'pct_from_high': s.get('pct_from_high'),
                    'vol_ratio':     s.get('vol_ratio'),
                    'passed':        s.get('passed'),
                    'vcp_daily':     bool(s.get('vcp_last_daily')),
                    'vcp_weekly':    bool(s.get('vcp_last_weekly')),
                    'pct_from_ma50': s.get('pct_from_ma50'),
                    'q_eps_yoy':     s.get('q_eps_yoy'),
                    'screens':       [],
                }
            stock_data[tk]['screens'].append(SCREEN_SHORT.get(scr_key, scr_key))

    return {
        'date':          date,
        'regime':        ma.get('regime', 'Unknown'),
        'total_scanned': data.get('scanned', 0),
        'screen_counts': {k: len(screens.get(k, {}).get('stocks', [])) for k in TRACKED_SCREENS},
        'stocks':        list(stock_data.values()),
    }


def append_snapshot(snap: dict, path: str = HISTORY_JSONL):
    """Append snapshot to JSONL, replacing any existing entry for same date."""
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get('date') != snap['date']:
                            existing.append(line)
                    except Exception:
                        pass
    existing.append(json.dumps(snap, separators=(',', ':')))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(existing) + '\n')
    os.replace(tmp, path)


# ── Document builder ──────────────────────────────────────────────────────────
def _stock_to_doc(date: str, regime: str, s: dict) -> str:
    """Convert one stock-on-one-date into a searchable text document."""
    scrs   = ', '.join(s.get('screens', []))
    vcp    = []
    if s.get('vcp_daily'):  vcp.append('daily')
    if s.get('vcp_weekly'): vcp.append('weekly')
    vcp_str = 'VCP detected on ' + '+'.join(vcp) if vcp else 'no VCP flag'
    vol    = s.get('vol_ratio')
    vol_str = f"{vol:.2f}x avg ({'drying up' if vol and vol < 0.8 else 'elevated' if vol and vol > 1.2 else 'normal'})" if vol else 'unknown'
    eps    = s.get('q_eps_yoy')
    eps_str = f", EPS YoY: {eps:+.0f}%" if eps else ''
    return (
        f"On {date} ({regime} market): {s['ticker']} appeared in {scrs}. "
        f"RS rank: {s.get('rs_rank', '?')}, Price: ₹{s.get('price', '?')}, "
        f"% from 52w high: {s.get('pct_from_high', '?')}%, "
        f"Volume: {vol_str}, {vcp_str}, "
        f"Criteria passed: {s.get('passed', '?')}/8{eps_str}."
    )


# ── RAG index builder ─────────────────────────────────────────────────────────
def build_history_index():
    """Index all scan_history.jsonl records into ChromaDB collection 'scan_history'."""
    if not os.path.exists(HISTORY_JSONL):
        print('  No scan_history.jsonl found. Run take_snapshot() first.')
        return

    from sentence_transformers import SentenceTransformer
    import chromadb

    print('  Loading embedding model...')
    model  = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=RAG_DB)

    try:
        client.delete_collection('scan_history')
    except Exception:
        pass
    collection = client.create_collection('scan_history', metadata={'hnsw:space': 'cosine'})

    records = []
    with open(HISTORY_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    print(f'  Indexing {len(records)} scan dates...')
    docs, metas, ids = [], [], []
    doc_id = 0

    for rec in records:
        date   = rec['date']
        regime = rec.get('regime', 'Unknown')

        # One document per stock per date
        for s in rec.get('stocks', []):
            doc = _stock_to_doc(date, regime, s)
            docs.append(doc)
            metas.append({
                'date':   date,
                'ticker': s['ticker'],
                'regime': regime,
                'rs':     str(s.get('rs_rank', '')),
                'screens': '|'.join(s.get('screens', [])),
            })
            ids.append(f'hist_{doc_id}')
            doc_id += 1

        # Also one document for the day's overview
        counts = rec.get('screen_counts', {})
        overview = (
            f"Scan on {date} ({regime} regime): "
            f"Full Template {counts.get('full_template', 0)}, "
            f"VCP Setup {counts.get('vcp_setup', 0)}, "
            f"CCI Daily {counts.get('ck_daily_100', 0)}, "
            f"CCI Weekly {counts.get('ck_weekly_100', 0)}, "
            f"Near Breakout {counts.get('near_breakout', 0)}, "
            f"MA Pullback {counts.get('ma_pullback', 0)} stocks."
        )
        docs.append(overview)
        metas.append({'date': date, 'ticker': '__overview__', 'regime': regime, 'rs': '', 'screens': ''})
        ids.append(f'hist_{doc_id}')
        doc_id += 1

    # Embed and store in batches
    batch = 128
    for i in range(0, len(docs), batch):
        embeddings = model.encode(docs[i:i+batch], show_progress_bar=False).tolist()
        collection.add(
            documents=docs[i:i+batch],
            embeddings=embeddings,
            metadatas=metas[i:i+batch],
            ids=ids[i:i+batch],
        )
        print(f'  Indexed {min(i+batch, len(docs))}/{len(docs)} documents', end='\r')

    print(f'\n  History RAG ready: {collection.count()} documents from {len(records)} scan dates.')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n  AlphaEdge Scan History RAG Builder')
    print('  ────────────────────────────────────')

    snap = take_snapshot()
    if snap:
        append_snapshot(snap)
        print(f'  Snapshot saved: {snap["date"]} ({len(snap["stocks"])} stocks, {snap["regime"]} regime)')
    else:
        print('  WARNING: No results.json found — using existing history only.')

    build_history_index()
    print()
