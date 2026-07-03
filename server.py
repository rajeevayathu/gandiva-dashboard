#!/usr/bin/env python3
"""
Local dashboard server for Minervini Trend Template Scanner.

When running via this server, browser refresh always shows the latest scan data.
A "Scan Now" button in the dashboard triggers a fresh scan in the background.

Usage:
  python3 server.py              # starts on http://localhost:8765
  python3 server.py --port 9000  # custom port
"""
import http.server, json, os, sys, threading, subprocess, time, gzip
from datetime import datetime

PORT      = 8765
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON     = os.path.join(BASE_DIR, 'results.json')
BREADTH_JSON     = os.path.join(BASE_DIR, 'breadth.json')
CCI_HISTORY_JSON = os.path.join(BASE_DIR, 'cci_history.json')
SCANNER_PY   = os.path.join(BASE_DIR, 'scanner.py')
PYTHON       = sys.executable

# ── EXPORTS FOLDER ────────────────────────────────────────────────────────────
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')

# Map screen key → subfolder name
_SCREEN_FOLDER = {
    'full_template':        'Composite',
    'near_breakout':        'Composite',
    'rs_leaders':           'Composite',
    'vcp_setup':            'Composite',
    'new_highs':            'Composite',
    'watch_list':           'Composite',
    'minervini_backtest':   'Composite',
    'minervini_picks':      'Composite',
    'ipo_watch':            'IPO',
    'htf_setup':            'HTF_Trend',
    'htf_potential':        'HTF_Trend',
    'ma_pullback':          'HTF_Trend',
    'hhhl_pullback':        'HTF_Trend',
    'cci34_best_setups':    'CCI',
    'cci34_daily_cross_100':'CCI',
    'cci34_weekly_cross_100':'CCI',
    'cci34_daily_100':      'CCI',
    'cci34_weekly_100':     'CCI',
    'cci34_daily_neg100':   'CCI',
    'cci34_weekly_neg100':  'CCI',
    'strong_earnings':      'CCI',
    'fo_momentum':          'FnO',
    'fo_strong_uptrend':    'FnO',
    'newly_added':          'Composite',
}

def _get_export_path(screen_key, filename):
    subfolder = _SCREEN_FOLDER.get(screen_key, 'Individual')
    # Individual criteria screens C1-C8
    if screen_key.startswith('c') and screen_key[1:].isdigit():
        subfolder = 'Individual'
    folder = os.path.join(EXPORTS_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# ── Scan state (shared across threads) ────────────────────────────────────────
_lock           = threading.Lock()
_scanning       = False
_progress_lines = []
_scan_started   = None


def _run_scanner():
    global _scanning, _progress_lines, _scan_started
    with _lock:
        _scan_started   = datetime.now().isoformat()
        _progress_lines = ['Starting scanner...']
    try:
        proc = subprocess.Popen(
            [PYTHON, SCANNER_PY],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                with _lock:
                    _progress_lines.append(line)
                    if len(_progress_lines) > 30:
                        _progress_lines.pop(0)
                print(line)   # also echo to terminal
        proc.wait()
        with _lock:
            _progress_lines.append('Scan complete.')
    except Exception as e:
        with _lock:
            _progress_lines.append(f'Error: {e}')
    finally:
        with _lock:
            _scanning = False


def trigger_scan():
    """Start scanner in background. Returns True if started, False if already running."""
    global _scanning
    with _lock:
        if _scanning:
            return False
        _scanning = True
    threading.Thread(target=_run_scanner, daemon=True).start()
    return True


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == '/api/results':
            self._serve_results()
        elif self.path == '/api/breadth':
            self._serve_file_json(BREADTH_JSON)
        elif self.path == '/api/cci-history':
            self._serve_file_json(CCI_HISTORY_JSON)
        elif self.path == '/api/status':
            self._serve_status()
        elif self.path in ('/', '/dashboard.html'):
            self.path = '/dashboard.html'
            super().do_GET()
        else:
            super().do_GET()

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == '/api/scan':
            started = trigger_scan()
            self._send_json({'status': 'started' if started else 'already_running'})
        elif self.path == '/api/save-csv':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = self.rfile.read(length)
                data   = json.loads(body)
                screen_key = data.get('screen_key', 'unknown')
                filename   = data.get('filename', 'export.csv')
                csv_data   = data.get('csv', '')
                filepath   = _get_export_path(screen_key, filename)
                with open(filepath, 'w', encoding='utf-8-sig') as f:
                    f.write(csv_data)
                self._send_json({'status': 'saved', 'path': filepath})
            except Exception as e:
                self._send_json({'status': 'error', 'message': str(e)}, status=500)
        else:
            self.send_response(404)
            self.end_headers()

    # ── API helpers ───────────────────────────────────────────────────────────
    def _serve_results(self):
        if not os.path.exists(RESULTS_JSON):
            self._send_json({'error': 'no_data'}, status=404)
            return
        try:
            with open(RESULTS_JSON, 'r') as f:
                raw = f.read()
            # NaN/Infinity → null so browser JSON.parse doesn't throw
            import re
            raw = re.sub(r'\bNaN\b', 'null', raw)
            raw = re.sub(r'\bInfinity\b', 'null', raw)
            raw = re.sub(r'\b-Infinity\b', 'null', raw)
        except Exception as e:
            self._send_json({'error': str(e)}, status=500)
            return

        body = raw.encode()

        # Gzip if browser supports it — 4.6 MB → ~400 KB
        accept_enc = self.headers.get('Accept-Encoding', '')
        use_gzip = 'gzip' in accept_enc
        if use_gzip:
            body = gzip.compress(body, compresslevel=6)

        # ETag based on file mtime — lets browser skip re-download if unchanged
        mtime     = os.path.getmtime(RESULTS_JSON)
        etag      = f'"{int(mtime)}"'
        if self.headers.get('If-None-Match') == etag:
            self.send_response(304)
            self._cors()
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')   # revalidate; ETag handles skip
        self.send_header('ETag', etag)
        if use_gzip:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_file_json(self, path):
        """Serve a JSON file with gzip + ETag support."""
        if not os.path.exists(path):
            self._send_json({'error': 'no_data'}, status=404)
            return
        try:
            with open(path, 'r') as f:
                raw = f.read()
        except Exception as e:
            self._send_json({'error': str(e)}, status=500)
            return
        body = raw.encode()
        accept_enc = self.headers.get('Accept-Encoding', '')
        use_gzip = 'gzip' in accept_enc
        if use_gzip:
            body = gzip.compress(body, compresslevel=6)
        mtime = os.path.getmtime(path)
        etag  = f'"{int(mtime)}"'
        if self.headers.get('If-None-Match') == etag:
            self.send_response(304)
            self._cors()
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('ETag', etag)
        if use_gzip:
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_status(self):
        mtime = age_min = None
        if os.path.exists(RESULTS_JSON):
            mt      = os.path.getmtime(RESULTS_JSON)
            mtime   = datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M')
            age_min = round((time.time() - mt) / 60)
        with _lock:
            sc   = _scanning
            prog = list(_progress_lines)
        self._send_json({
            'scanning': sc,
            'last_scan': mtime,
            'age_min':   age_min,
            'progress':  prog[-8:] if prog else [],
        })

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')

    def log_message(self, fmt, *args):
        # Only log API hits; suppress noisy static-file requests
        msg = str(args[0]) if args else ''
        if '/api/' in msg:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if '--port' in sys.argv:
        try:
            PORT = int(sys.argv[sys.argv.index('--port') + 1])
        except (ValueError, IndexError):
            pass

    os.chdir(BASE_DIR)
    url = f'http://localhost:{PORT}'

    print(f'\n  Minervini Dashboard Server')
    print(f'  ──────────────────────────────')
    print(f'  Open:  {url}')
    print(f'  Press Ctrl+C to stop\n')

    # Auto-scan if data is missing or stale (>23 hrs)
    if not os.path.exists(RESULTS_JSON):
        print('  No results.json — starting initial scan in background...\n')
        trigger_scan()
    elif time.time() - os.path.getmtime(RESULTS_JSON) > 23 * 3600:
        age_h = (time.time() - os.path.getmtime(RESULTS_JSON)) / 3600
        print(f'  Scan data is {age_h:.1f}h old — refreshing in background...\n')
        trigger_scan()
    else:
        mt = datetime.fromtimestamp(os.path.getmtime(RESULTS_JSON)).strftime('%Y-%m-%d %H:%M')
        print(f'  Serving existing data from {mt}')
        print(f'  Click "Scan Now" in the dashboard to refresh.\n')

    server = http.server.HTTPServer(('localhost', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Server stopped.')
