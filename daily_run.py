"""
Gandiva Daily Runner — scheduled at 8 PM via launchd.

Sequence:
  1. Run scanner.py  (fetches live data, updates all screens)
  2. Run morning_brief.py  (AI analysis + Telegram alert)

Logs written to: logs/daily_run.log
"""
import subprocess, sys, os, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'daily_run.log')
PYTHON   = '/opt/homebrew/bin/python3.10'
VIEWER_DIR = os.path.join(os.path.dirname(BASE_DIR), 'chart-pattern-viewer')


def log(msg):
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_step(label, script, args=None, cwd=None, timeout=1800):
    log(f'START  {label}')
    try:
        result = subprocess.run(
            [PYTHON, script] + (args or []),
            cwd=cwd or BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        for line in result.stdout.splitlines():
            log(f'  {line}')
        if result.returncode != 0:
            log(f'ERROR  {label} exited with code {result.returncode}')
            for line in result.stderr.splitlines():
                log(f'  STDERR: {line}')
            return False
        log(f'DONE   {label}')
        return True
    except subprocess.TimeoutExpired:
        log(f'TIMEOUT  {label} exceeded {timeout//60} minutes')
        return False
    except Exception as e:
        log(f'EXCEPTION  {label}: {e}')
        return False


if __name__ == '__main__':
    log('=' * 50)
    log('Gandiva daily run starting')

    ok = run_step('Scanner', os.path.join(BASE_DIR, 'scanner.py'))
    if ok:
        # Rebuild the chart viewer's data from the scan we just produced.
        # Without this the nightly run updates results.json but leaves
        # india_data.json / us_data.json untouched, so the viewer drifts
        # behind the scanner until a market-hours re-scan happens to fix it.
        # Over a weekend that is three days of stale screens.
        run_step('Charts', os.path.join(VIEWER_DIR, 'generate_charts.py'),
                 args=['--market', 'both', '--daily'], cwd=VIEWER_DIR,
                 timeout=3600)
        run_step('Morning Brief', os.path.join(BASE_DIR, 'morning_brief.py'))
    else:
        log('Scanner failed — skipping charts and brief')

    log('Gandiva daily run complete')
    log('=' * 50)
