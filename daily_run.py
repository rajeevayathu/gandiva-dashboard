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


def log(msg):
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_step(label, script):
    log(f'START  {label}')
    try:
        result = subprocess.run(
            [PYTHON, script],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
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
        log(f'TIMEOUT  {label} exceeded 30 minutes')
        return False
    except Exception as e:
        log(f'EXCEPTION  {label}: {e}')
        return False


if __name__ == '__main__':
    log('=' * 50)
    log('Gandiva daily run starting')

    ok = run_step('Scanner', os.path.join(BASE_DIR, 'scanner.py'))
    if ok:
        run_step('Morning Brief', os.path.join(BASE_DIR, 'morning_brief.py'))
    else:
        log('Scanner failed — skipping brief')

    log('Gandiva daily run complete')
    log('=' * 50)
