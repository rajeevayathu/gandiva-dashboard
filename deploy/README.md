# Deployment — launchd agents

Three agents keep the dashboard current without anyone starting anything by
hand. Install them with:

```sh
cp deploy/com.gandiva.*.plist ~/Library/LaunchAgents/
for j in scanner priceupdater webserver; do
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gandiva.$j.plist
done
```

Check status with `launchctl print gui/$(id -u)/com.gandiva.<name>`.

| Agent | What it does | Trigger |
|---|---|---|
| `com.gandiva.scanner` | `daily_run.py` — full scan, chart rebuild, then the morning brief | 20:00 daily |
| `com.gandiva.priceupdater` | `price_updater.py` — prices every 60s, full re-scan every 30 min while a market is open | always on, `KeepAlive` |
| `com.gandiva.webserver` | `deploy/serve.py` — static server on :8765 with caching disabled | always on, `KeepAlive` |

`KeepAlive` is the important part for the latter two. They were previously
started by hand, so they died on reboot or when the terminal closed and
stayed dead silently — which is how the dashboard ended up serving week-old
data. launchd now restarts them whenever they exit.

## The EX_CONFIG(78) trap

If an agent reports `last exit code = 78: EX_CONFIG` and produces **no output
at all**, the cause is almost certainly its log files rather than its code.

launchd opens `StandardOutPath` / `StandardErrorPath` itself, before it execs
the program. A file carrying the `com.apple.macl` extended attribute — macOS
attaches these when a sandboxed app touches a file under `~/Desktop` — cannot
be opened by launchd, so the job aborts before running. Nothing is logged,
because the logging is what failed.

This silently killed the nightly scan between 2026-07-11 and 2026-10-06.

Diagnose:

```sh
ls -l@ logs/launchd_out.log          # look for com.apple.macl
xattr -l logs/launchd_out.log
```

Fix by moving the affected files aside; launchd recreates them clean:

```sh
mv logs/launchd_out.log logs/launchd_out.log.bak
launchctl kickstart -k gui/$(id -u)/com.gandiva.scanner
```

The two new agents write to `/tmp` instead of `~/Desktop` specifically to stay
out of reach of this attribute.

## Verifying KeepAlive

Kill a process and confirm launchd brings it back:

```sh
pkill -f serve.py && sleep 10 && curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/
```

A `200` means the agent is doing its job.

## Why the nightly run rebuilds charts

`daily_run.py` originally ran only `scanner.py` and `morning_brief.py`. That
updates `results.json`, but leaves the chart viewer's `india_data.json` and
`us_data.json` untouched — they are only rebuilt by `generate_charts.py`.

In practice the viewer was rescued by `price_updater.py`, which rebuilds charts
after each market-hours re-scan. But that loop does nothing when markets are
closed, so across a weekend the viewer sat three days behind the scanner, with
every mirrored screen empty while the scanner's own screens were current.

The nightly run now rebuilds charts itself, so the two stay in step regardless
of whether the market-hours loop ran.

## Why the nightly job runs under `caffeinate`

Unattended runs were taking 3-4 hours. The cause was not the scan itself: the
Mac idle-sleeps between maintenance wakes, which suspends `daily_run.py` and
its child scanner. Wall-clock time inflates while almost no work happens.

This also silently disables the per-step timeout. Python's `time.monotonic()`
does not advance while macOS sleeps, so `subprocess.run(timeout=1800)` counts
only *awake* seconds. On 2026-10-12 a run sat for 81 wall-clock minutes — hung
on a rate-limited Yahoo socket — without the 30-minute timeout ever firing,
because barely 30 minutes of awake time had accrued.

`caffeinate -i` holds off idle sleep for the lifetime of the run, so the job
makes continuous progress and the timeout measures something real.

Diagnose a suspected sleep-stall with:

```sh
pmset -g log | grep -iE "Entering Sleep|Wake from" | tail
```

A run spanning hours with repeated `Entering Sleep` entries inside its window
is this problem, not a slow scan.

### Yahoo rate limiting

Separately, heavy days produce bursts of `Too Many Requests` from Yahoo and the
scan slows while yfinance retries. The per-request timeouts (10-20s) are in
place, but the retry loop can still stall a run for a long time. If a scan
appears frozen, check for an open socket and no cache writes:

```sh
p=$(pgrep -f scanner.py | head -1)
lsof -p $p | grep TCP                              # one socket to yahoo
find .cache -name '*.pkl' -newermt '-2 minutes' | wc -l   # 0 means stalled
```
