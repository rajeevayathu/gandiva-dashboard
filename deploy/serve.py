#!/usr/bin/env python3.10
"""
Static server for the Gandiva dashboard, with HTTP caching disabled.

The stock http.server lets the browser cache results_embed.js, dashboard.html
and chart_viewer.html, which is how stale prices kept surviving a hard
refresh. Every response here carries no-store headers so the browser is
forced to re-fetch.

Run by the com.gandiva.webserver launchd agent, which restarts it if it
exits. To run by hand:  python3.10 deploy/serve.py
"""
import http.server
import os

PORT = 8765
# deploy/ lives inside the scanner repo; the site root is two levels up,
# the directory that holds both projects.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, *args):
        # Silent: under launchd this would otherwise grow an unbounded log.
        pass


if __name__ == '__main__':
    os.chdir(ROOT)
    http.server.test(HandlerClass=NoCacheHandler, port=PORT, bind='')
