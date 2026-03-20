"""
server.py

Policy server for SeeRed. Works both locally and on Render.

Local:   python server.py  → http://localhost:7734
Render:  deploys automatically, visit your Render URL

Endpoints:
  GET  /            — serves browser_guide.html
  GET  /state       — current belief state and recommendation
  POST /reveal      — submit a reveal: {"cell": 13, "color": 4}
  POST /reset       — reset to initial state
"""

import os
import json
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from board_generator import compute_board_weights, COLOR_NAMES, COLOR_VALUES, NUM_CELLS, CENTER
from belief_state import FullBeliefState
from strategies import VOIGreedy

MAX_CLICKS  = 5
PORT        = int(os.environ.get('PORT', 7734))

# Choose which strategy to use for recommendations.
# Options after running main.py:
#   cache/pomdp_cache.pkl   — optimal (789 MB, slow to load)
#   cache/voi_d3_cache.pkl  — 16.6 MB, 0.01 pts below optimal (recommended)
#   cache/voi_d2_cache.pkl  — 1.3 MB, 1.14 pts below optimal
#   cache/voi_d1_cache.pkl  — 0.1 MB, fastest load
POLICY_CACHE = 'cache/voi_d3_cache.pkl'
POLICY_DEPTH = 3                   # must match the cache file chosen above
BOARDS_PATH  = 'cache/all_boards.npy'

# ── load everything once at startup ──────────────────────────────────────────

print("Loading boards...")
all_boards = np.load(BOARDS_PATH)
weights    = compute_board_weights(all_boards)
FullBeliefState.load_boards(all_boards, weights=weights)
print(f"  {len(all_boards):,} boards loaded.")

print(f"Loading policy from {POLICY_CACHE}...")
with open(POLICY_CACHE, 'rb') as f:
    data = pickle.load(f)
policy = VOIGreedy(depth=POLICY_DEPTH)
policy._value_memo  = data.get('value_memo', {})
policy._policy_memo = data['policy_memo']
print(f"  Policy loaded ({len(policy._policy_memo):,} states).")

# ── game state (single session) ───────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.belief      = FullBeliefState()
        self.clicks_left = MAX_CLICKS
        self.history     = []
        self.score       = 0
        self.done        = False

    def recommend(self):
        if self.done:
            return None
        return policy(self.belief, self.clicks_left)

    def reveal(self, cell: int, color: int):
        if self.done:
            return
        reward       = COLOR_VALUES[color]
        self.score  += reward
        self.history.append({'cell': cell, 'color': color,
                             'color_name': COLOR_NAMES[color], 'reward': reward})
        self.belief      = self.belief.update(cell, color)
        self.clicks_left -= 1
        if self.clicks_left == 0:
            self.done = True

    def to_dict(self):
        rec        = self.recommend()
        candidates = sorted(list(self.belief.red_candidates()))
        cells      = []
        revealed_map = {h['cell']: h for h in self.history}
        for i in range(NUM_CELLS):
            if i in revealed_map:
                h = revealed_map[i]
                cells.append({'index': i, 'state': 'revealed',
                               'color': h['color'], 'color_name': h['color_name'],
                               'reward': h['reward']})
            elif i == CENTER and self.clicks_left == MAX_CLICKS:
                cells.append({'index': i, 'state': 'center', 'color': -1})
            elif i == rec:
                cells.append({'index': i, 'state': 'recommended', 'color': -1})
            elif i in candidates:
                cells.append({'index': i, 'state': 'candidate', 'color': -1})
            else:
                cells.append({'index': i, 'state': 'normal', 'color': -1})

        return {
            'clicks_left':       self.clicks_left,
            'score':             self.score,
            'done':              self.done,
            'recommended':       rec,
            'candidates':        candidates,
            'candidate_count':   len(candidates),
            'history':           self.history,
            'cells':             cells,
            'consistent_boards': len(self.belief.board_indices),
        }

game = GameState()

# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path):
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_json({'error': 'not found'}, 404)

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send_html('browser_guide.html')
        elif path == '/state':
            self._send_json(game.to_dict())
        else:
            self._send_json({'error': 'not found'}, 404)

    def do_POST(self):
        path   = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body   = json.loads(self.rfile.read(length)) if length else {}

        if path == '/reveal':
            cell  = int(body.get('cell', -1))
            color = int(body.get('color', -1))
            if cell < 0 or cell >= NUM_CELLS or color < 0 or color > 5:
                self._send_json({'error': 'invalid cell or color'}, 400)
                return
            if cell in {h['cell'] for h in game.history}:
                self._send_json({'error': 'cell already revealed'}, 400)
                return
            game.reveal(cell, color)
            self._send_json(game.to_dict())

        elif path == '/reset':
            game.reset()
            self._send_json(game.to_dict())

        else:
            self._send_json({'error': 'not found'}, 404)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    host = '0.0.0.0'
    server = HTTPServer((host, PORT), Handler)
    print(f"\nSeeRed running at http://{host}:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")