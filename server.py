"""
server.py

Local HTTP server that loads the precomputed POMDP policy and serves
recommendations via a simple JSON API.

Run with: python server.py
Then open browser_guide.html in your browser.

Endpoints:
  GET  /state          — current belief state and recommendation
  POST /reveal         — submit a reveal: {"cell": 13, "color": 4}
  POST /reset          — reset to initial state
"""

import json
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from board_generator import compute_board_weights, COLOR_NAMES, COLOR_VALUES, NUM_CELLS, CENTER
from belief_state import FullBeliefState
from strategies import VOIGreedy

MAX_CLICKS  = 5
PORT        = 7734

# Choose which strategy to use for recommendations.
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
        self.history     = []   # list of (cell, color, reward)
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
        rec = self.recommend()
        # candidate cells for red
        candidates = sorted(list(self.belief.red_candidates()))
        # per-cell info
        cells = []
        revealed_map = {h['cell']: h for h in self.history}
        for i in range(NUM_CELLS):
            if i in revealed_map:
                h = revealed_map[i]
                cells.append({
                    'index': i, 'state': 'revealed',
                    'color': h['color'], 'color_name': h['color_name'],
                    'reward': h['reward'],
                })
            elif i == CENTER and self.clicks_left == MAX_CLICKS:
                cells.append({'index': i, 'state': 'center', 'color': -1})
            elif i == rec:
                cells.append({'index': i, 'state': 'recommended', 'color': -1})
            elif i in candidates:
                cells.append({'index': i, 'state': 'candidate', 'color': -1})
            else:
                cells.append({'index': i, 'state': 'normal', 'color': -1})

        return {
            'clicks_left':   self.clicks_left,
            'score':         self.score,
            'done':          self.done,
            'recommended':   rec,
            'candidates':    candidates,
            'candidate_count': len(candidates),
            'history':       self.history,
            'cells':         cells,
            'consistent_boards': len(self.belief.board_indices),
        }

game = GameState()

# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def _send(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/state':
            self._send(game.to_dict())
        else:
            self._send({'error': 'not found'}, 404)

    def do_POST(self):
        path    = urlparse(self.path).path
        length  = int(self.headers.get('Content-Length', 0))
        body    = json.loads(self.rfile.read(length)) if length else {}

        if path == '/reveal':
            cell  = int(body.get('cell', -1))
            color = int(body.get('color', -1))
            if cell < 0 or cell >= NUM_CELLS or color < 0 or color > 5:
                self._send({'error': 'invalid cell or color'}, 400)
                return
            if cell in {h['cell'] for h in game.history}:
                self._send({'error': 'cell already revealed'}, 400)
                return
            game.reveal(cell, color)
            self._send(game.to_dict())

        elif path == '/reset':
            game.reset()
            self._send(game.to_dict())

        else:
            self._send({'error': 'not found'}, 404)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    server = HTTPServer(('localhost', PORT), Handler)
    print(f"\nSeeRed policy server running at http://localhost:{PORT}")
    print("Open browser_guide.html in your browser.")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")