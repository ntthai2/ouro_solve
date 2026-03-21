"""
Unified policy server for OC + OQ modes.

Local:   python server.py  -> http://localhost:7734

Endpoints:
    GET  /                     -> serves guide.html
  GET  /state?mode=oc|oq     -> current belief state and recommendation
  POST /reveal?mode=oc|oq    -> submit reveal JSON {"cell": 13, "color": 4}
  POST /reset?mode=oc|oq     -> reset selected game mode

Default mode is OC when mode is not specified.
"""

import os
import json
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from oc_board_generator import (
    compute_board_weights,
    COLOR_NAMES as OC_COLOR_NAMES,
    COLOR_VALUES as OC_COLOR_VALUES,
    NUM_CELLS as OC_NUM_CELLS,
    CENTER as OC_CENTER,
)
from oc_belief_state import FullBeliefState as OCFullBeliefState
from oc_strategies import VOIGreedy

from oq_board_generator import (
    NUM_CELLS as OQ_NUM_CELLS,
    COLOR_NAMES as OQ_COLOR_NAMES,
    COLOR_VALUES as OQ_COLOR_VALUES,
    COLOR_PURPLE,
    COLOR_RED,
)
from oq_belief_state import OQFullBeliefState
from oq_strategies import OQVOIGreedy

MAX_CLICKS = 5
PORT = int(os.environ.get("PORT", 7734))

# OC cache config
OC_POLICY_CACHE = "cache/voi_d3_cache.pkl"
OC_POLICY_DEPTH = 3
OC_BOARDS_PATH = "cache/all_boards.npy"

# OQ cache config
OQ_POLICY_CACHE = "cache/voi_oq_d2_cache.pkl"
OQ_POLICY_DEPTH = 2
OQ_BOARDS_PATH = "cache/all_boards_oq.npy"


# -- startup loading ----------------------------------------------------------

print("Loading OC boards...")
oc_boards = np.load(OC_BOARDS_PATH)
oc_weights = compute_board_weights(oc_boards)
OCFullBeliefState.load_boards(oc_boards, weights=oc_weights)
print(f"  OC boards loaded: {len(oc_boards):,}")

print(f"Loading OC policy from {OC_POLICY_CACHE}...")
with open(OC_POLICY_CACHE, "rb") as f:
    oc_data = pickle.load(f)
oc_policy = VOIGreedy(depth=OC_POLICY_DEPTH)
oc_policy._value_memo = oc_data.get("value_memo", {})
oc_policy._policy_memo = oc_data["policy_memo"]
print(f"  OC policy loaded ({len(oc_policy._policy_memo):,} states).")

print("Loading OQ boards...")
oq_boards = np.load(OQ_BOARDS_PATH)
OQFullBeliefState.load_boards(oq_boards)
print(f"  OQ boards loaded: {len(oq_boards):,}")

print(f"Loading OQ policy from {OQ_POLICY_CACHE}...")
with open(OQ_POLICY_CACHE, "rb") as f:
    oq_data = pickle.load(f)
oq_policy = OQVOIGreedy(depth=OQ_POLICY_DEPTH)
oq_policy._value_memo = oq_data.get("value_memo", {})
oq_policy._policy_memo = oq_data["policy_memo"]
print(f"  OQ policy loaded ({len(oq_policy._policy_memo):,} states).")


# -- OC game state (copied unchanged from oc_server.py) ----------------------

policy = oc_policy


class GameState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.belief = OCFullBeliefState()
        self.clicks_left = MAX_CLICKS
        self.history = []
        self.score = 0
        self.done = False

    def recommend(self):
        if self.done:
            return None
        return policy(self.belief, self.clicks_left)

    def reveal(self, cell: int, color: int):
        if self.done:
            return
        reward = OC_COLOR_VALUES[color]
        self.score += reward
        self.history.append({"cell": cell, "color": color,
                             "color_name": OC_COLOR_NAMES[color], "reward": reward})
        self.belief = self.belief.update(cell, color)
        self.clicks_left -= 1
        if self.clicks_left == 0:
            self.done = True

    def to_dict(self):
        rec = self.recommend()
        candidates = sorted(list(self.belief.red_candidates()))
        cells = []
        revealed_map = {h["cell"]: h for h in self.history}
        for i in range(OC_NUM_CELLS):
            if i in revealed_map:
                h = revealed_map[i]
                cells.append({"index": i, "state": "revealed",
                              "color": h["color"], "color_name": h["color_name"],
                              "reward": h["reward"]})
            elif i == OC_CENTER and self.clicks_left == MAX_CLICKS:
                cells.append({"index": i, "state": "center", "color": -1})
            elif i == rec:
                cells.append({"index": i, "state": "recommended", "color": -1})
            elif i in candidates:
                cells.append({"index": i, "state": "candidate", "color": -1})
            else:
                cells.append({"index": i, "state": "normal", "color": -1})

        return {
            "clicks_left": self.clicks_left,
            "score": self.score,
            "done": self.done,
            "recommended": rec,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "history": self.history,
            "cells": cells,
            "consistent_boards": len(self.belief.board_indices),
        }


# Alias for clarity in unified server.
OCGame = GameState


class OQGame:
    MAX_PAID_CLICKS = 7
    MAX_SCORE = 495

    def __init__(self):
        self.reset()

    def reset(self):
        self.belief = OQFullBeliefState()
        self.paid_clicks_left = self.MAX_PAID_CLICKS
        self.score = 0
        self.done = False
        self.history = []
        self.purples_found = 0
        self.conversion_cell = None
        self.red_found = False

    def _clicked_cells(self):
        return {h["cell"] for h in self.history}

    def _try_activate_conversion(self):
        if self.purples_found != 3:
            return
        possible = set(self.belief.possible_purple_cells()) - self._clicked_cells()
        if len(possible) == 1:
            self.conversion_cell = next(iter(possible))
            self.belief = self.belief.peek(self.conversion_cell, COLOR_PURPLE)

    def reveal(self, cell, color):
        if self.done:
            return

        if self.conversion_cell is not None and cell == self.conversion_cell:
            reward = int(OQ_COLOR_VALUES[COLOR_RED])
            self.score += reward
            self.paid_clicks_left -= 1
            self.red_found = True
            self.belief = self.belief.update(cell, COLOR_PURPLE)
            self.history.append({
                "cell": cell,
                "color": COLOR_RED,
                "color_name": OQ_COLOR_NAMES[COLOR_RED],
                "reward": reward,
                "paid_click": True,
                "converted_red": True,
            })
            self.done = True
            return

        if color == COLOR_PURPLE:
            reward = int(OQ_COLOR_VALUES[COLOR_PURPLE])
            self.score += reward
            self.purples_found += 1
            self.belief = self.belief.update(cell, COLOR_PURPLE)
            self.history.append({
                "cell": cell,
                "color": COLOR_PURPLE,
                "color_name": OQ_COLOR_NAMES[COLOR_PURPLE],
                "reward": reward,
                "paid_click": False,
                "converted_red": False,
            })
            self._try_activate_conversion()
            return

        reward = int(OQ_COLOR_VALUES[color])
        self.score += reward
        self.paid_clicks_left -= 1
        self.belief = self.belief.update(cell, color)
        self.history.append({
            "cell": cell,
            "color": color,
            "color_name": OQ_COLOR_NAMES[color],
            "reward": reward,
            "paid_click": True,
            "converted_red": False,
        })

        if self.paid_clicks_left == 0:
            self.done = True

    def recommend(self):
        if self.done:
            return None
        return oq_policy(self.belief, self.paid_clicks_left)

    def to_dict(self):
        rec = self.recommend()
        purple_candidates = sorted(list(self.belief.possible_purple_cells() - self._clicked_cells()))

        revealed_map = {h["cell"]: h for h in self.history}
        cells = []
        for i in range(OQ_NUM_CELLS):
            if i in revealed_map:
                h = revealed_map[i]
                cells.append({
                    "index": i,
                    "state": "revealed",
                    "color": h["color"],
                    "color_name": h["color_name"],
                    "reward": h["reward"],
                })
            elif self.conversion_cell is not None and i == self.conversion_cell:
                cells.append({"index": i, "state": "conversion", "color": -1})
            elif i == rec:
                cells.append({"index": i, "state": "recommended", "color": -1})
            elif i in purple_candidates:
                cells.append({"index": i, "state": "purple_candidate", "color": -1})
            else:
                cells.append({"index": i, "state": "normal", "color": -1})

        return {
            "paid_clicks_left": self.paid_clicks_left,
            "score": self.score,
            "done": self.done,
            "red_found": self.red_found,
            "purples_found": self.purples_found,
            "conversion_cell": self.conversion_cell,
            "recommended": rec,
            "purple_candidates": purple_candidates,
            "history": self.history,
            "cells": cells,
            "consistent_boards": len(self.belief.board_indices),
            "max_score": self.MAX_SCORE,
        }


oc_game = OCGame()
oq_game = OQGame()


def _mode_from_path(path: str) -> str:
    query = parse_qs(urlparse(path).query)
    mode = query.get("mode", ["oc"])[0].lower()
    return mode


def _game_for_mode(mode: str):
    if mode == "oc":
        return oc_game, OC_NUM_CELLS, 5
    if mode == "oq":
        return oq_game, OQ_NUM_CELLS, 6
    return None, None, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        mode = _mode_from_path(self.path)

        if path in ("/", "/index.html"):
            self._send_html("guide.html")
            return

        if path == "/state":
            game, _, _ = _game_for_mode(mode)
            if game is None:
                self._send_json({"error": "invalid mode, use oc or oq"}, 400)
                return
            payload = game.to_dict()
            payload["mode"] = mode
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        mode = _mode_from_path(self.path)
        game, num_cells, max_color = _game_for_mode(mode)

        if game is None:
            self._send_json({"error": "invalid mode, use oc or oq"}, 400)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/reveal":
            cell = int(body.get("cell", -1))
            color = int(body.get("color", -1))
            if cell < 0 or cell >= num_cells or color < 0 or color > max_color:
                self._send_json({"error": "invalid cell or color"}, 400)
                return
            if cell in {h["cell"] for h in game.history}:
                self._send_json({"error": "cell already revealed"}, 400)
                return
            game.reveal(cell, color)
            payload = game.to_dict()
            payload["mode"] = mode
            self._send_json(payload)
            return

        if path == "/reset":
            game.reset()
            payload = game.to_dict()
            payload["mode"] = mode
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    host = "0.0.0.0"
    server = HTTPServer((host, PORT), Handler)
    print(f"\nUnified server running at http://{host}:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
