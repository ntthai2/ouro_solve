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
import math
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from oc.board_generator import (
    compute_board_weights,
    COLOR_NAMES as OC_COLOR_NAMES,
    COLOR_VALUES as OC_COLOR_VALUES,
    NUM_CELLS as OC_NUM_CELLS,
    CENTER as OC_CENTER,
)
from oc.belief_state import FullBeliefState as OCFullBeliefState
from oc.strategies import VOIGreedy

from oq.board_generator import (
    NUM_CELLS as OQ_NUM_CELLS,
    COLOR_NAMES as OQ_COLOR_NAMES,
    COLOR_VALUES as OQ_COLOR_VALUES,
    COLOR_PURPLE,
    COLOR_RED,
)
from oq.belief_state import OQFullBeliefState
from oq.strategies import OQVOIGreedy

from ot.board_generator import (
    NUM_CELLS as OT_NUM_CELLS,
    COLOR_NAMES as OT_COLOR_NAMES,
    COLOR_VALUES as OT_COLOR_VALUES,
    COLOR_BLUE as OT_COLOR_BLUE,
    MAX_BLUE_CLICKS
)
from ot.belief_state import OTBeliefState
from ot.strategies import OTInfoGainStrategy

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
                              "reward": h["reward"], "certain_color": h["color"]})
            else:
                poss = self.belief.possible_colors(i)
                certain_col = int(poss[0]) if len(poss) == 1 else -1

                if i == OC_CENTER and self.clicks_left == MAX_CLICKS:
                    st = "center"
                elif i == rec:
                    st = "recommended"
                elif i in candidates:
                    st = "candidate"
                else:
                    st = "normal"

                cells.append({
                    "index": i,
                    "state": st,
                    "color": -1,
                    "certain_color": certain_col,
                })

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
            if self.paid_clicks_left == 0:
                self.done = True
            return

        if color == COLOR_RED:
            if self.purples_found != 3:
                return
            reward = int(OQ_COLOR_VALUES[COLOR_RED])
            self.score += reward
            self.paid_clicks_left -= 1
            self.red_found = True
            self.conversion_cell = cell
            self.belief = self.belief.update(cell, COLOR_PURPLE)
            self.history.append({
                "cell": cell,
                "color": COLOR_RED,
                "color_name": OQ_COLOR_NAMES[COLOR_RED],
                "reward": reward,
                "paid_click": True,
                "converted_red": True,
            })
            if self.paid_clicks_left == 0:
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
                    "certain_color": h["color"],
                })
            else:
                poss = self.belief.possible_colors(i)
                if self.conversion_cell is not None and i == self.conversion_cell:
                    certain_col = COLOR_RED
                    st = "conversion"
                else:
                    certain_col = int(poss[0]) if len(poss) == 1 else -1
                    if i == rec:
                        st = "recommended"
                    elif i in purple_candidates:
                        st = "purple_candidate"
                    else:
                        st = "normal"

                cells.append({
                    "index": i,
                    "state": st,
                    "color": -1,
                    "certain_color": certain_col,
                })

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


class OTGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.belief = OTBeliefState()
        self.blue_clicks = 0
        self.score = 0
        self.done = False
        self.history = []
        self.clicked_cells = set()
        self._cached_rec = None

    def reveal(self, cell, color):
        if self.done: return
        if cell in self.clicked_cells: return
        
        self.clicked_cells.add(cell)
        self._cached_rec = None
        val = OT_COLOR_VALUES[color] if 0 <= color < len(OT_COLOR_VALUES) else None
        reward = val if val is not None else 150 # Fallback for rare colors
        c_name = OT_COLOR_NAMES[color] if 0 <= color < len(OT_COLOR_NAMES) else "Unknown"
        
        self.score += reward
        self.history.append({
            "cell": cell,
            "color": color,
            "color_name": c_name,
            "reward": reward
        })
        
        self.belief = self.belief.update(cell, color)
        if color == OT_COLOR_BLUE:
            self.blue_clicks += 1
            if self.blue_clicks >= MAX_BLUE_CLICKS:
                self.done = True

    def recommend(self):
        if self.done: return None
        remaining = [c for c in range(OT_NUM_CELLS) if c not in self.clicked_cells]
        if not remaining: return None
        # Root opening: C3 (Cell 12) is proven lowest p_blue (~33.2%)
        if len(self.clicked_cells) == 0:
            return 12
        if self._cached_rec is None or self._cached_rec not in remaining:
            self._cached_rec = ot_policy(self.belief, remaining)
        return self._cached_rec
        
    def to_dict(self):
        rec = self.recommend()
        safe_cells = list(self.belief.certain_safe_cells())
        
        revealed_map = {h["cell"]: h for h in self.history}
        cells = []
        for i in range(OT_NUM_CELLS):
            if i in revealed_map:
                h = revealed_map[i]
                cells.append({
                    "index": i,
                    "state": "revealed",
                    "color": h["color"],
                    "color_name": h["color_name"],
                    "reward": h["reward"]
                })
            else:
                if i == rec:
                    st = "recommended"
                elif i in safe_cells:
                    st = "safe_candidate"
                else:
                    st = "normal"
                cells.append({
                    "index": i,
                    "state": st,
                    "color": -1
                })
                
        return {
            "score": self.score,
            "blue_clicks": self.blue_clicks,
            "max_blue_clicks": MAX_BLUE_CLICKS,
            "done": self.done,
            "recommended": rec,
            "safe_cells": safe_cells,
            "history": self.history,
            "cells": cells,
            "max_score": 1045, # Theoretical max roughly
        }

oc_game = OCGame()
oq_game = OQGame()
ot_game = OTGame()

# λ=1.00 (Greedy) chosen after N=1500 validation showed InfoGain(λ=0.95) was not statistically better (p=0.86)
OT_LAMBDA = 1.00 
ot_policy = OTInfoGainStrategy(lam=OT_LAMBDA, use_exact_endgame=True, n_samples=1000)


def _mode_from_path(path: str) -> str:
    query = parse_qs(urlparse(path).query)
    mode = query.get("mode", ["oc"])[0].lower()
    return mode


def _game_for_mode(mode: str):
    if mode == "oc":
        return oc_game, OC_NUM_CELLS, 5
    if mode == "oq":
        return oq_game, OQ_NUM_CELLS, 6
    if mode == "ot":
        # maximum color id for ot is 8 (0=Blue,1=Teal,2=Green,3=Yellow,4=Orange,5=White,6=Black,7=Red,8=Rainbow)
        return ot_game, OT_NUM_CELLS, 8
    return None, None, None


def _parse_cell_arg(cell_val: str, num_cells: int = 25) -> int:
    """Parse cell integer index (0-24) or coordinate string like B1, C3."""
    if cell_val is None:
        return -1
    cell_val = str(cell_val).strip().upper()
    if cell_val.isdigit():
        idx = int(cell_val)
        return idx if 0 <= idx < num_cells else -1
    if len(cell_val) >= 2 and cell_val[0] in "ABCDE" and cell_val[1:].isdigit():
        col = ord(cell_val[0]) - ord("A")
        row = int(cell_val[1:]) - 1
        if 0 <= col < 5 and 0 <= row < 5:
            return row * 5 + col
    return -1


def explain_oc(belief, clicks_left, target_cell=None):
    rec = oc_policy(belief, clicks_left)
    cell = rec if target_cell is None or target_cell < 0 else target_cell

    unclicked = list(belief.unclicked())
    if cell not in unclicked and unclicked:
        cell = unclicked[0]

    scores = []
    for c in unclicked:
        ev = 0.0
        for color in belief.possible_colors(c):
            p = belief.p_color(c, color)
            if p == 0.0:
                continue
            reward = OC_COLOR_VALUES[color]
            new_b = belief.update(c, color)
            future = oc_policy._value(new_b, clicks_left - 1, current_depth=1)
            ev += p * (reward + future)
        scores.append((float(ev), int(c)))
    scores.sort(key=lambda x: x[0], reverse=True)

    breakdown = []
    imm_ev = 0.0
    exp_boards_elim = 0.0
    exp_h_post = 0.0

    idx_arr = np.array(list(belief.board_indices), dtype=np.int32)
    if len(idx_arr) > 0:
        w = OCFullBeliefState.ALL_WEIGHTS[idx_arr]
        w_norm = w / w.sum()
        red_pos = OCFullBeliefState.ALL_BOARDS[idx_arr, :] == 5
        p_red = np.zeros(OC_NUM_CELLS)
        for i in range(OC_NUM_CELLS):
            p_red[i] = w_norm[red_pos[:, i]].sum()
        h_curr = -sum(float(p) * math.log2(float(p)) for p in p_red if p > 0)
    else:
        h_curr = 0.0

    for color in belief.possible_colors(cell):
        p = float(belief.p_color(cell, color))
        if p == 0.0:
            continue
        reward = int(OC_COLOR_VALUES[color])
        imm_ev += p * reward
        next_b = belief.update(cell, color)

        future_v = float(oc_policy._value(next_b, clicks_left - 1, current_depth=1))

        next_idx = np.array(list(next_b.board_indices), dtype=np.int32)
        if len(next_idx) > 0:
            nw = OCFullBeliefState.ALL_WEIGHTS[next_idx]
            nw_norm = nw / nw.sum()
            n_red_pos = OCFullBeliefState.ALL_BOARDS[next_idx, :] == 5
            p_red_next = np.zeros(OC_NUM_CELLS)
            for i in range(OC_NUM_CELLS):
                p_red_next[i] = nw_norm[n_red_pos[:, i]].sum()
            h_post = -sum(float(p_n) * math.log2(float(p_n)) for p_n in p_red_next if p_n > 0)
        else:
            h_post = 0.0
        exp_h_post += p * h_post

        b_elim = len(belief.board_indices) - len(next_b.board_indices)
        exp_boards_elim += p * b_elim

        breakdown.append({
            "color": int(color),
            "name": OC_COLOR_NAMES[color],
            "prob": round(p, 4),
            "prob_pct": f"{p * 100:.1f}%",
            "reward": reward,
            "weighted_reward": round(p * reward, 2),
            "future_ev": round(future_v, 2),
            "boards_left": len(next_b.board_indices),
            "candidates_left": len(next_b.red_candidates()),
        })

    info_gain = max(0.0, h_curr - exp_h_post)

    runner_up = None
    for s, c in scores:
        if c != cell:
            col_letter = chr(ord("A") + (c % 5))
            row_num = (c // 5) + 1
            runner_up = {
                "cell": int(c),
                "label": f"{col_letter}{row_num}",
                "total_ev": round(s, 2),
                "diff": round(scores[0][0] - s, 2),
            }
            break

    target_total_ev = scores[0][0] if cell == rec else next((s for s, c in scores if c == cell), 0.0)
    col_letter = chr(ord("A") + (cell % 5))
    row_num = (cell // 5) + 1

    return {
        "mode": "oc",
        "cell": int(cell),
        "cell_label": f"{col_letter}{row_num}",
        "recommended": int(rec) if rec is not None else None,
        "is_recommended": (cell == rec),
        "clicks_left": int(clicks_left),
        "immediate_ev": round(imm_ev, 2),
        "total_ev": round(target_total_ev, 2),
        "info_gain_bits": round(info_gain, 2),
        "expected_boards_eliminated": round(exp_boards_elim, 1),
        "total_boards": len(belief.board_indices),
        "breakdown": breakdown,
        "runner_up": runner_up,
    }


def explain_oq(game: OQGame, target_cell=None):
    belief = game.belief
    clicks_left = game.paid_clicks_left
    if game.conversion_cell is not None:
        rec = game.conversion_cell
    else:
        rec = oq_policy(belief, clicks_left)

    cell = rec if target_cell is None or target_cell < 0 else target_cell

    unclicked = list(belief.unclicked())
    if cell not in unclicked and unclicked:
        cell = unclicked[0]

    scores = []
    for c in unclicked:
        ev = 0.0
        for color in belief.possible_colors(c):
            p = belief.p_color(c, color)
            if p == 0.0:
                continue
            reward = oq_policy._effective_reward(c, color, belief)
            new_b = belief.update(c, color)
            if color == COLOR_PURPLE:
                future = oq_policy._value(new_b, clicks_left, current_depth=1)
            else:
                future = oq_policy._value(new_b, clicks_left - 1, current_depth=1)
            ev += p * (reward + future)
        scores.append((float(ev), int(c)))
    scores.sort(key=lambda x: x[0], reverse=True)

    breakdown = []
    imm_ev = 0.0
    exp_boards_elim = 0.0
    exp_purple_cands = 0.0

    for color in belief.possible_colors(cell):
        p = float(belief.p_color(cell, color))
        if p == 0.0:
            continue
        reward = int(oq_policy._effective_reward(cell, color, belief))
        imm_ev += p * reward
        next_b = belief.update(cell, color)

        if color == COLOR_PURPLE:
            future_v = float(oq_policy._value(next_b, clicks_left, current_depth=1))
        else:
            future_v = float(oq_policy._value(next_b, clicks_left - 1, current_depth=1))

        b_elim = len(belief.board_indices) - len(next_b.board_indices)
        exp_boards_elim += p * b_elim

        cands_left = len(next_b.possible_purple_cells() - next_b.revealed)
        exp_purple_cands += p * cands_left

        breakdown.append({
            "color": int(color),
            "name": OQ_COLOR_NAMES[color],
            "prob": round(p, 4),
            "prob_pct": f"{p * 100:.1f}%",
            "reward": reward,
            "weighted_reward": round(p * reward, 2),
            "future_ev": round(future_v, 2),
            "boards_left": len(next_b.board_indices),
            "candidates_left": cands_left,
        })

    runner_up = None
    for s, c in scores:
        if c != cell:
            col_letter = chr(ord("A") + (c % 5))
            row_num = (c // 5) + 1
            runner_up = {
                "cell": int(c),
                "label": f"{col_letter}{row_num}",
                "total_ev": round(s, 2),
                "diff": round(scores[0][0] - s, 2),
            }
            break

    curr_cands = len(belief.possible_purple_cells() - belief.revealed)
    cands_elim = max(0.0, curr_cands - exp_purple_cands)

    target_total_ev = scores[0][0] if cell == rec else next((s for s, c in scores if c == cell), 0.0)
    col_letter = chr(ord("A") + (cell % 5))
    row_num = (cell // 5) + 1

    return {
        "mode": "oq",
        "cell": int(cell),
        "cell_label": f"{col_letter}{row_num}",
        "recommended": int(rec) if rec is not None else None,
        "is_recommended": (cell == rec),
        "clicks_left": int(clicks_left),
        "immediate_ev": round(imm_ev, 2),
        "total_ev": round(target_total_ev, 2),
        "expected_boards_eliminated": round(exp_boards_elim, 1),
        "expected_cands_eliminated": round(cands_elim, 1),
        "total_boards": len(belief.board_indices),
        "current_candidates": curr_cands,
        "breakdown": breakdown,
        "runner_up": runner_up,
    }

def explain_ot(game: OTGame, target_cell=None):
    belief = game.belief
    rec = game.recommend()
    cell = rec if target_cell is None or target_cell < 0 else target_cell
    
    remaining = [c for c in range(OT_NUM_CELLS) if c not in game.clicked_cells]
    if cell not in remaining and remaining:
        cell = remaining[0]
        
    probs = belief.p_color_all(use_exact_endgame=ot_policy.use_exact_endgame, n_samples=ot_policy.n_samples)
    p_blue = probs[OT_COLOR_BLUE][cell]
    
    e_info = 0.0
    breakdown = []
    
    for c, p_c_list in probs.items():
        p_c = p_c_list[cell]
        if p_c > 0:
            if c != OT_COLOR_BLUE:
                new_belief = belief.update(cell, c)
                new_safe = new_belief.certain_safe_cells()
                valid_new_safe = [s for s in new_safe if s in remaining and s != cell]
                info_gain = len(valid_new_safe)
                e_info += p_c * info_gain
            else:
                info_gain = 0
                
            c_name = OT_COLOR_NAMES[c] if 0 <= c < len(OT_COLOR_NAMES) else "Unknown"
            breakdown.append({
                "color": int(c),
                "name": c_name,
                "prob": round(p_c, 4),
                "prob_pct": f"{p_c * 100:.1f}%",
                "info_gain_cells": info_gain
            })
            
    score = -ot_policy.lam * p_blue + (1.0 - ot_policy.lam) * e_info
    
    target_total_score = score
    col_letter = chr(ord("A") + (cell % 5))
    row_num = (cell // 5) + 1
    
    return {
        "mode": "ot",
        "cell": int(cell),
        "cell_label": f"{col_letter}{row_num}",
        "recommended": int(rec) if rec is not None else None,
        "is_recommended": (cell == rec),
        "blue_clicks_left": MAX_BLUE_CLICKS - game.blue_clicks,
        "immediate_p_blue": round(p_blue, 4),
        "expected_info_gain": round(e_info, 2),
        "total_score": round(target_total_score, 4),
        "breakdown": breakdown,
    }


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
        query = parse_qs(parsed.query)
        mode = _mode_from_path(self.path)

        if path in ("/", "/index.html"):
            self._send_html("guide.html")
            return

        if path == "/state":
            game, _, _ = _game_for_mode(mode)
            if game is None:
                self._send_json({"error": "invalid mode, use oc, oq, or ot"}, 400)
                return
            payload = game.to_dict()
            payload["mode"] = mode
            self._send_json(payload)
            return

        if path == "/explain":
            cell_arg = query.get("cell", [None])[0]
            target_cell = _parse_cell_arg(cell_arg)
            if mode == "oc":
                if oc_game.done:
                    self._send_json({"error": "game already done"}, 400)
                    return
                data = explain_oc(oc_game.belief, oc_game.clicks_left, target_cell)
                self._send_json(data)
                return
            elif mode == "oq":
                if oq_game.done:
                    self._send_json({"error": "game already done"}, 400)
                    return
                data = explain_oq(oq_game, target_cell)
                self._send_json(data)
                return
            elif mode == "ot":
                if ot_game.done:
                    self._send_json({"error": "game already done"}, 400)
                    return
                data = explain_ot(ot_game, target_cell)
                self._send_json(data)
                return
            else:
                self._send_json({"error": "invalid mode, use oc, oq, or ot"}, 400)
                return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        mode = _mode_from_path(self.path)
        game, num_cells, max_color = _game_for_mode(mode)

        if game is None:
            self._send_json({"error": "invalid mode, use oc, oq, or ot"}, 400)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/reveal":
            cell = int(body.get("cell", -1))
            color = int(body.get("color", -1))
            if cell < 0 or cell >= num_cells or color < 0 or color > max_color:
                self._send_json({"error": "invalid cell or color"}, 400)
                return
            if mode == "oq" and color == COLOR_RED and game.purples_found != 3:
                self._send_json({"error": "red can only be revealed after 3 purples"}, 400)
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
