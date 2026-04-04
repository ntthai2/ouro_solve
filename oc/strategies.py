"""
strategies.py

Implements all strategies. Each strategy is a callable:

    strategy(belief, clicks_left) -> cell_index

Strategies:
    0. baseline_center_random   — center first, then rule-informed random
    1. exact_pomdp              — provably optimal, memoized backward induction
    2. voi_greedy               — direct reward + expected future value, depth-2 lookahead
    3. entropy_minimization     — minimize expected posterior entropy over red's position
    4. candidate_halving        — minimize expected candidate set size (Mastermind-style)
"""

from __future__ import annotations
import math
import random
from functools import lru_cache
from typing import Dict, Tuple, FrozenSet

from oc.board_generator import (
    NUM_CELLS, CENTER, COLOR_NAMES, COLOR_VALUES,
    COLOR_RED, COLOR_ORANGE, COLOR_YELLOW, COLOR_GREEN, COLOR_TEAL, COLOR_BLUE,
)
from oc.belief_state import LightBeliefState, FullBeliefState, _red_candidates_after_reveal

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

def _unclicked(revealed: FrozenSet[int]) -> list:
    return [c for c in range(NUM_CELLS) if c not in revealed]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 0: Baseline — center first, then rule-informed random
# ─────────────────────────────────────────────────────────────────────────────

class BaselineCenterRandom:
    """
    Click center on first move.
    Subsequently pick uniformly at random from valid red candidates.
    Never clicks an eliminated cell.
    """
    name = "center_first_random"

    def __call__(self, belief: LightBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        if first_click:
            return CENTER
        candidates = list(belief.candidates - belief.revealed)
        if not candidates:
            # fallback: any unclicked cell
            candidates = _unclicked(belief.revealed)
        return random.choice(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Exact POMDP
# ─────────────────────────────────────────────────────────────────────────────

class ExactPOMDP:
    """
    Provably optimal strategy via backward induction over the full belief state.

    Two memo tables:
    - _value_memo  keyed on (board_indices, clicks_left)          — shareable across paths
    - _policy_memo keyed on (board_indices, revealed, clicks_left) — correct per revealed set
    """
    name = "exact_pomdp"

    def __init__(self):
        self._value_memo:  Dict[Tuple, float] = {}
        self._policy_memo: Dict[Tuple, int]   = {}

    def _vkey(self, belief: FullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _pkey(self, belief: FullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), belief.revealed, clicks_left)

    def value(self, belief: FullBeliefState, clicks_left: int) -> float:
        if clicks_left == 0:
            return 0.0
        if not belief.board_indices:
            return 0.0

        vkey = self._vkey(belief, clicks_left)
        pkey = self._pkey(belief, clicks_left)

        if vkey in self._value_memo and pkey in self._policy_memo:
            return self._value_memo[vkey]

        best_value = -1.0
        best_cell  = -1
        unclicked  = list(belief.unclicked())

        for cell in unclicked:
            ev = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue
                reward     = COLOR_VALUES[color]
                new_belief = belief.update(cell, color)
                future     = self.value(new_belief, clicks_left - 1)
                ev        += p * (reward + future)

            if ev > best_value:
                best_value = ev
                best_cell  = cell

        self._value_memo[vkey] = best_value
        self._policy_memo[pkey] = best_cell
        return best_value

    def __call__(self, belief: FullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        pkey = self._pkey(belief, clicks_left)
        if pkey not in self._policy_memo:
            self.value(belief, clicks_left)
        cell = self._policy_memo.get(pkey, -1)
        if cell == -1 or cell in belief.revealed:
            unclicked = list(belief.unclicked())
            return unclicked[0] if unclicked else 0
        return cell

    def precompute(self, initial_belief: FullBeliefState, max_clicks: int = 5):
        """Precompute full policy tree from initial belief state."""
        print("Precomputing POMDP policy tree...")
        v = self.value(initial_belief, max_clicks)
        print(f"Optimal expected score: {v:.4f}")
        print(f"Memo table size: {len(self._value_memo):,} states")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: VOI Greedy
# ─────────────────────────────────────────────────────────────────────────────

class VOIGreedy:
    """
    At each step, pick cell maximizing:
        immediate_expected_reward + expected_future_value

    Future value estimated via lookahead up to `depth` steps.
    Uses split memo keys — value on (board_indices, clicks_left),
    policy on (board_indices, revealed, clicks_left).
    """

    def __init__(self, depth: int = 2):
        self.depth = depth
        self.name  = f"voi_greedy_d{depth}"
        self._value_memo:  Dict[Tuple, float] = {}
        self._policy_memo: Dict[Tuple, int]   = {}

    def _vkey(self, belief: FullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _pkey(self, belief: FullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), belief.revealed, clicks_left)

    def _approx_future(self, belief: FullBeliefState, clicks_left: int) -> float:
        if clicks_left == 0:
            return 0.0
        red_cands = len(belief.red_candidates())
        if red_cands == 0:
            return 0.0
        p_find_red = min(1.0, clicks_left / red_cands)
        expected_post_red = 0.0
        if clicks_left >= 2:
            expected_post_red = p_find_red * (COLOR_VALUES[COLOR_ORANGE] + COLOR_VALUES[COLOR_YELLOW]) * min(1.0, (clicks_left - 1) / 2)
        return p_find_red * COLOR_VALUES[COLOR_RED] + expected_post_red

    def _value(self, belief: FullBeliefState, clicks_left: int,
               current_depth: int) -> float:
        if clicks_left == 0 or not belief.board_indices:
            return 0.0
        if current_depth >= self.depth:
            return self._approx_future(belief, clicks_left)

        vkey = self._vkey(belief, clicks_left)
        pkey = self._pkey(belief, clicks_left)

        if vkey in self._value_memo and pkey in self._policy_memo:
            return self._value_memo[vkey]

        best_value = 0.0
        best_cell  = -1
        for cell in belief.unclicked():
            ev = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue
                reward     = COLOR_VALUES[color]
                new_belief = belief.update(cell, color)
                future     = self._value(new_belief, clicks_left - 1,
                                         current_depth + 1)
                ev        += p * (reward + future)
            if ev > best_value:
                best_value = ev
                best_cell  = cell

        self._value_memo[vkey]  = best_value
        self._policy_memo[pkey] = best_cell
        return best_value

    def precompute(self, initial_belief: FullBeliefState, max_clicks: int = 5):
        print("Precomputing VOI policy tree...")
        v = self._value(initial_belief, max_clicks, current_depth=0)
        print(f"VOI expected score (approx): {v:.4f}")
        print(f"VOI memo table size: {len(self._value_memo):,} states")
        return v

    def __call__(self, belief: FullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        pkey = self._pkey(belief, clicks_left)
        if pkey not in self._policy_memo:
            self._value(belief, clicks_left, current_depth=0)
        cell = self._policy_memo.get(pkey, -1)
        if cell == -1 or cell in belief.revealed:
            unclicked = list(belief.unclicked())
            return unclicked[0] if unclicked else 0
        return cell


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Entropy Minimization
# ─────────────────────────────────────────────────────────────────────────────

class EntropyMinimization:
    """
    Pick the cell minimizing expected posterior entropy over red's position.
    Memoized on board_indices only. __call__ validates cell is unclicked.
    """
    name = "entropy_minimization"

    def __init__(self):
        self._policy_memo: Dict[tuple, int] = {}

    def _key(self, belief: FullBeliefState) -> tuple:
        return belief.key()

    def _best_cell(self, belief: FullBeliefState) -> int:
        light = belief.as_light()

        # red found and clicked — pivot to highest-value unclicked cell
        if light.is_red_found():
            unclicked = list(belief.unclicked())
            if not unclicked:
                return 0
            return max(unclicked, key=lambda c: belief.expected_reward(c))

        # red located but not yet clicked — click it immediately
        if light.is_red_located():
            only = next(iter(light.candidates))
            if only not in belief.revealed:
                return only
            # already clicked (shouldn't happen) — fall through to entropy
            unclicked = list(belief.unclicked())
            if not unclicked:
                return 0
            return max(unclicked, key=lambda c: belief.expected_reward(c))

        best_h    = float('inf')
        best_cell = -1
        for cell in belief.unclicked():
            h = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue
                k = len(belief.as_light().update(cell, color).candidates)
                h += p * (math.log2(k) if k > 0 else 0.0)
            if h < best_h:
                best_h    = h
                best_cell = cell
        return best_cell

    def precompute(self, initial_belief: FullBeliefState, max_clicks: int = 5):
        """Walk the full decision tree to populate memo table."""
        print("Precomputing entropy minimization policy tree...")
        self._walk(initial_belief, max_clicks)
        print(f"  Memo table: {len(self._policy_memo):,} states")

    def _walk(self, belief: FullBeliefState, clicks_left: int):
        if clicks_left == 0 or not belief.board_indices:
            return
        key = self._key(belief)
        if key in self._policy_memo:
            return
        cell = self._best_cell(belief)
        self._policy_memo[key] = cell
        for color in belief.possible_colors(cell):
            self._walk(belief.update(cell, color), clicks_left - 1)

    def __call__(self, belief: FullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        key = self._key(belief)
        if key not in self._policy_memo:
            self._policy_memo[key] = self._best_cell(belief)
        cell = self._policy_memo[key]
        if cell in belief.revealed:
            return self._best_cell(belief)
        return cell


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Candidate Halving (Mastermind-style)
# ─────────────────────────────────────────────────────────────────────────────

class CandidateHalving:
    """
    Pick the cell minimizing expected remaining red candidate set size.
    Memoized on board_indices only. __call__ validates cell is unclicked.
    """
    name = "candidate_halving"

    def __init__(self):
        self._policy_memo: Dict[tuple, int] = {}

    def _key(self, belief: FullBeliefState) -> tuple:
        return belief.key()

    def _best_cell(self, belief: FullBeliefState) -> int:
        light = belief.as_light()

        # red found and clicked — pivot to highest-value unclicked cell
        if light.is_red_found():
            unclicked = list(belief.unclicked())
            if not unclicked:
                return 0
            return max(unclicked, key=lambda c: belief.expected_reward(c))

        # red located but not yet clicked — click it immediately
        if light.is_red_located():
            only = next(iter(light.candidates))
            if only not in belief.revealed:
                return only
            unclicked = list(belief.unclicked())
            if not unclicked:
                return 0
            return max(unclicked, key=lambda c: belief.expected_reward(c))

        best_score = float('inf')
        best_cell  = -1
        for cell in belief.unclicked():
            score = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue
                score += p * len(belief.as_light().update(cell, color).candidates)
            if score < best_score:
                best_score = score
                best_cell  = cell
        return best_cell

    def precompute(self, initial_belief: FullBeliefState, max_clicks: int = 5):
        """Walk the full decision tree to populate memo table."""
        print("Precomputing candidate halving policy tree...")
        self._walk(initial_belief, max_clicks)
        print(f"  Memo table: {len(self._policy_memo):,} states")

    def _walk(self, belief: FullBeliefState, clicks_left: int):
        if clicks_left == 0 or not belief.board_indices:
            return
        key = self._key(belief)
        if key in self._policy_memo:
            return
        cell = self._best_cell(belief)
        self._policy_memo[key] = cell
        for color in belief.possible_colors(cell):
            self._walk(belief.update(cell, color), clicks_left - 1)

    def __call__(self, belief: FullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        key = self._key(belief)
        if key not in self._policy_memo:
            self._policy_memo[key] = self._best_cell(belief)
        cell = self._policy_memo[key]
        if cell in belief.revealed:
            return self._best_cell(belief)
        return cell


# ── strategy registry ─────────────────────────────────────────────────────────

def get_all_strategies():
    return [
        BaselineCenterRandom(),
        ExactPOMDP(),
        VOIGreedy(depth=2),
        EntropyMinimization(),
        CandidateHalving(),
    ]