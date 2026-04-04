"""
strategies_oq.py

Implements OQ strategies as callables:

    strategy(belief, clicks_left) -> cell_index

Included strategies:
    - OQExactPOMDP
    - OQVOIGreedy

OQ mechanics encoded here:
    - up to 7 paid clicks
    - purple (ID 5) reveals are free (do not decrement paid clicks)
    - once 3 purples are found, the final purple is treated as converted red
      reward (150) and should be clicked immediately when known.
"""

from __future__ import annotations
from typing import Dict, Tuple, FrozenSet

from oq.board_generator import (
    NUM_CELLS,
    COLOR_VALUES,
    COLOR_PURPLE,
    COLOR_RED,
)
from oq.belief_state import OQFullBeliefState

MAX_CLICKS = 7


# -- helpers -------------------------------------------------------------------

def _unclicked(revealed: FrozenSet[int]) -> list:
    return [c for c in range(NUM_CELLS) if c not in revealed]


def _purples_found(belief: OQFullBeliefState) -> int:
    return len(belief.purple_candidates() & belief.revealed)


def _conversion_target_if_known(belief: OQFullBeliefState) -> int | None:
    """
    Returns the known unclicked conversion target if conversion is active,
    else None.
    """
    purples_found = _purples_found(belief)
    if purples_found != 3:
        return None

    unclicked_purples = belief.possible_purple_cells() - belief.revealed
    if len(unclicked_purples) == 1:
        return next(iter(unclicked_purples))
    return None


def _purple_cascade_bonus(purples_found: int) -> float:
    """Heuristic bonus for progress toward purple-triggered red conversion."""
    if purples_found == 2:
        return 150.0
    if purples_found == 1:
        return 125.0    # was 75
    return 80.0     # was 40


# -----------------------------------------------------------------------------
# Strategy 1: Exact POMDP (OQ)
# -----------------------------------------------------------------------------

class OQExactPOMDP:
    """
    Provably optimal strategy via backward induction over the OQ full belief.

    Split memo tables (mirrors existing strategy structure):
    - _value_memo  keyed on (board_indices, clicks_left)
    - _policy_memo keyed on (board_indices, revealed, clicks_left)
    """

    name = "exact_pomdp_oq"

    def __init__(self):
        self._value_memo: Dict[Tuple, float] = {}
        self._policy_memo: Dict[Tuple, int] = {}

    def _vkey(self, belief: OQFullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _pkey(self, belief: OQFullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _effective_reward(self, cell: int, color: int, belief: OQFullBeliefState) -> float:
        if color == COLOR_PURPLE:
            target = _conversion_target_if_known(belief)
            if target is not None and cell == target:
                return float(COLOR_VALUES[COLOR_RED])
        return float(COLOR_VALUES[color])

    def value(self, belief: OQFullBeliefState, clicks_left: int) -> float:
        if clicks_left == 0:
            return 0.0
        if not belief.board_indices:
            return 0.0

        vkey = self._vkey(belief, clicks_left)
        pkey = self._pkey(belief, clicks_left)

        if vkey in self._value_memo and pkey in self._policy_memo:
            return self._value_memo[vkey]

        best_value = -1.0
        best_cell = -1
        unclicked = list(belief.unclicked())

        for cell in unclicked:
            ev = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue

                reward = self._effective_reward(cell, color, belief)
                new_belief = belief.update(cell, color)

                if color == COLOR_PURPLE:
                    future = self.value(new_belief, clicks_left)
                else:
                    future = self.value(new_belief, clicks_left - 1)

                ev += p * (reward + future)

            if ev > best_value:
                best_value = ev
                best_cell = cell

        self._value_memo[vkey] = best_value
        self._policy_memo[pkey] = best_cell
        return best_value

    def __call__(self, belief: OQFullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        target = _conversion_target_if_known(belief)
        if target is not None:
            return target

        pkey = self._pkey(belief, clicks_left)
        if pkey not in self._policy_memo:
            self.value(belief, clicks_left)

        cell = self._policy_memo.get(pkey, -1)
        if cell == -1 or cell in belief.revealed:
            unclicked = list(belief.unclicked())
            return unclicked[0] if unclicked else 0
        return cell

    def precompute(self, initial_belief: OQFullBeliefState, max_clicks: int = MAX_CLICKS):
        print("Precomputing OQ POMDP policy tree...")
        v = self.value(initial_belief, max_clicks)
        print(f"Optimal expected score (OQ): {v:.4f}")
        print(f"Memo table size: {len(self._value_memo):,} states")
        return v


# -----------------------------------------------------------------------------
# Strategy 2: VOI Greedy (OQ)
# -----------------------------------------------------------------------------

class OQVOIGreedy:
    """
    Greedy VOI strategy with depth-limited lookahead.

    Future fallback at depth limit uses a greedy estimate from expected_reward.
    Free-purple click and converted-red reward rules match OQExactPOMDP.
    """

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.name = f"voi_greedy_oq_d{depth}"
        self._value_memo: Dict[Tuple, float] = {}
        self._policy_memo: Dict[Tuple, int] = {}

    def _vkey(self, belief: OQFullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _pkey(self, belief: OQFullBeliefState, clicks_left: int) -> Tuple:
        return (belief.key(), clicks_left)

    def _effective_reward(self, cell: int, color: int, belief: OQFullBeliefState) -> float:
        if color == COLOR_PURPLE:
            target = _conversion_target_if_known(belief)
            if target is not None and cell == target:
                return float(COLOR_VALUES[COLOR_RED])
        return float(COLOR_VALUES[color])

    def _approx_future(self, belief: OQFullBeliefState, clicks_left: int) -> float:
        if clicks_left == 0:
            return 0.0

        target = _conversion_target_if_known(belief)
        if target is not None:
            return float(COLOR_VALUES[COLOR_RED])

        unclicked = list(belief.unclicked())
        if not unclicked:
            return 0.0

        # Greedy fallback: estimate paid-click horizon by summing top expected
        # immediate rewards over distinct unclicked cells.
        rewards = sorted((belief.expected_reward(c) for c in unclicked), reverse=True)
        take = min(clicks_left, len(rewards))
        return float(sum(rewards[:take]))

    def _value(self, belief: OQFullBeliefState, clicks_left: int,
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
        best_cell = -1

        for cell in belief.unclicked():
            ev = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue

                reward = self._effective_reward(cell, color, belief)
                new_belief = belief.update(cell, color)

                if color == COLOR_PURPLE:
                    future = self._value(new_belief, clicks_left, current_depth + 1)
                else:
                    future = self._value(new_belief, clicks_left - 1, current_depth + 1)

                ev += p * (reward + future)

            if ev > best_value:
                best_value = ev
                best_cell = cell

        self._value_memo[vkey] = best_value
        self._policy_memo[pkey] = best_cell
        return best_value

    def count_states(self, belief: OQFullBeliefState, clicks_left: int,
                     visited: set[Tuple] | None = None):
        """Dry-run reachability walk for OQ transitions (no memo writes)."""
        if visited is None:
            visited = set()

        key = self._vkey(belief, clicks_left)
        if key in visited:
            return visited
        if clicks_left == 0 or belief.consistent_count() == 0:
            return visited

        visited.add(key)

        for cell in belief.unclicked():
            for color in belief.possible_colors(cell):
                new_belief = belief.update(cell, color)
                if color == COLOR_PURPLE:
                    self.count_states(new_belief, clicks_left, visited)
                else:
                    self.count_states(new_belief, clicks_left - 1, visited)

        return visited

    def precompute(self, initial_belief: OQFullBeliefState, max_clicks: int = MAX_CLICKS):
        print("Precomputing OQ VOI policy tree...")
        v = self._value(initial_belief, max_clicks, current_depth=0)
        print(f"VOI expected score (OQ, approx): {v:.4f}")
        print(f"VOI memo table size: {len(self._policy_memo):,} states")

    def __call__(self, belief: OQFullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        target = _conversion_target_if_known(belief)
        if target is not None:
            return target

        pkey = self._pkey(belief, clicks_left)
        if pkey in self._policy_memo:
            cell = self._policy_memo[pkey]
            if cell not in belief.revealed:
                return cell

        purples_found = len(belief.purple_candidates() & belief.revealed)
        best_cell = -1
        best_ev = -1.0
        for cell in belief.unclicked():
            ev = 0.0
            for color in belief.possible_colors(cell):
                p = belief.p_color(cell, color)
                if p == 0.0:
                    continue
                if color == COLOR_PURPLE:
                    bonus = _purple_cascade_bonus(purples_found)
                    ev += p * (COLOR_VALUES[COLOR_PURPLE] + bonus)
                else:
                    ev += p * self._effective_reward(cell, color, belief)
            if ev > best_ev:
                best_ev = ev
                best_cell = cell

        if best_cell != -1:
            return best_cell
        unclicked = list(belief.unclicked())
        return unclicked[0] if unclicked else 0


# -- Purple-First Greedy Strategy --------------------------------------------------

def _purple_first_greedy(belief: OQFullBeliefState) -> int:
    """
    Purple-first greedy: seek purples until 3 are found, then maximize reward.
    """
    purples_found = len(belief.purple_candidates() & belief.revealed)

    if purples_found < 3:
        # phase 1: click cell with highest P(purple)
        best_cell = -1
        best_p = -1.0
        for cell in belief.unclicked():
            p = belief.p_color(cell, COLOR_PURPLE)
            if p > best_p:
                best_p = p
                best_cell = cell
        if best_cell != -1:
            return best_cell
        # fallback if no purple found: use expected reward
        best_cell = -1
        best_ev = -1.0
        for cell in belief.unclicked():
            ev = belief.expected_reward(cell)
            if ev > best_ev:
                best_ev = ev
                best_cell = cell
        return best_cell if best_cell != -1 else next(iter(belief.unclicked()))
    else:
        # phase 2: 3 purples found, conversion happened — click highest expected reward
        best_cell = -1
        best_ev = -1.0
        for cell in belief.unclicked():
            ev = belief.expected_reward(cell)
            if ev > best_ev:
                best_ev = ev
                best_cell = cell
        return best_cell if best_cell != -1 else next(iter(belief.unclicked()))


class OQPurpleFirstGreedy:
    """
    Simple strategy: prioritize purple-finding until conversion, then maximize reward.
    """
    name = "purple_first_greedy_oq"

    def __call__(self, belief: OQFullBeliefState, clicks_left: int,
                 first_click: bool = False) -> int:
        target = _conversion_target_if_known(belief)
        if target is not None:
            return target
        return _purple_first_greedy(belief)

    def precompute(self, initial_belief: OQFullBeliefState, max_clicks: int = MAX_CLICKS):
        pass  # no precompute needed
