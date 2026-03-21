"""
belief_state.py

Two belief state implementations:

LightBeliefState  — tracks only which cells are valid red candidates.
                    Used by entropy minimization and candidate halving.

FullBeliefState   — tracks the full set of consistent board indices.
                    Used by POMDP and VOI greedy.
"""

from __future__ import annotations
from typing import FrozenSet, Dict, List
import numpy as np

from oc_board_generator import (
    NUM_CELLS, CENTER, COLOR_NAMES, COLOR_VALUES,
    COLOR_RED, COLOR_ORANGE, COLOR_YELLOW, COLOR_GREEN, COLOR_TEAL, COLOR_BLUE,
    rc, immediate_neighbors, full_diagonal_cells, same_row_col_cells,
)

# ── deduction: given a revealed cell and color, which cells can still be red? ─

def _red_candidates_after_reveal(revealed_cell: int, color: int,
                                  current_candidates: FrozenSet[int]) -> FrozenSet[int]:
    """
    Apply deduction rules to filter red candidates.
    Always removes revealed_cell and CENTER from candidates.
    """
    r, c = rc(revealed_cell)
    cands = set(current_candidates)
    cands.discard(revealed_cell)
    cands.discard(CENTER)

    if color == COLOR_ORANGE:
        # red is an immediate neighbor of revealed cell
        valid = set(immediate_neighbors(revealed_cell))
        cands &= valid

    elif color == COLOR_YELLOW:
        # red is on the full diagonal lines of revealed cell
        valid = set(full_diagonal_cells(revealed_cell))
        cands &= valid

    elif color == COLOR_GREEN:
        # red shares row or column with revealed cell
        rv, cv = rc(revealed_cell)
        cands = {p for p in cands
                 if rc(p)[0] == rv or rc(p)[1] == cv}

    elif color == COLOR_TEAL:
        # red shares row, col, or diagonal with revealed cell
        rv, cv = rc(revealed_cell)
        cands = {p for p in cands
                 if rc(p)[0] == rv
                 or rc(p)[1] == cv
                 or abs(rc(p)[0] - rv) == abs(rc(p)[1] - cv)}

    elif color == COLOR_BLUE:
        # red shares NOTHING with revealed cell
        rv, cv = rc(revealed_cell)
        cands = {p for p in cands
                 if rc(p)[0] != rv
                 and rc(p)[1] != cv
                 and abs(rc(p)[0] - rv) != abs(rc(p)[1] - cv)}

    # COLOR_RED: red is found, candidate set collapses to revealed_cell
    elif color == COLOR_RED:
        cands = {revealed_cell}

    return frozenset(cands)


# ── initial candidates ────────────────────────────────────────────────────────

def _initial_candidates() -> FrozenSet[int]:
    """All cells except center are valid red candidates initially."""
    return frozenset(i for i in range(NUM_CELLS) if i != CENTER)


# ── LightBeliefState ──────────────────────────────────────────────────────────

class LightBeliefState:
    """
    Tracks only which cells are valid red candidates.
    Efficient for entropy minimization and candidate halving.
    """

    def __init__(self, candidates: FrozenSet[int] = None,
                 revealed: FrozenSet[int] = None):
        self.candidates: FrozenSet[int] = candidates or _initial_candidates()
        self.revealed:   FrozenSet[int] = revealed   or frozenset()

    def update(self, cell: int, color: int) -> LightBeliefState:
        """Return new belief state after observing color at cell."""
        new_cands   = _red_candidates_after_reveal(cell, color, self.candidates)
        new_revealed = self.revealed | {cell}
        return LightBeliefState(new_cands, new_revealed)

    def unclicked(self) -> FrozenSet[int]:
        return frozenset(range(NUM_CELLS)) - self.revealed

    def candidate_count(self) -> int:
        return len(self.candidates)

    def is_red_found(self) -> bool:
        """
        Red is 'found' when its position is known with certainty —
        i.e. exactly one candidate remains AND it has been revealed (clicked).
        Before clicking it, red is merely 'located' (one candidate, unclicked).
        """
        if len(self.candidates) != 1:
            return False
        only = next(iter(self.candidates))
        return only in self.revealed

    def is_red_located(self) -> bool:
        """Red's position is known but not yet clicked."""
        return len(self.candidates) == 1

    def __repr__(self):
        return f"LightBeliefState(candidates={len(self.candidates)}, revealed={len(self.revealed)})"


# ── FullBeliefState ───────────────────────────────────────────────────────────

class FullBeliefState:
    """
    Tracks the full set of consistent board indices.
    Used by POMDP and VOI greedy.
    Requires ALL_BOARDS to be loaded once at module level.
    """

    # class-level board array and weights, set once via FullBeliefState.load_boards()
    ALL_BOARDS:   np.ndarray = None
    ALL_WEIGHTS:  np.ndarray = None  # per-board weights for correct prior
    NUM_BOARDS:   int = 0

    @classmethod
    def load_boards(cls, boards: np.ndarray, weights: np.ndarray = None):
        cls.ALL_BOARDS  = boards
        cls.NUM_BOARDS  = len(boards)
        if weights is not None:
            cls.ALL_WEIGHTS = weights
        else:
            # uniform over boards (biased — only use for testing)
            cls.ALL_WEIGHTS = np.ones(len(boards), dtype=np.float64) / len(boards)

    def __init__(self, board_indices: FrozenSet[int] = None,
                 revealed: FrozenSet[int] = None):
        if board_indices is None:
            board_indices = frozenset(range(self.NUM_BOARDS))
        self.board_indices: FrozenSet[int] = board_indices
        self.revealed:      FrozenSet[int] = revealed or frozenset()

    def update(self, cell: int, color: int) -> FullBeliefState:
        """Filter to boards consistent with observing color at cell."""
        boards = self.ALL_BOARDS
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        mask = boards[idx_arr, cell] == color
        new_indices = frozenset(idx_arr[mask].tolist())
        new_revealed = self.revealed | {cell}
        return FullBeliefState(new_indices, new_revealed)

    def p_color(self, cell: int, color: int) -> float:
        """P(cell = color | current belief) under weighted board distribution."""
        if not self.board_indices:
            return 0.0
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        w = self.ALL_WEIGHTS[idx_arr]
        w = w / w.sum()
        mask = self.ALL_BOARDS[idx_arr, cell] == color
        return float(w[mask].sum())

    def expected_reward(self, cell: int) -> float:
        """Expected immediate reward from clicking cell under weighted distribution."""
        if not self.board_indices:
            return 0.0
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        w = self.ALL_WEIGHTS[idx_arr]
        w = w / w.sum()
        rewards = np.array([COLOR_VALUES[c] for c in self.ALL_BOARDS[idx_arr, cell]])
        return float((rewards * w).sum())

    def possible_colors(self, cell: int) -> List[int]:
        """Colors that appear at cell in at least one consistent board."""
        if not self.board_indices:
            return []
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        return list(np.unique(self.ALL_BOARDS[idx_arr, cell]))

    def unclicked(self) -> FrozenSet[int]:
        return frozenset(range(NUM_CELLS)) - self.revealed

    def consistent_count(self) -> int:
        return len(self.board_indices)

    def red_candidates(self) -> FrozenSet[int]:
        """Cells where red could still be, derived from consistent boards."""
        if not self.board_indices:
            return frozenset()
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        red_positions = self.ALL_BOARDS[idx_arr, :] == COLOR_RED
        possible_red_cells = np.where(red_positions.any(axis=0))[0]
        return frozenset(possible_red_cells.tolist())

    def as_light(self) -> LightBeliefState:
        """Downcast to LightBeliefState for hybrid strategies."""
        return LightBeliefState(self.red_candidates(), self.revealed)

    def key(self) -> tuple:
        """Hashable key for memoization."""
        return self.board_indices

    def __repr__(self):
        return (f"FullBeliefState(consistent={len(self.board_indices)}, "
                f"revealed={len(self.revealed)})")