"""
belief_state.py

Two belief state implementations:

LightBeliefState  — tracks only which cells are valid red candidates.
                    Used by entropy minimization and candidate halving.

FullBeliefState   — tracks the full set of consistent board indices.
                    Used by POMDP and VOI greedy.

Speed optimisations vs original:
- FullBeliefState stores `_idx_arr` (np.int32 array) alongside `board_indices`
  (frozenset), eliminating the repeated frozenset → list → np.array conversion
  that was the hot path in update(), p_color(), expected_reward(), possible_colors().
- `_idx_arr` is computed once at construction and reused by all query methods.
- `update()` builds the filtered array directly with boolean indexing, then
  derives `board_indices` from it (frozenset only needed for memoisation keys).
- `_red_candidates_after_reveal` precomputes geometry sets once per cell to
  avoid reconstructing them inside set comprehensions.
"""

from __future__ import annotations
from typing import FrozenSet, Dict, List
import numpy as np

from oc.board_generator import (
    NUM_CELLS, CENTER, COLOR_NAMES, COLOR_VALUES,
    COLOR_RED, COLOR_ORANGE, COLOR_YELLOW, COLOR_GREEN, COLOR_TEAL, COLOR_BLUE,
    rc, immediate_neighbors, full_diagonal_cells, same_row_col_cells,
)

# ── Precompute geometry sets for all cells once at import time ────────────────

_IMMEDIATE_NEIGHBORS: List[FrozenSet[int]] = [
    frozenset(immediate_neighbors(c)) for c in range(NUM_CELLS)
]
_FULL_DIAGONAL_CELLS: List[FrozenSet[int]] = [
    frozenset(full_diagonal_cells(c)) for c in range(NUM_CELLS)
]
_ROW_COL_CELLS: List[FrozenSet[int]] = [
    frozenset(same_row_col_cells(c)) for c in range(NUM_CELLS)
]
# Precompute (row, col) for each cell
_RC: List[tuple] = [rc(c) for c in range(NUM_CELLS)]

# ── deduction: given a revealed cell and color, which cells can still be red? ─

def _red_candidates_after_reveal(revealed_cell: int, color: int,
                                  current_candidates: FrozenSet[int]) -> FrozenSet[int]:
    """
    Apply deduction rules to filter red candidates.
    Always removes revealed_cell and CENTER from candidates.
    Uses precomputed geometry sets for O(1) lookup.
    """
    cands = set(current_candidates)
    cands.discard(revealed_cell)
    cands.discard(CENTER)

    if color == COLOR_ORANGE:
        cands &= _IMMEDIATE_NEIGHBORS[revealed_cell]

    elif color == COLOR_YELLOW:
        cands &= _FULL_DIAGONAL_CELLS[revealed_cell]

    elif color == COLOR_GREEN:
        rv, cv = _RC[revealed_cell]
        cands = {p for p in cands if _RC[p][0] == rv or _RC[p][1] == cv}

    elif color == COLOR_TEAL:
        rv, cv = _RC[revealed_cell]
        cands = {p for p in cands
                 if _RC[p][0] == rv
                 or _RC[p][1] == cv
                 or abs(_RC[p][0] - rv) == abs(_RC[p][1] - cv)}

    elif color == COLOR_BLUE:
        rv, cv = _RC[revealed_cell]
        cands = {p for p in cands
                 if _RC[p][0] != rv
                 and _RC[p][1] != cv
                 and abs(_RC[p][0] - rv) != abs(_RC[p][1] - cv)}

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

    Key optimisation: stores `_idx_arr` (np.int32) once per state so that
    p_color(), expected_reward(), possible_colors(), and update() can use
    vectorised NumPy operations directly without frozenset→list→array conversion.
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
            cls.ALL_WEIGHTS = np.ones(len(boards), dtype=np.float64) / len(boards)

    def __init__(self, board_indices: FrozenSet[int] = None,
                 revealed: FrozenSet[int] = None,
                 _idx_arr: np.ndarray = None):
        if board_indices is None:
            board_indices = frozenset(range(self.NUM_BOARDS))
        self.board_indices: FrozenSet[int] = board_indices
        self.revealed:      FrozenSet[int] = revealed or frozenset()

        # Cache the numpy index array — built once, reused by all query methods
        if _idx_arr is not None:
            self._idx_arr = _idx_arr
        else:
            self._idx_arr = np.array(sorted(board_indices), dtype=np.int32)

    def update(self, cell: int, color: int) -> FullBeliefState:
        """Filter to boards consistent with observing color at cell."""
        mask = self.ALL_BOARDS[self._idx_arr, cell] == color
        new_arr = self._idx_arr[mask]
        new_indices = frozenset(new_arr.tolist())
        new_revealed = self.revealed | {cell}
        return FullBeliefState(new_indices, new_revealed, _idx_arr=new_arr)

    def p_color(self, cell: int, color: int) -> float:
        """P(cell = color | current belief) under weighted board distribution."""
        if self._idx_arr.size == 0:
            return 0.0
        w = self.ALL_WEIGHTS[self._idx_arr]
        w = w / w.sum()
        mask = self.ALL_BOARDS[self._idx_arr, cell] == color
        return float(w[mask].sum())

    def expected_reward(self, cell: int) -> float:
        """Expected immediate reward from clicking cell under weighted distribution."""
        if self._idx_arr.size == 0:
            return 0.0
        w = self.ALL_WEIGHTS[self._idx_arr]
        w = w / w.sum()
        colors_at_cell = self.ALL_BOARDS[self._idx_arr, cell]
        rewards = np.empty(len(colors_at_cell), dtype=np.float64)
        cv = np.array(COLOR_VALUES, dtype=np.float64)
        rewards = cv[colors_at_cell]
        return float((rewards * w).sum())

    def possible_colors(self, cell: int) -> List[int]:
        """Colors that appear at cell in at least one consistent board."""
        if self._idx_arr.size == 0:
            return []
        return list(np.unique(self.ALL_BOARDS[self._idx_arr, cell]))

    def unclicked(self) -> FrozenSet[int]:
        return frozenset(range(NUM_CELLS)) - self.revealed

    def consistent_count(self) -> int:
        return len(self.board_indices)

    def red_candidates(self) -> FrozenSet[int]:
        """Cells where red could still be, derived from consistent boards."""
        if self._idx_arr.size == 0:
            return frozenset()
        red_positions = self.ALL_BOARDS[self._idx_arr, :] == COLOR_RED
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