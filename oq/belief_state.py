"""
belief_state_oq.py

OQ belief state implementation.

In OQ, revealed non-purple colors encode the exact count of purple cells among
8-directional Moore neighbors:
    blue=0, teal=1, green=2, yellow=3, orange=4.

Belief filtering is done by board consistency:
    observe (cell, color) -> keep boards where board[cell] == color.

Purple-to-red conversion after the 3rd purple is simulator-managed and is not
tracked in this belief state.

Speed optimisations vs original:
- Stores `_idx_arr` (np.int32) once per instance — eliminates the repeated
  frozenset → list → np.array conversion that dominated hot paths.
- update() filters via direct boolean indexing on _idx_arr and passes the
  resulting array to the child state, avoiding re-sort and re-conversion.
- peek() follows the same pattern.
- purple_candidates() / possible_purple_cells() use _idx_arr directly.
"""

from __future__ import annotations
from typing import FrozenSet, List, Optional
import numpy as np

from oq.board_generator import (
    NUM_CELLS,
    COLOR_VALUES,
    COLOR_PURPLE,
    moore_neighbors,
)


class OQFullBeliefState:
    """
    Tracks the full set of consistent OQ board indices.
    Requires ALL_BOARDS to be loaded once at class level.
    """

    # class-level board array and weights, set once via OQFullBeliefState.load_boards()
    ALL_BOARDS: np.ndarray = None
    ALL_WEIGHTS: np.ndarray = None
    NUM_BOARDS: int = 0

    @classmethod
    def load_boards(cls, boards: np.ndarray, weights: np.ndarray = None):
        cls.ALL_BOARDS = boards
        cls.NUM_BOARDS = len(boards)
        if weights is not None:
            cls.ALL_WEIGHTS = weights
        else:
            cls.ALL_WEIGHTS = np.ones(len(boards), dtype=np.float64) / len(boards)

    def __init__(self, board_indices: FrozenSet[int] = None,
                 revealed: FrozenSet[int] = None,
                 _idx_arr: Optional[np.ndarray] = None):
        if board_indices is None:
            board_indices = frozenset(range(self.NUM_BOARDS))
        self.board_indices: FrozenSet[int] = board_indices
        self.revealed: FrozenSet[int] = revealed or frozenset()

        # Cache the numpy index array — built once, reused by all query methods
        if _idx_arr is not None:
            self._idx_arr = _idx_arr
        else:
            self._idx_arr = np.array(sorted(board_indices), dtype=np.int32)

    def update(self, cell: int, color: int) -> OQFullBeliefState:
        """Filter to boards consistent with observing color at cell."""
        mask = self.ALL_BOARDS[self._idx_arr, cell] == color
        new_arr = self._idx_arr[mask]
        new_indices = frozenset(new_arr.tolist())
        new_revealed = self.revealed | {cell}
        return OQFullBeliefState(new_indices, new_revealed, _idx_arr=new_arr)

    def peek(self, cell: int, color: int) -> OQFullBeliefState:
        """Filter boards like update() but do not add cell to revealed.
        Used by simulator to reveal conversion target without marking it clicked."""
        mask = self.ALL_BOARDS[self._idx_arr, cell] == color
        new_arr = self._idx_arr[mask]
        new_indices = frozenset(new_arr.tolist())
        return OQFullBeliefState(new_indices, self.revealed, _idx_arr=new_arr)

    def p_color(self, cell: int, color: int) -> float:
        """P(cell = color | current belief) under weighted board distribution."""
        if self._idx_arr.size == 0:
            return 0.0
        w = self.ALL_WEIGHTS[self._idx_arr]
        w = w / w.sum()
        mask = self.ALL_BOARDS[self._idx_arr, cell] == color
        return float(w[mask].sum())

    def expected_reward(self, cell: int) -> float:
        """Expected immediate reward from revealing a cell under weighted distribution."""
        if self._idx_arr.size == 0:
            return 0.0
        w = self.ALL_WEIGHTS[self._idx_arr]
        w = w / w.sum()
        colors_at_cell = self.ALL_BOARDS[self._idx_arr, cell]
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

    def purple_candidates(self) -> FrozenSet[int]:
        """Cells that are purple in ALL consistent boards."""
        if self._idx_arr.size == 0:
            return frozenset()
        purple_mask = (self.ALL_BOARDS[self._idx_arr] == COLOR_PURPLE)
        all_purple = purple_mask.all(axis=0)
        return frozenset(np.where(all_purple)[0].tolist())

    def possible_purple_cells(self) -> FrozenSet[int]:
        """Cells that are purple in ANY consistent board."""
        if self._idx_arr.size == 0:
            return frozenset()
        purple_mask = (self.ALL_BOARDS[self._idx_arr] == COLOR_PURPLE)
        any_purple = purple_mask.any(axis=0)
        return frozenset(np.where(any_purple)[0].tolist())

    def key(self) -> tuple:
        """Hashable key for memoization."""
        return self.board_indices

    def __repr__(self):
        return (f"OQFullBeliefState(consistent={len(self.board_indices)}, "
                f"revealed={len(self.revealed)})")
