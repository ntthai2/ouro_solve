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
"""

from __future__ import annotations
from typing import FrozenSet, List
import numpy as np

from oq_board_generator import (
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
                 revealed: FrozenSet[int] = None):
        if board_indices is None:
            board_indices = frozenset(range(self.NUM_BOARDS))
        self.board_indices: FrozenSet[int] = board_indices
        self.revealed: FrozenSet[int] = revealed or frozenset()

    def update(self, cell: int, color: int) -> OQFullBeliefState:
        """Filter to boards consistent with observing color at cell."""
        boards = self.ALL_BOARDS
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        mask = boards[idx_arr, cell] == color
        new_indices = frozenset(idx_arr[mask].tolist())
        new_revealed = self.revealed | {cell}
        return OQFullBeliefState(new_indices, new_revealed)

    def peek(self, cell: int, color: int) -> OQFullBeliefState:
        """Filter boards like update() but do not add cell to revealed.
        Used by simulator to reveal conversion target without marking it clicked."""
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        mask = self.ALL_BOARDS[idx_arr, cell] == color
        new_indices = frozenset(idx_arr[mask].tolist())
        return OQFullBeliefState(new_indices, self.revealed)

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
        """Expected immediate reward from revealing a cell under weighted distribution."""
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

    def purple_candidates(self) -> FrozenSet[int]:
        """Cells that are purple in ALL consistent boards."""
        if not self.board_indices:
            return frozenset()
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        purple_mask = (self.ALL_BOARDS[idx_arr] == COLOR_PURPLE)
        all_purple = purple_mask.all(axis=0)
        return frozenset(np.where(all_purple)[0].tolist())

    def possible_purple_cells(self) -> FrozenSet[int]:
        """Cells that are purple in ANY consistent board."""
        if not self.board_indices:
            return frozenset()
        idx_arr = np.array(list(self.board_indices), dtype=np.int32)
        purple_mask = (self.ALL_BOARDS[idx_arr] == COLOR_PURPLE)
        any_purple = purple_mask.any(axis=0)
        return frozenset(np.where(any_purple)[0].tolist())

    def key(self) -> tuple:
        """Hashable key for memoization."""
        return self.board_indices

    def __repr__(self):
        return (f"OQFullBeliefState(consistent={len(self.board_indices)}, "
                f"revealed={len(self.revealed)})")
