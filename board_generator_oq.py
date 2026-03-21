"""
board_generator_oq.py

Exhaustively enumerates all valid OQ board configurations.

Board is 5x5 = 25 cells, indexed 0..24.
Cell index = row * 5 + col, both 0-indexed.

Color encoding (int for speed):
    0 = blue
    1 = teal
    2 = green
    3 = yellow
    4 = orange
    5 = purple
    6 = red (runtime only, never appears in generated boards)
"""

from itertools import combinations
from typing import List, Tuple
import os

import numpy as np

# -- constants -----------------------------------------------------------------

GRID_SIZE = 5
NUM_CELLS = 25
NUM_PURPLES = 4

COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE = 0, 1, 2, 3, 4
COLOR_PURPLE = 5
COLOR_RED = 6  # runtime only, never appears in generated boards

COLOR_NAMES = ['blue', 'teal', 'green', 'yellow', 'orange', 'purple', 'red']
COLOR_VALUES = [10, 20, 35, 55, 90, 5, 150]


# -- geometry helpers -----------------------------------------------------------

def rc(cell: int) -> Tuple[int, int]:
    return cell // GRID_SIZE, cell % GRID_SIZE


def cell(r: int, c: int) -> int:
    return r * GRID_SIZE + c


def moore_neighbors(idx: int) -> List[int]:
    """All valid 8-directional neighbors within grid boundaries."""
    r, c = rc(idx)
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                result.append(cell(nr, nc))
    return result


# -- board construction ---------------------------------------------------------

def build_board(purple_cells: Tuple[int, ...]) -> np.ndarray:
    """
    Given exactly 4 purple positions as a sorted tuple, return a length-25
    uint8 array where:
      - purple cells get ID 5
      - all other cells get ID = count of Moore neighbors that are purple
        (0->blue, 1->teal, 2->green, 3->yellow, 4->orange)
    """
    if len(purple_cells) != NUM_PURPLES:
        raise ValueError(f"Expected exactly {NUM_PURPLES} purple cells, got {len(purple_cells)}")
    if tuple(sorted(purple_cells)) != purple_cells:
        raise ValueError("purple_cells must be provided as a sorted tuple")

    purple_set = set(purple_cells)
    board = np.zeros(NUM_CELLS, dtype=np.uint8)

    for idx in range(NUM_CELLS):
        if idx in purple_set:
            board[idx] = COLOR_PURPLE
            continue

        purple_neighbor_count = sum(1 for n in moore_neighbors(idx) if n in purple_set)
        if purple_neighbor_count > COLOR_ORANGE:
            raise ValueError(
                f"Cell {idx} has {purple_neighbor_count} purple neighbors; exceeds max encodable color 4"
            )
        board[idx] = np.uint8(purple_neighbor_count)

    return board


# -- board enumeration ----------------------------------------------------------

def enumerate_boards() -> np.ndarray:
    """
    Returns all valid boards as a numpy array of shape (12650, 25), dtype uint8.
    Each row is a board; values are color ints (0..5).
    """
    boards = [build_board(purple_cells) for purple_cells in combinations(range(NUM_CELLS), NUM_PURPLES)]
    return np.array(boards, dtype=np.uint8)


def get_purple_positions(board: np.ndarray) -> Tuple[int, ...]:
    """Return sorted tuple of indices where board value == COLOR_PURPLE."""
    return tuple(np.where(board == COLOR_PURPLE)[0].tolist())


if __name__ == '__main__':
    os.makedirs('cache', exist_ok=True)

    print('Enumerating all valid OQ boards...')
    boards = enumerate_boards()
    np.save('cache/all_boards_oq.npy', boards)

    print(f"Saved to cache/all_boards_oq.npy")
    print(f"Shape: {boards.shape}")

    # quick sanity checks
    assert boards.shape == (12650, NUM_CELLS), f"Unexpected shape: {boards.shape}"
    assert (boards == COLOR_PURPLE).sum(axis=1).min() == NUM_PURPLES
    assert (boards == COLOR_PURPLE).sum(axis=1).max() == NUM_PURPLES

    sample = boards[0].reshape(GRID_SIZE, GRID_SIZE)
    print('\nSample board (5x5, color IDs):')
    for r in range(GRID_SIZE):
        print('  ' + ' '.join(f"{int(v):d}" for v in sample[r]))
