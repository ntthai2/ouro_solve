"""
board_generator.py

Exhaustively enumerates all valid Ourosphere board configurations.

Board is 5x5 = 25 cells, indexed 0..24.
Cell index = row * 5 + col, both 0-indexed.
Center = cell 12 (row 2, col 2). Red is never placed here.

Color encoding (int for speed):
    0 = blue
    1 = teal
    2 = green
    3 = yellow
    4 = orange
    5 = red
"""

from itertools import combinations
from typing import List, Tuple
import numpy as np
import torch

# ── constants ────────────────────────────────────────────────────────────────

GRID_SIZE   = 5
NUM_CELLS   = 25
CENTER      = 12  # (2,2)

COLOR_BLUE   = 0
COLOR_TEAL   = 1
COLOR_GREEN  = 2
COLOR_YELLOW = 3
COLOR_ORANGE = 4
COLOR_RED    = 5

COLOR_NAMES  = ['blue', 'teal', 'green', 'yellow', 'orange', 'red']
COLOR_VALUES = [10, 20, 35, 55, 90, 150]

# ── geometry helpers ──────────────────────────────────────────────────────────

def rc(cell: int) -> Tuple[int, int]:
    return cell // GRID_SIZE, cell % GRID_SIZE

def cell(r: int, c: int) -> int:
    return r * GRID_SIZE + c

def immediate_neighbors(idx: int) -> List[int]:
    """4-directional immediate neighbors within grid."""
    r, c = rc(idx)
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            result.append(cell(nr, nc))
    return result

def full_diagonal_cells(idx: int) -> List[int]:
    """All cells on the diagonal lines through idx, excluding idx itself."""
    r, c = rc(idx)
    result = []
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r+dr, c+dc
        while 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            result.append(cell(nr, nc))
            nr += dr
            nc += dc
    return result

def same_row_col_cells(idx: int) -> List[int]:
    """All cells in same row or column as idx, excluding idx itself."""
    r, c = rc(idx)
    result = []
    for cc in range(GRID_SIZE):
        if cc != c:
            result.append(cell(r, cc))
    for rr in range(GRID_SIZE):
        if rr != r:
            result.append(cell(rr, c))
    return result

def teal_cells(red_idx: int, occupied: set) -> List[int]:
    """
    Cells sharing row, col, or diagonal with red,
    excluding red itself and already occupied cells.
    """
    r, c = rc(red_idx)
    result = []
    for idx in range(NUM_CELLS):
        if idx in occupied:
            continue
        ir, ic = rc(idx)
        if ir == r or ic == c or abs(ir - r) == abs(ic - c):
            result.append(idx)
    return result

def blue_cells(red_idx: int, occupied: set) -> List[int]:
    """
    Cells sharing nothing with red (not same row, col, or diagonal),
    excluding occupied cells.
    """
    r, c = rc(red_idx)
    result = []
    for idx in range(NUM_CELLS):
        if idx in occupied:
            continue
        ir, ic = rc(idx)
        if ir != r and ic != c and abs(ir - r) != abs(ic - c):
            result.append(idx)
    return result

# ── board enumeration ─────────────────────────────────────────────────────────

def enumerate_boards() -> np.ndarray:
    """
    Returns all valid boards as a numpy array of shape (N, 25) dtype uint8.
    Each row is a board; values are color ints (0..5).
    """
    boards = []

    for red_pos in range(NUM_CELLS):
        if red_pos == CENTER:
            continue

        # orange: choose 2 from immediate neighbors
        orange_pool = immediate_neighbors(red_pos)
        if len(orange_pool) < 2:
            continue  # should never happen on a 5x5

        for orange_pair in combinations(orange_pool, 2):
            occupied = {red_pos, *orange_pair}

            # yellow: choose 3 from full diagonal lines, excluding occupied
            yellow_pool = [c for c in full_diagonal_cells(red_pos) if c not in occupied]
            if len(yellow_pool) < 3:
                continue

            for yellow_triple in combinations(yellow_pool, 3):
                occupied2 = occupied | set(yellow_triple)

                # green: choose 4 from same row+col, excluding occupied
                green_pool = [c for c in same_row_col_cells(red_pos) if c not in occupied2]
                if len(green_pool) < 4:
                    continue

                for green_quad in combinations(green_pool, 4):
                    occupied3 = occupied2 | set(green_quad)

                    # teal and blue are fully determined
                    board = np.zeros(NUM_CELLS, dtype=np.uint8)
                    board[red_pos]            = COLOR_RED
                    for o in orange_pair:     board[o] = COLOR_ORANGE
                    for y in yellow_triple:   board[y] = COLOR_YELLOW
                    for g in green_quad:      board[g] = COLOR_GREEN

                    for t in teal_cells(red_pos, occupied3):
                        board[t] = COLOR_TEAL
                    # blue cells are already 0 (COLOR_BLUE) by default

                    boards.append(board)

    return np.array(boards, dtype=np.uint8)


def compute_board_weights(boards: np.ndarray) -> np.ndarray:
    """
    Compute sampling weights for each board so that red is uniform
    across all 24 non-center positions (Hypothesis A).

    Since different red positions have different numbers of valid boards,
    naive uniform sampling over boards oversamples inner-ring red positions.

    Fix: weight each board by 1 / (count of boards sharing its red position).
    This makes every red position equally likely in expectation.

    Returns a float64 array of shape (N,) summing to 1.
    """
    red_positions = np.argmax(boards == COLOR_RED, axis=1)
    counts_per_pos = np.bincount(red_positions, minlength=NUM_CELLS).astype(np.float64)

    # weight for board i = 1 / count_of_boards_with_same_red_position
    raw_weights = 1.0 / counts_per_pos[red_positions]

    # normalise so weights sum to 1
    return raw_weights / raw_weights.sum()


def boards_to_tensor(boards: np.ndarray, device='cuda') -> torch.Tensor:
    """Convert boards numpy array to torch tensor on device."""
    return torch.tensor(boards, dtype=torch.uint8, device=device)


if __name__ == '__main__':
    print("Enumerating all valid boards...")
    boards = enumerate_boards()
    print(f"Total valid boards: {len(boards):,}")

    # sanity checks
    assert (boards[:, CENTER] != COLOR_RED).all(), "Red found at center!"
    red_counts   = (boards == COLOR_RED).sum(axis=1)
    orange_counts = (boards == COLOR_ORANGE).sum(axis=1)
    yellow_counts = (boards == COLOR_YELLOW).sum(axis=1)
    green_counts  = (boards == COLOR_GREEN).sum(axis=1)
    assert (red_counts    == 1).all(), "Red count wrong"
    assert (orange_counts == 2).all(), "Orange count wrong"
    assert (yellow_counts == 3).all(), "Yellow count wrong"
    assert (green_counts  == 4).all(), "Green count wrong"
    print("All sanity checks passed.")

    # color distribution at center
    center_colors = boards[:, CENTER]
    print("\nColor distribution at center cell:")
    for ci, name in enumerate(COLOR_NAMES):
        count = (center_colors == ci).sum()
        print(f"  {name:8s}: {count:6,}  ({100*count/len(boards):.1f}%)")

    # teal/blue count stats
    teal_counts = (boards == COLOR_TEAL).sum(axis=1)
    blue_counts = (boards == COLOR_BLUE).sum(axis=1)
    print(f"\nTeal per board: min={teal_counts.min()} max={teal_counts.max()} mean={teal_counts.mean():.2f}")
    print(f"Blue per board: min={blue_counts.min()} max={blue_counts.max()} mean={blue_counts.mean():.2f}")

    # red position frequency — should be uniform across 24 non-center cells
    red_positions = np.argmax(boards == COLOR_RED, axis=1)
    print("\nRed position frequency (should be uniform across 24 non-center cells):")
    counts = np.bincount(red_positions, minlength=NUM_CELLS)
    expected = len(boards) / 24
    print(f"  Expected count per cell: {expected:.1f}")
    print(f"  Min: {counts[counts>0].min()}  Max: {counts.max()}  "
          f"Std: {counts[counts>0].std():.1f}")
    print()

    # print as 5x5 grid
    print("  Red position count grid (. = center, excluded):")
    for r in range(5):
        row_str = "  "
        for c in range(5):
            idx = r * 5 + c
            if idx == CENTER:
                row_str += "   . "
            else:
                row_str += f"{counts[idx]:4d} "
        print(row_str)

    # flag any cell deviating more than 10% from expected
    print()
    deviations = []
    for idx in range(NUM_CELLS):
        if idx == CENTER:
            continue
        dev = abs(counts[idx] - expected) / expected * 100
        if dev > 10:
            r, c = rc(idx)
            deviations.append((idx, r, c, counts[idx], dev))
    if deviations:
        print("  WARNING — cells deviating >10% from expected frequency.")
    else:
        print("  All red positions within 10% of expected — distribution looks uniform.")

    np.save('cache/all_boards.npy', boards)
    print("\nSaved to cache/all_boards.npy")