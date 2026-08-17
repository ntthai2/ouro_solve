"""
board_generator.py

OT (Ourotrace) board generation with constraint-based random placement.
"""

import numpy as np
import random
from typing import List, Tuple, Set, Optional

# Constants
GRID_SIZE = 5
NUM_CELLS = 25

COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW = 0, 1, 2, 3
COLOR_ORANGE, COLOR_WHITE, COLOR_BLACK = 4, 5, 6
COLOR_RED, COLOR_RAINBOW = 7, 8

COLOR_NAMES = ['blue', 'teal', 'green', 'yellow', 'orange', 'white', 'black', 'red', 'rainbow']
COLOR_VALUES = [10, 20, 35, 55, 90, None, None, 150, 500]
MAX_BLUE_CLICKS = 4

RUN_LENGTHS = {
    COLOR_TEAL: 4,
    COLOR_GREEN: 3,
    COLOR_YELLOW: 3,
    COLOR_ORANGE: 2,
    COLOR_WHITE: 2,
    COLOR_BLACK: 2,
    COLOR_RED: 2,
    COLOR_RAINBOW: 2,
}

RARE_COLORS = [COLOR_WHITE, COLOR_BLACK, COLOR_RED, COLOR_RAINBOW]

RARE_COLOR_WEIGHTS = {
    COLOR_WHITE: 0.429,
    COLOR_BLACK: 0.381,
    COLOR_RAINBOW: 0.095,
    COLOR_RED: 0.095,
}

def sample_rare_colors_without_replacement(k: int) -> List[int]:
    """Sample k distinct rare colors using weighted sampling without replacement."""
    pool = list(RARE_COLORS)
    weights = [RARE_COLOR_WEIGHTS[c] for c in pool]
    chosen = []
    for _ in range(k):
        selected = random.choices(pool, weights=weights, k=1)[0]
        idx = pool.index(selected)
        pool.pop(idx)
        weights.pop(idx)
        chosen.append(selected)
    return chosen

def cell(r: int, c: int) -> int:
    return r * GRID_SIZE + c

def enumerate_line_placements(length: int) -> List[Tuple[int, ...]]:
    """Enumerate all valid placements (horizontal and vertical) for a given length."""
    placements = []
    # Horizontal
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - length + 1):
            placements.append(tuple(cell(r, c + i) for i in range(length)))
    # Vertical
    for c in range(GRID_SIZE):
        for r in range(GRID_SIZE - length + 1):
            placements.append(tuple(cell(r + i, c) for i in range(length)))
    return placements

# Cache placements for all possible lengths
PLACEMENTS_CACHE = {
    length: enumerate_line_placements(length) 
    for length in set(RUN_LENGTHS.values())
}

# Empirical prior weights over m_extra in {1, 2} based on observed games.
M_PROBABILITIES = [0.6667, 0.3333]

def generate_random_board(m_extra: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Generate a random board using backtracking with exact prior weights over m_extra in {1, 2}.
    Returns the board as a numpy array, or None if it fails to find a valid placement.
    """
    if m_extra is None:
        m_extra = random.choices([1, 2], weights=M_PROBABILITIES, k=1)[0]
    
    chosen_rare_colors = sample_rare_colors_without_replacement(m_extra)
    colors_to_place = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE] + chosen_rare_colors
    
    board = np.full(NUM_CELLS, COLOR_BLUE, dtype=np.uint8)
    
    def backtrack(color_index: int, used_cells: Set[int]) -> bool:
        if color_index == len(colors_to_place):
            return True
            
        color = colors_to_place[color_index]
        length = RUN_LENGTHS[color]
        possible_placements = PLACEMENTS_CACHE[length]
        
        # Filter valid placements
        valid_placements = [p for p in possible_placements if not used_cells.intersection(p)]
        
        # Shuffle for randomness
        random.shuffle(valid_placements)
        
        for p in valid_placements:
            # Place
            for c in p:
                board[c] = color
                used_cells.add(c)
                
            if backtrack(color_index + 1, used_cells):
                return True
                
            # Undo
            for c in p:
                board[c] = COLOR_BLUE
                used_cells.remove(c)
                
        return False

    success = backtrack(0, set())
    if success:
        return board
    return None

def generate_n_random_boards(n: int, m_extra: Optional[int] = None) -> List[np.ndarray]:
    """Generate a sample of N random boards."""
    boards = []
    for _ in range(n):
        b = generate_random_board(m_extra)
        if b is not None:
            boards.append(b)
    return boards
