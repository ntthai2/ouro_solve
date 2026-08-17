"""
simulation.py

Simulation runner for OT mode.
"""

from typing import Any, Dict, List
import time
import random
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ot.belief_state import OTBeliefState
from ot.board_generator import NUM_CELLS, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK, COLOR_VALUES, COLOR_NAMES

MAX_BLUE_CLICKS = 4

# Base values
RED_BASE_VALUE = 150
RAINBOW_BASE_VALUE = 500
PURPLE_BASE_VALUE = 5

SPAWN_POOL_9 = [
    "blue", "teal", "green", "yellow", "orange",
    "purple", "red", "rainbow", "white"
]

BASE_VALUES = {
    "blue": 10,
    "teal": 20,
    "green": 35,
    "yellow": 55,
    "orange": 90,
    "purple": PURPLE_BASE_VALUE,
    "red": RED_BASE_VALUE,
    "rainbow": RAINBOW_BASE_VALUE,
    "white": 100,  # Fallback for recursion guard
    "black": 120,  # Fallback for recursion guard
}

WHITE_SPAWN_WEIGHTS = {
    "blue": 0.516,
    "teal": 0.129,
    "green": 0.065,
    "orange": 0.065,
    "purple": 0.065,
    "yellow": 0.032,
    "red": 0.032,
    "rainbow": 0.032,
}

WHITE_SPAWN_COLORS = list(WHITE_SPAWN_WEIGHTS.keys())
WHITE_SPAWN_PROBS = [WHITE_SPAWN_WEIGHTS[c] for c in WHITE_SPAWN_COLORS]

def sample_white_cascade() -> int:
    """
    White click: spawns 4-5 spheres. Total value = sum of base values + 16 (once).
    White and Black are excluded from its spawn pool, so no further cascade occurs.
    """
    count = random.choice([4, 5])
    spawned = random.choices(WHITE_SPAWN_COLORS, weights=WHITE_SPAWN_PROBS, k=count)
    total_base = sum(BASE_VALUES[color_name] for color_name in spawned)
    return total_base + 16

def sample_black_cascade() -> int:
    """
    Black click: spawns 1 sphere uniformly from 9 colors (excluding black).
    Value = base of spawned sphere + 16.
    If spawned sphere is White, its cascade is resolved.
    """
    spawned = random.choice(SPAWN_POOL_9)
    if spawned == "white":
        sphere_val = sample_white_cascade()
    else:
        sphere_val = BASE_VALUES[spawned]
    return sphere_val + 16

def sample_value(color: int) -> int:
    if color == COLOR_WHITE:
        return sample_white_cascade()
    elif color == COLOR_BLACK:
        return sample_black_cascade()
    else:
        return COLOR_VALUES[color]

def run_game_ot(board: np.ndarray, strategy) -> Dict[str, Any]:
    belief = OTBeliefState()
    score = 0
    blue_clicks = 0
    clicked = set()
    clicks = []
    unrevealed_when_lost = -1
    
    while blue_clicks < MAX_BLUE_CLICKS:
        remaining = [c for c in range(NUM_CELLS) if c not in clicked]
        if not remaining:
            break
            
        cell = strategy(belief, remaining)
        color = int(board[cell])
        clicked.add(cell)
        
        if color == COLOR_BLUE:
            blue_clicks += 1
            reward = COLOR_VALUES[COLOR_BLUE]
            score += reward
            belief = belief.update(cell, COLOR_BLUE)
            clicks.append((cell, color, reward, False))
            
            if blue_clicks == MAX_BLUE_CLICKS:
                unrevealed_when_lost = len([c for c in range(NUM_CELLS) if c not in clicked])
            continue
            
        reward = sample_value(color)
        score += reward
        belief = belief.update(cell, color)
        clicks.append((cell, color, reward, True))
        
    total_non_blue = 25 - np.sum(board == COLOR_BLUE)
    cleared_non_blue = len(clicked) - blue_clicks
    win = (cleared_non_blue == total_non_blue)
    
    return {
        'score': score,
        'lost': blue_clicks >= MAX_BLUE_CLICKS,
        'win': win,
        'cells_cleared': len(clicked),
        'blue_clicks': blue_clicks,
        'total_non_blue': total_non_blue,
        'cleared_non_blue': cleared_non_blue,
        'unrevealed_when_lost': unrevealed_when_lost,
        'click_sequence': str(clicks)
    }

def run_simulation_ot(boards: List[np.ndarray], strategies: list, verbose: bool = True) -> pd.DataFrame:
    results = []
    
    for strategy in strategies:
        strat_name = getattr(strategy, 'name', strategy.__class__.__name__)
        if verbose:
            print(f"\nRunning strategy: {strat_name}")
            
        t0 = time.time()
        
        board_iter = enumerate(boards)
        if verbose and tqdm is not None:
            board_iter = tqdm(board_iter, total=len(boards), desc=strat_name)
            
        for idx, board in board_iter:
            res = run_game_ot(board, strategy)
            results.append({
                'strategy': strat_name,
                'board_idx': idx,
                'score': res['score'],
                'lost': res['lost'],
                'win': res['win'],
                'cells_cleared': res['cells_cleared'],
                'blue_clicks': res['blue_clicks'],
                'total_non_blue': res['total_non_blue'],
                'cleared_non_blue': res['cleared_non_blue'],
                'unrevealed_when_lost': res['unrevealed_when_lost']
            })
            
        if verbose:
            print(f"  Done in {time.time() - t0:.2f}s")
            
    return pd.DataFrame(results)
