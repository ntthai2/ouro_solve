"""
simulation_oq.py

Simulation runner for OQ mode.

Key OQ mechanics:
- 7 paid clicks maximum.
- Purple clicks are free.
- After the 3rd purple click, simulator reveals the 4th purple position to the
  strategy via belief.peek(...), without marking it clicked.
- Clicking that 4th purple yields converted red reward (150) and consumes one
  paid click.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from oq.board_generator import (
    NUM_CELLS,
    COLOR_VALUES,
    COLOR_PURPLE,
    COLOR_RED,
    get_purple_positions,
)
from oq.belief_state import OQFullBeliefState

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

MAX_PAID_CLICKS = 7


def _iter_with_progress(items: Iterable, enabled: bool, desc: str = ""):
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def _choose_fallback_unclicked(clicked_cells: set[int]) -> int | None:
    for c in range(NUM_CELLS):
        if c not in clicked_cells:
            return c
    return None


def run_game_oq(board: np.ndarray, strategy) -> Dict[str, Any]:
    """
    Run one OQ game on a single board.

    Returns:
    {
        'score': score,
        'found_red': found_red,
        'paid_clicks_used': paid_clicks_used,
        'purples_clicked': purples_clicked,
        'click_sequence': str(clicks),
    }
    """
    paid_clicks_used = 0
    purples_clicked = 0
    fourth_purple = None
    belief = OQFullBeliefState()
    score = 0
    found_red = False
    first_click = True
    clicks: List[tuple[int, int, int, bool]] = []

    while paid_clicks_used < MAX_PAID_CLICKS:
        clicks_left = MAX_PAID_CLICKS - paid_clicks_used
        cell = strategy(belief, clicks_left, first_click=first_click)
        first_click = False

        clicked_cells = {cl[0] for cl in clicks}
        if cell in clicked_cells:
            fallback = _choose_fallback_unclicked(clicked_cells)
            if fallback is None:
                break
            cell = fallback

        color = int(board[cell])

        if fourth_purple is not None and cell == fourth_purple:
            # Conversion target click: reward as red, belief observed as purple.
            reward = int(COLOR_VALUES[COLOR_RED])
            found_red = True
            paid_clicks_used += 1
            belief = belief.update(cell, COLOR_PURPLE)
            clicks.append((cell, COLOR_RED, reward, False))
            score += reward
            break

        if color == COLOR_PURPLE:
            reward = int(COLOR_VALUES[COLOR_PURPLE])
            purples_clicked += 1
            belief = belief.update(cell, COLOR_PURPLE)
            clicks.append((cell, COLOR_PURPLE, reward, True))
            score += reward

            # Free click: no paid click increment.
            if purples_clicked == 3:
                all_purples = set(get_purple_positions(board))
                clicked_after_this = {cl[0] for cl in clicks}
                remaining = all_purples - clicked_after_this
                if len(remaining) == 1:
                    fourth_purple = next(iter(remaining))
                    belief = belief.peek(fourth_purple, COLOR_PURPLE)
            continue

        reward = int(COLOR_VALUES[color])
        paid_clicks_used += 1
        belief = belief.update(cell, color)
        clicks.append((cell, color, reward, False))
        score += reward

    # After converted red is found, spend remaining paid clicks on best known cells.
    if found_red:
        while paid_clicks_used < MAX_PAID_CLICKS:
            clicked_cells = {cl[0] for cl in clicks}
            unclicked = [c for c in range(NUM_CELLS) if c not in clicked_cells]
            if not unclicked:
                break

            best = max(
                unclicked,
                key=lambda c: (COLOR_VALUES[int(board[c])] if int(board[c]) != COLOR_PURPLE else 0),
            )
            reward = int(COLOR_VALUES[int(board[best])])
            score += reward
            clicks.append((best, int(board[best]), reward, False))
            paid_clicks_used += 1

    return {
        'score': score,
        'found_red': found_red,
        'paid_clicks_used': paid_clicks_used,
        'purples_clicked': purples_clicked,
        'click_sequence': str(clicks),
    }


def run_simulation_oq(all_boards: np.ndarray,
                      strategies: list,
                      verbose: bool = True) -> pd.DataFrame:
    """Run every strategy on every board and return a result DataFrame."""
    OQFullBeliefState.load_boards(all_boards)

    results: List[Dict[str, Any]] = []
    n_boards = len(all_boards)

    for strategy in strategies:
        strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
        if verbose:
            print(f"\nRunning strategy: {strategy_name}")
        t0 = time.time()

        board_iter = _iter_with_progress(
            enumerate(all_boards),
            enabled=verbose,
            desc=f"{strategy_name}",
        )

        for board_idx, board in board_iter:
            game_result = run_game_oq(board, strategy)
            results.append({
                'strategy': strategy_name,
                'board_idx': board_idx,
                'score': game_result['score'],
                'found_red': game_result['found_red'],
                'paid_clicks_used': game_result['paid_clicks_used'],
                'purples_clicked': game_result['purples_clicked'],
            })

        if verbose:
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s over {n_boards:,} boards")

    return pd.DataFrame(results)
