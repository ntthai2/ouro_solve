"""
simulation.py

Exact evaluation of all strategies across every valid board.
One row per (strategy, board) in the output DataFrame.

For the baseline (random), runs multiple trials per board and averages,
since it has stochasticity. All other strategies are deterministic.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from oc.board_generator import (
    NUM_CELLS, CENTER, COLOR_NAMES, COLOR_VALUES,
    COLOR_RED, compute_board_weights,
)
from oc.belief_state import LightBeliefState, FullBeliefState
from oc.strategies import (
    BaselineCenterRandom, ExactPOMDP, VOIGreedy,
    EntropyMinimization, CandidateHalving,
)

MAX_CLICKS = 5
BASELINE_TRIALS = 200  # random trials per board for baseline strategy


# ── single game runner ────────────────────────────────────────────────────────

def run_game_full(strategy, board: np.ndarray, all_boards: np.ndarray,
                  board_index: int, board_weight: float = 1.0) -> Dict[str, Any]:
    """
    Run one game of a deterministic strategy on a given board.
    Uses FullBeliefState (required by POMDP, VOI, entropy, halving).
    All 5 clicks are always used — finding red does not end the game.
    """
    belief      = FullBeliefState(revealed=frozenset())
    score       = 0
    clicks      = []
    found_red   = False
    first_click = True

    for click_num in range(MAX_CLICKS):
        cell = strategy(belief, MAX_CLICKS - click_num, first_click=first_click)
        first_click = False

        color  = int(board[cell])
        reward = COLOR_VALUES[color]
        score += reward
        clicks.append((cell, color, reward))

        if color == COLOR_RED:
            found_red = True

        belief = belief.update(cell, color)

    return {
        'board_index':    board_index,
        'board_weight':   board_weight,
        'strategy':       strategy.name,
        'score':          score,
        'found_red':      found_red,
        'num_clicks':     len(clicks),
        'click_sequence': str([(c, COLOR_NAMES[col]) for c, col, _ in clicks]),
        'center_color':   COLOR_NAMES[int(board[CENTER])],
        'red_position':   int(np.argmax(board == COLOR_RED)),
    }


def run_game_baseline(strategy: BaselineCenterRandom, board: np.ndarray,
                      board_index: int, board_weight: float = 1.0,
                      n_trials: int = BASELINE_TRIALS) -> Dict[str, Any]:
    """
    Run multiple random trials for the baseline strategy, return averaged results.
    """
    scores     = []
    found_reds = []
    click_counts = []

    for _ in range(n_trials):
        belief     = LightBeliefState()
        score      = 0
        found_red  = False
        first_click = True
        n_clicks   = 0

        for click_num in range(MAX_CLICKS):
            cell = strategy(belief, MAX_CLICKS - click_num, first_click=first_click)
            first_click = False

            color  = int(board[cell])
            reward = COLOR_VALUES[color]
            score += reward
            n_clicks += 1

            if color == COLOR_RED:
                found_red = True

            belief = belief.update(cell, color)

        scores.append(score)
        found_reds.append(found_red)
        click_counts.append(n_clicks)

    return {
        'board_index':    board_index,
        'board_weight':   board_weight,
        'strategy':       strategy.name,
        'score':          np.mean(scores),
        'score_std':      np.std(scores),
        'found_red':      np.mean(found_reds),
        'num_clicks':     np.mean(click_counts),
        'click_sequence': '',
        'center_color':   COLOR_NAMES[int(board[CENTER])],
        'red_position':   int(np.argmax(board == COLOR_RED)),
    }


# ── full simulation ───────────────────────────────────────────────────────────

def run_simulation(all_boards: np.ndarray,
                   pomdp_strategy: ExactPOMDP = None,
                   voi_strategies: list = None,
                   entropy_strategy: EntropyMinimization = None,
                   halving_strategy: CandidateHalving = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate all strategies on all boards.
    voi_strategies: list of VOIGreedy instances at different depths.
    """
    FullBeliefState.load_boards(all_boards)
    n_boards = len(all_boards)
    weights  = compute_board_weights(all_boards)

    det_strategies = []
    if pomdp_strategy is not None:
        det_strategies.append(pomdp_strategy)
    for voi in (voi_strategies or []):
        det_strategies.append(voi)
    det_strategies.append(
        entropy_strategy if entropy_strategy is not None else EntropyMinimization())
    det_strategies.append(
        halving_strategy if halving_strategy is not None else CandidateHalving())

    baseline = BaselineCenterRandom()
    results  = []

    # ── deterministic strategies ──────────────────────────────────────────────
    for strategy in det_strategies:
        if verbose:
            print(f"\nRunning strategy: {strategy.name}")
        t0 = time.time()

        for i, board in enumerate(all_boards):
            if verbose and i % 5000 == 0:
                print(f"  Board {i:,} / {n_boards:,}", end='\r')
            result = run_game_full(strategy, board, all_boards, i,
                                   board_weight=float(weights[i]))
            results.append(result)

        if verbose:
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

    # ── baseline (stochastic) ─────────────────────────────────────────────────
    if verbose:
        print(f"\nRunning strategy: {baseline.name} ({BASELINE_TRIALS} trials/board)")
    t0 = time.time()

    for i, board in enumerate(all_boards):
        if verbose and i % 5000 == 0:
            print(f"  Board {i:,} / {n_boards:,}", end='\r')
        result = run_game_baseline(baseline, board, i,
                                   board_weight=float(weights[i]))
        results.append(result)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    df = pd.DataFrame(results)
    return df


# ── summary stats ─────────────────────────────────────────────────────────────

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean — weights need not sum to 1 within the group."""
    w = weights / weights.sum()
    return float((values * w).sum())


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-strategy summary statistics using board_weight for correct
    expected values under Hypothesis A (uniform red position).
    """
    rows = []
    for strategy_name, group in df.groupby('strategy'):
        w = group['board_weight'].values
        s = group['score'].values
        f = group['found_red'].values
        c = group['num_clicks'].values
        rows.append({
            'strategy':          strategy_name,
            'expected_score':    _weighted_mean(s, w),
            'score_std':         float(np.sqrt(_weighted_mean((s - _weighted_mean(s, w))**2, w))),
            'p_find_red':        _weighted_mean(f, w),
            'avg_clicks':        _weighted_mean(c, w),
            'score_min':         float(s.min()),
            'score_max':         float(s.max()),
            'n_boards':          len(group),
        })

    summary = pd.DataFrame(rows).sort_values('expected_score', ascending=False)
    return summary


def compute_by_center_color(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break down expected score and P(find red) by center cell color,
    using board_weight for correct expected values.
    """
    rows = []
    for (strategy_name, center_color), group in df.groupby(['strategy', 'center_color']):
        w = group['board_weight'].values
        rows.append({
            'strategy':       strategy_name,
            'center_color':   center_color,
            'expected_score': _weighted_mean(group['score'].values, w),
            'p_find_red':     _weighted_mean(group['found_red'].values, w),
            'count':          len(group),
        })

    return (pd.DataFrame(rows)
              .sort_values(['strategy', 'expected_score'], ascending=[True, False]))