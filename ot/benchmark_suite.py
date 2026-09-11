"""
benchmark_suite.py

Deterministic, 100% reproducible benchmark runner for Ourotrace ($ot).
Evaluates strategies on the standardized 200-board Stratified Representative Suite
(cache/ot_representative_suite_200.pkl) under the Board-Seeded Cascade Protocol.

Protocol Specification:
- Suite: cache/ot_representative_suite_200.pkl (N=200 boards, 150 6-color, 50 7-color)
- Board-Seeded Cascade: random.seed(20260911 + board_idx) before each game.
  This ensures 100% identical cascade sequences across all strategies for every board.
- Output: Exact EV, Std, 95% CI, Min/Max Range, Win Rate, and Average Latency.
"""

import os
import sys
import time
import pickle
import random
from typing import List, Tuple, Any
import numpy as np

# Ensure root in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ot.board_generator import COLOR_BLUE
from ot.simulation import run_game_ot, sample_value
from ot.strategies import OTInfoGainStrategy, OTHybridStrategy

SUITE_PATH = os.path.join(ROOT, "cache", "ot_representative_suite_200.pkl")
SEED_BASE = 20260911


def run_benchmark():
    if not os.path.exists(SUITE_PATH):
        raise FileNotFoundError(f"Suite not found at {SUITE_PATH}")

    with open(SUITE_PATH, "rb") as f:
        boards = pickle.load(f)

    print(f"Loaded {len(boards)} boards from {SUITE_PATH}")
    print(f"Protocol: Board-Seeded Cascade (seed = {SEED_BASE} + board_idx)")
    print("=" * 135)

    # Strategy configurations: (Display Name, Strategy Object, is_blind)
    strategies: List[Tuple[str, Any, Any]] = [
        ("Oracle Bound", None, "oracle"),
        ("Color-Conditioned VOI (lam=0.88, DP<=18) [Production]",
         OTInfoGainStrategy(lam=0.88, use_exact_endgame=True, n_samples=3500), False),
        ("Blind Prior VOI (lam=0.88, DP<=16) [Previous]",
         OTInfoGainStrategy(lam=0.88, use_exact_endgame=True, n_samples=3500), True),
        ("Blind Prior VOI (lam=0.95, DP<=16)",
         OTInfoGainStrategy(lam=0.95, use_exact_endgame=True, n_samples=3500), True),
        ("No Lookahead VOI (lam=0.88, K=1, Blind)",
         OTInfoGainStrategy(lam=0.88, use_exact_endgame=True, n_samples=3500, k_prune=1), True),
        ("Hybrid Greedy (lam=1.00, Pure Survival) [Baseline]",
         OTHybridStrategy(use_exact_endgame=True, n_samples=3500), True),
    ]

    raw_results = []
    baseline_ev = 0.0

    for name, strat, is_blind in strategies:
        scores = []
        wins = []
        cleared_pcts = []
        t0 = time.time()

        if is_blind == "oracle":
            for i, b in enumerate(boards):
                random.seed(SEED_BASE + i)
                score = sum(sample_value(int(b[c])) for c in range(25) if b[c] != COLOR_BLUE) + 40
                scores.append(score)
                wins.append(1)
                cleared_pcts.append(100.0)
        else:
            for i, b in enumerate(boards):
                random.seed(SEED_BASE + i)
                res = run_game_ot(b, strat, blind_prior=is_blind)
                scores.append(res["score"])
                wins.append(1 if res["win"] else 0)
                tot_non_blue = res.get("total_non_blue", 14)
                clr_non_blue = res.get("cleared_non_blue", 0)
                cleared_pcts.append((clr_non_blue / max(1, tot_non_blue)) * 100.0)

        elapsed = time.time() - t0
        scores_arr = np.array(scores)
        ev = float(np.mean(scores_arr))
        std = float(np.std(scores_arr, ddof=1))
        ci = 1.96 * std / np.sqrt(len(scores_arr))
        wr = float(np.mean(wins)) * 100.0
        clr = float(np.mean(cleared_pcts))
        lat = (elapsed / len(boards)) * 1000.0
        s_min = int(np.min(scores_arr))
        s_max = int(np.max(scores_arr))

        if "Baseline" in name:
            baseline_ev = ev

        raw_results.append({
            "name": name,
            "ev": ev,
            "std": std,
            "ci_l": ev - ci,
            "ci_u": ev + ci,
            "min": s_min,
            "max": s_max,
            "wr": wr,
            "clr": clr,
            "lat": lat
        })

    print(f"{'Strategy':<55} | {'EV':<7} | {'Std':<6} | {'95% CI':<17} | {'Range':<11} | {'Delta':<7} | {'WR':<5} | {'% Clr':<6} | {'Latency':<7}")
    print("-" * 135)

    for r in raw_results:
        delta = r["ev"] - baseline_ev
        delta_str = f"{delta:+6.2f}" if r["name"] != "Hybrid Greedy (lam=1.00, Pure Survival) [Baseline]" else "Baseline"
        print(f"{r['name']:<55} | {r['ev']:7.2f} | {r['std']:6.2f} | [{r['ci_l']:6.2f}, {r['ci_u']:6.2f}] | [{r['min']:4d}, {r['max']:4d}] | {delta_str:<7} | {r['wr']:4.1f}% | {r['clr']:5.1f}% | {r['lat']:5.1f}ms")

    print("=" * 135)

    # 6-color vs 7-color subspace breakdown for Production
    prod_strat = strategies[1][1]
    scores_6 = []
    scores_7 = []
    for i, b in enumerate(boards):
        random.seed(SEED_BASE + i)
        res = run_game_ot(b, prod_strat, blind_prior=False)
        n_unique = len(np.unique(b))
        if n_unique <= 6:
            scores_6.append(res["score"])
        else:
            scores_7.append(res["score"])

    ev_6 = np.mean(scores_6)
    ev_7 = np.mean(scores_7)
    print(f"\nSubspace Breakdown for Production (Color-Conditioned):")
    print(f"  6-Color Boards (N={len(scores_6)}): EV = {ev_6:.2f}")
    print(f"  7-Color Boards (N={len(scores_7)}): EV = {ev_7:.2f}")
    print(f"  Weighted (75/25): {0.75*ev_6 + 0.25*ev_7:.2f} (Suite average: {np.mean(scores_6 + scores_7):.2f})")


if __name__ == "__main__":
    run_benchmark()
