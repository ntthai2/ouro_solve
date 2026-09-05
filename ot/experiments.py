"""
experiments.py

Unified ablation test harness for Ourotrace ($ot) on the standardized
200-board Stratified Representative Suite (cache/ot_representative_suite_200.pkl).

Evaluates all 8 hypotheses documented in README (Table T4):
  - H1: Extended Exact DP Threshold (<=14 vs <=16 vs <=17 vs <=18)
  - H2: Dynamic Endgame Lookahead (U <= 10, K=3)
  - H3: Negative Blue Information Gain (beta in [0.2, 0.6])
  - H4: Adaptive Lambda by Remaining Lives (lambda_early -> lambda_critical)
  - H5: Continuous Placement Elimination Gain (alpha in [0.10, 0.25])
  - H6: Orientation-Locking Probes (gamma in [0.15, 0.50])
  - H7: Nonlinear Survival-Aware Hazard Penalty (convex multiplier in [1.5, 2.0])
  - H8: Calibrated Midgame Lookahead (U in [11, 13])
"""

import os
import sys
import time
import pickle
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import scipy.stats as stats

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ot.board_generator import (
    NUM_CELLS, COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW,
    COLOR_ORANGE, COLOR_VALUES, RUN_LENGTHS, RARE_COLORS
)
from ot.belief_state import OTBeliefState
from ot.strategies import (
    OTInfoGainStrategy, MOVE2_OPENING_BOOK, _score_safe_cell
)
from ot.simulation import run_game_ot

SUITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "ot_representative_suite_200.pkl")


# ── Helpers for Hypothesis Testing ───────────────────────────────────────────

def _has_unlocked_orientation(belief: OTBeliefState, color: int) -> bool:
    """Check if color placements still span both horizontal and vertical axes."""
    placements = belief.candidate_placements.get(color, [])
    if len(placements) <= 1:
        return False
    is_h = False
    is_v = False
    for p in placements:
        if len(p) >= 2:
            delta = p[1] - p[0]
            if delta == 1:
                is_h = True
            elif delta == 5:
                is_v = True
            if is_h and is_v:
                return True
    return is_h and is_v


class ConfigurableOTStrategy:
    """
    Parametric strategy wrapper supporting any combination of the 8 hypotheses.
    """
    def __init__(
        self,
        name: str,
        lam: float = 0.88,
        n_samples: int = 3500,
        exact_threshold: int = 16,
        # H2: Dynamic Endgame Lookahead
        enable_endgame_lookahead: bool = True,
        endgame_u_threshold: int = 10,
        endgame_k: int = 3,
        # H3: Negative Blue Info Gain
        beta_blue_info: float = 0.0,
        # H4: Adaptive Lambda by Lives Left
        adaptive_lambda: bool = False,
        lam_early: float = 0.88,
        lam_critical: float = 0.98,
        # H5: Continuous Placement Elimination Gain
        alpha_placement_gain: float = 0.0,
        # H6: Orientation-Locking Probes
        gamma_orientation_probe: float = 0.0,
        # H7: Nonlinear Survival Hazard Penalty
        enable_survival_penalty: bool = False,
        survival_exp: float = 1.5,
        # H8: Midgame Lookahead
        enable_midgame_lookahead: bool = False,
        midgame_u_max: int = 13,
        midgame_k: int = 2,
    ):
        self.name = name
        self.lam = lam
        self.n_samples = n_samples
        self.exact_threshold = exact_threshold
        self.enable_endgame_lookahead = enable_endgame_lookahead
        self.endgame_u_threshold = endgame_u_threshold
        self.endgame_k = endgame_k
        self.beta_blue_info = beta_blue_info
        self.adaptive_lambda = adaptive_lambda
        self.lam_early = lam_early
        self.lam_critical = lam_critical
        self.alpha_placement_gain = alpha_placement_gain
        self.gamma_orientation_probe = gamma_orientation_probe
        self.enable_survival_penalty = enable_survival_penalty
        self.survival_exp = survival_exp
        self.enable_midgame_lookahead = enable_midgame_lookahead
        self.midgame_u_max = midgame_u_max
        self.midgame_k = midgame_k

    def _get_active_lambda(self, lives_left: int) -> float:
        if not self.adaptive_lambda:
            return self.lam
        if lives_left >= 3:
            return self.lam_early
        elif lives_left == 2:
            return self.lam
        else:
            return self.lam_critical

    def __call__(self, belief: OTBeliefState, remaining: List[int]) -> int:
        # Move 1: Pinned C3
        if len(belief.revealed) == 0:
            return 12

        # Move 2: Opening Book for C3
        if len(belief.revealed) == 1 and 12 in belief.revealed:
            col_at_12 = belief.revealed[12]
            if col_at_12 in MOVE2_OPENING_BOOK:
                rec_m2 = MOVE2_OPENING_BOOK[col_at_12]
                if rec_m2 in remaining:
                    return rec_m2

        # Phase 1: Certain safe cells (deterministic)
        safe_cells = belief.certain_safe_cells()
        valid_safe = [c for c in safe_cells if c in remaining]
        if valid_safe:
            return max(valid_safe, key=lambda c: _score_safe_cell(belief, c))

        # Determine effective lambda
        blue_hits = sum(1 for col in belief.revealed.values() if col == COLOR_BLUE)
        lives_left = max(1, 4 - blue_hits)
        active_lam = self._get_active_lambda(lives_left)

        # Phase 2: Posterior probabilities
        use_dp = (len(remaining) <= self.exact_threshold)
        probs = belief.p_color_all(use_exact_endgame=use_dp, n_samples=self.n_samples)

        # Zero-Hazard Cells
        zero_risk_cells = [c for c in remaining if probs[COLOR_BLUE][c] == 0.0]
        if zero_risk_cells:
            best_c, best_s = -1, -float('inf')
            for c in zero_risk_cells:
                e_info = 0.0
                for col_id, p_list in probs.items():
                    if col_id != COLOR_BLUE and p_list[c] > 0:
                        nb = belief.update(c, col_id)
                        v_ns = [s for s in nb.certain_safe_cells() if s in remaining and s != c]
                        e_info += p_list[c] * len(v_ns)
                tie = _score_safe_cell(belief, c)
                total_s = (e_info, tie[0], tie[1], tie[2])
                if total_s > (best_s, -1, -1, -100):
                    best_s = e_info
                    best_c = c
            return best_c

        # Phase 3: 1-Step Screening
        unrev = len(remaining)
        total_curr_placements = sum(len(p) for p in belief.candidate_placements.values())

        step1_scores = []
        for c in remaining:
            p_blue = probs[COLOR_BLUE][c]
            e_safe = 0.0
            e_placements_elim = 0.0
            orientation_gain = 0.0

            for col_id, p_list in probs.items():
                p_c = p_list[c]
                if p_c <= 0:
                    continue
                if col_id != COLOR_BLUE:
                    nb = belief.update(c, col_id)
                    ns = nb.certain_safe_cells()
                    v_ns = [s for s in ns if s in remaining and s != c]
                    e_safe += p_c * len(v_ns)

                    if self.alpha_placement_gain > 0:
                        nb_placements = sum(len(p) for p in nb.candidate_placements.values())
                        e_placements_elim += p_c * (total_curr_placements - nb_placements)

                    if self.gamma_orientation_probe > 0:
                        if _has_unlocked_orientation(belief, col_id) and not _has_unlocked_orientation(nb, col_id):
                            orientation_gain += p_c * 1.0
                else:
                    if self.beta_blue_info > 0:
                        nb = belief.update(c, COLOR_BLUE)
                        ns = nb.certain_safe_cells()
                        v_ns = [s for s in ns if s in remaining and s != c]
                        e_safe += self.beta_blue_info * p_c * len(v_ns)

            # Hazard penalty calculation
            if self.enable_survival_penalty:
                hazard_weight = active_lam * ((5 - lives_left) ** self.survival_exp) / 2.0
            else:
                hazard_weight = active_lam

            score = -hazard_weight * p_blue + (1.0 - active_lam) * e_safe
            if self.alpha_placement_gain > 0 and total_curr_placements > 0:
                score += (1.0 - active_lam) * self.alpha_placement_gain * (e_placements_elim / max(1, total_curr_placements))
            if self.gamma_orientation_probe > 0:
                score += (1.0 - active_lam) * self.gamma_orientation_probe * orientation_gain

            step1_scores.append((score, c))

        step1_scores.sort(key=lambda x: x[0], reverse=True)

        # Lookahead Check
        active_k = 1
        if self.enable_endgame_lookahead and unrev <= self.endgame_u_threshold:
            active_k = self.endgame_k
        elif self.enable_midgame_lookahead and (self.endgame_u_threshold < unrev <= self.midgame_u_max):
            active_k = self.midgame_k

        if active_k <= 1 or len(step1_scores) <= 1:
            return step1_scores[0][1]

        # Multi-Step Lookahead across top-k candidates
        best_c = step1_scores[0][1]
        best_ev = -float('inf')

        for _, c1 in step1_scores[:active_k]:
            p1_blue = probs[COLOR_BLUE][c1]
            e_ev2 = 0.0

            for col_id, p_list in probs.items():
                p_c1 = p_list[c1]
                if p_c1 <= 0:
                    continue

                new_belief = belief.update(c1, col_id)
                rem_next = [x for x in remaining if x != c1]

                if col_id == COLOR_BLUE:
                    step1_loss = -active_lam
                    e_ev2 += p_c1 * step1_loss
                    continue

                safe2 = [s for s in new_belief.certain_safe_cells() if s in rem_next]
                if safe2:
                    step2_val = len(safe2)
                else:
                    if rem_next and len(rem_next) <= self.exact_threshold:
                        p2 = new_belief.p_color_all(use_exact_endgame=True)
                        min_p2_blue = min(p2[COLOR_BLUE][x] for x in rem_next)
                        step2_val = -min_p2_blue
                    else:
                        step2_val = 0.0

                e_ev2 += p_c1 * (1.0 + step2_val)

            c1_ev = -active_lam * p1_blue + (1.0 - active_lam) * e_ev2
            if c1_ev > best_ev:
                best_ev = c1_ev
                best_c = c1

        return best_c


# ── Benchmark Evaluation Runner ──────────────────────────────────────────────

def evaluate_paired(boards: List[np.ndarray], baseline: ConfigurableOTStrategy, challengers: List[ConfigurableOTStrategy]):
    n_boards = len(boards)
    print(f"\nEvaluating {len(challengers)} challenger(s) against Baseline on {n_boards} boards...")
    print("=" * 105)

    # 1. Run Baseline
    t0 = time.time()
    base_scores = []
    base_wins = []
    for b in boards:
        res = run_game_ot(b, baseline)
        base_scores.append(res['score'])
        base_wins.append(1 if res['win'] else 0)
    base_time = time.time() - t0

    base_ev = np.mean(base_scores)
    base_wr = np.mean(base_wins) * 100
    base_latency = (base_time / n_boards) * 1000

    print(f"{baseline.name:<45} | EV: {base_ev:6.2f} | WR: {base_wr:4.1f}% | Lat: {base_latency:5.1f}ms (REF)")
    print("-" * 105)

    # 2. Run Challengers
    for ch in challengers:
        t0 = time.time()
        ch_scores = []
        ch_wins = []
        for b in boards:
            res = run_game_ot(b, ch)
            ch_scores.append(res['score'])
            ch_wins.append(1 if res['win'] else 0)
        ch_time = time.time() - t0

        ch_ev = np.mean(ch_scores)
        ch_wr = np.mean(ch_wins) * 100
        ch_latency = (ch_time / n_boards) * 1000
        delta_ev = ch_ev - base_ev
        delta_wr = ch_wr - base_wr

        diffs = np.array(ch_scores) - np.array(base_scores)
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            _, p_val = stats.ttest_1samp(diffs, 0.0)

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"{ch.name:<45} | EV: {ch_ev:6.2f} ({delta_ev:+6.2f}) | WR: {ch_wr:4.1f}% ({delta_wr:+4.1f}%) | Lat: {ch_latency:5.1f}ms | p={p_val:.4f} ({sig})")

    print("=" * 105)


def make_production_baseline() -> ConfigurableOTStrategy:
    return ConfigurableOTStrategy(
        name="Production Baseline (lam=0.88, DP<=16, Lookahead U<=10)",
        lam=0.88,
        n_samples=3500,
        exact_threshold=16,
        enable_endgame_lookahead=True,
        endgame_u_threshold=10,
        endgame_k=3
    )


def main():
    parser = argparse.ArgumentParser(description="Unified Ablation Test Harness for $ot")
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"],
                        help="Hypothesis ID to run (1-8 or all)")
    parser.add_argument("--suite-size", type=int, default=50,
                        help="Number of suite boards to evaluate on (e.g. 20, 50, 100, 200)")
    args = parser.parse_args()

    if not os.path.exists(SUITE_PATH):
        print(f"Suite file not found at {SUITE_PATH}. Generating stratified suite...")
        from ot.stratified_suite import generate_stratified_suite, save_suite
        suite = generate_stratified_suite(200)
        save_suite(suite, SUITE_PATH)

    with open(SUITE_PATH, "rb") as f:
        all_suite = pickle.load(f)

    n = min(args.suite_size, len(all_suite))
    boards = all_suite[:n]
    print(f"Loaded {n}/{len(all_suite)} boards from standardized suite.")

    baseline = make_production_baseline()

    # ── H1: Extended Exact DP Threshold ──────────────────────────────────────
    if args.exp in ("1", "all"):
        print("\n>>> HYPOTHESIS 1: Extended Exact DP Threshold (<=14, <=16, <=17)")
        challengers = [
            ConfigurableOTStrategy("DP<=14 (Old Baseline)", lam=0.88, exact_threshold=14),
            ConfigurableOTStrategy("DP<=17 (Deeper Exact)", lam=0.88, exact_threshold=17),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H2: Dynamic Endgame Lookahead ────────────────────────────────────────
    if args.exp in ("2", "all"):
        print("\n>>> HYPOTHESIS 2: Dynamic Endgame Lookahead (Lookahead disabled vs U<=10 K=3)")
        challengers = [
            ConfigurableOTStrategy("No Endgame Lookahead (K=1)", lam=0.88, exact_threshold=16, enable_endgame_lookahead=False),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H3: Negative Blue Information Gain ───────────────────────────────────
    if args.exp in ("3", "all"):
        print("\n>>> HYPOTHESIS 3: Negative Blue Information Gain (beta in [0.2, 0.6])")
        challengers = [
            ConfigurableOTStrategy("Blue Info (beta=0.20)", lam=0.88, exact_threshold=16, beta_blue_info=0.20),
            ConfigurableOTStrategy("Blue Info (beta=0.40)", lam=0.88, exact_threshold=16, beta_blue_info=0.40),
            ConfigurableOTStrategy("Blue Info (beta=0.60)", lam=0.88, exact_threshold=16, beta_blue_info=0.60),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H4: Adaptive Lambda by Remaining Lives ───────────────────────────────
    if args.exp in ("4", "all"):
        print("\n>>> HYPOTHESIS 4: Adaptive Lambda by Remaining Lives")
        challengers = [
            ConfigurableOTStrategy("Adaptive Lam (0.88 -> 0.96)", lam=0.88, exact_threshold=16, adaptive_lambda=True, lam_early=0.88, lam_critical=0.96),
            ConfigurableOTStrategy("Adaptive Lam (0.88 -> 0.98)", lam=0.88, exact_threshold=16, adaptive_lambda=True, lam_early=0.88, lam_critical=0.98),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H5: Continuous Placement Elimination Gain ────────────────────────────
    if args.exp in ("5", "all"):
        print("\n>>> HYPOTHESIS 5: Continuous Placement Elimination Gain")
        challengers = [
            ConfigurableOTStrategy("Placement Gain (alpha=0.10)", lam=0.88, exact_threshold=16, alpha_placement_gain=0.10),
            ConfigurableOTStrategy("Placement Gain (alpha=0.25)", lam=0.88, exact_threshold=16, alpha_placement_gain=0.25),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H6: Orientation-Locking Probes ───────────────────────────────────────
    if args.exp in ("6", "all"):
        print("\n>>> HYPOTHESIS 6: Orientation-Locking Probes")
        challengers = [
            ConfigurableOTStrategy("Orientation Probe (gamma=0.20)", lam=0.88, exact_threshold=16, gamma_orientation_probe=0.20),
            ConfigurableOTStrategy("Orientation Probe (gamma=0.50)", lam=0.88, exact_threshold=16, gamma_orientation_probe=0.50),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H7: Nonlinear Survival Hazard Penalty ────────────────────────────────
    if args.exp in ("7", "all"):
        print("\n>>> HYPOTHESIS 7: Nonlinear Survival Hazard Penalty")
        challengers = [
            ConfigurableOTStrategy("Survival Penalty (exp=1.5)", lam=0.88, exact_threshold=16, enable_survival_penalty=True, survival_exp=1.5),
            ConfigurableOTStrategy("Survival Penalty (exp=2.0)", lam=0.88, exact_threshold=16, enable_survival_penalty=True, survival_exp=2.0),
        ]
        evaluate_paired(boards, baseline, challengers)

    # ── H8: Calibrated Midgame Lookahead (U in [11, 13]) ─────────────────────
    if args.exp in ("8", "all"):
        print("\n>>> HYPOTHESIS 8: Calibrated Midgame Lookahead (U in [11, 13])")
        challengers = [
            ConfigurableOTStrategy("Midgame Lookahead (U<=12, K=2)", lam=0.88, exact_threshold=16, enable_midgame_lookahead=True, midgame_u_max=12, midgame_k=2),
            ConfigurableOTStrategy("Midgame Lookahead (U<=13, K=2)", lam=0.88, exact_threshold=16, enable_midgame_lookahead=True, midgame_u_max=13, midgame_k=2),
        ]
        evaluate_paired(boards, baseline, challengers)


if __name__ == "__main__":
    main()
