"""
strategies.py

Strategies for OT mode.
"""

import random
from typing import List, Set

from ot.board_generator import COLOR_BLUE, RUN_LENGTHS

def _score_safe_cell(belief, cell: int) -> tuple:
    """
    Score a guaranteed safe cell by its potential to resolve longer runs
    and trigger maximum constraint propagation.
    """
    max_len = 0
    overlap_count = 0
    for c, placements in belief.candidate_placements.items():
        cell_placements = [p for p in placements if cell in p]
        if cell_placements:
            l = RUN_LENGTHS.get(c, 2)
            if l > max_len:
                max_len = l
            overlap_count += len(cell_placements)
    # Proximity to center C3 (row 2, col 2)
    r, c_col = cell // 5, cell % 5
    center_dist = abs(r - 2) + abs(c_col - 2)
    return (max_len, overlap_count, -center_dist)

class OTHybridStrategy:
    """
    Two-phase strategy:
    1. Deterministic: if any cell is certain to be non-blue, pick the most informative one.
    2. Probabilistic: pick the cell with the lowest probability of being blue.
    """
    def __init__(self, use_exact_endgame: bool = True, n_samples: int = 5000):
        self.use_exact_endgame = use_exact_endgame
        self.n_samples = n_samples
        
        name_parts = []
        if use_exact_endgame:
            name_parts.append("Exact Endgame")
        else:
            name_parts.append("MC Only")
        name_parts.append(f"{n_samples} samples")
            
        self.name = f"Hybrid Strategy ({', '.join(name_parts)})"
        
    def __call__(self, belief, remaining: List[int]) -> int:
        # Phase 1: Certain safe cells (deterministic, most informative first)
        safe_cells = belief.certain_safe_cells()
        valid_safe = [c for c in safe_cells if c in remaining]
        
        if valid_safe:
            return max(valid_safe, key=lambda c: _score_safe_cell(belief, c))
            
        # Phase 2: Lowest p_blue
        best_cell = -1
        min_p = 2.0
        
        probs = belief.p_blue_all(use_exact_endgame=self.use_exact_endgame, n_samples=self.n_samples)
        for c in remaining:
            if probs[c] < min_p:
                min_p = probs[c]
                best_cell = c
                
        return best_cell

# Optimal Move 2 responses after opening C3 (Cell 12) to eliminate MC noise at Move 2
MOVE2_OPENING_BOOK = {
    COLOR_BLUE: 16,   # B4 (safe diagonal reflection)
    1: 13,            # Teal -> D3 (orthogonal run extension)
    2: 7,             # Green -> C2 (orthogonal run extension)
    3: 17,            # Yellow -> C4 (orthogonal run extension)
    4: 17,            # Orange -> C4 (orthogonal run extension)
    5: 13,            # White -> D3 (orthogonal extension)
    6: 13,            # Black -> D3 (orthogonal extension)
}

class OTInfoGainStrategy:
    """
    2-Step Lookahead Value of Information (VOI) strategy with K-best pruning:
    - Phase 1: Guaranteed safe cells (deterministic topological ranking).
    - Phase 2: Joint-safe cells (P(Blue) == 0.0 across all consistent boards).
    - Phase 3: 1-step rapid screening across all cells + dynamic 2-step lookahead in endgame (<=10 cells)
      to anticipate future constraint propagation and avoid hazard traps.
    - Zero cache footprint (<10MB limit), response latency < 30ms (<200ms limit).
    """
    def __init__(self, lam: float = 0.88, use_exact_endgame: bool = True, n_samples: int = 3500, k_prune: int = 1):
        self.lam = lam
        self.use_exact_endgame = use_exact_endgame
        self.n_samples = n_samples
        self.k_prune = k_prune
        self.name = f"Optimized_VOI(lam={lam:.2f}, {n_samples} MC)"
        
    def __call__(self, belief, remaining: List[int]) -> int:
        # Move 1: Pinned C3
        if len(belief.revealed) == 0:
            return 12
            
        # Move 2: Opening Book for C3 (0.00ms, 0% MC noise)
        if len(belief.revealed) == 1 and 12 in belief.revealed:
            col_at_12 = belief.revealed[12]
            if col_at_12 in MOVE2_OPENING_BOOK:
                rec_m2 = MOVE2_OPENING_BOOK[col_at_12]
                if rec_m2 in remaining:
                    return rec_m2

        # Phase 1: Certain safe cells (deterministic, most informative first)
        safe_cells = belief.certain_safe_cells()
        valid_safe = [c for c in safe_cells if c in remaining]
        
        if valid_safe:
            return max(valid_safe, key=lambda c: _score_safe_cell(belief, c))
            
        # Phase 2: Posterior probabilities
        probs = belief.p_color_all(use_exact_endgame=self.use_exact_endgame, n_samples=self.n_samples)
        
        # Prioritize joint-safe cells (P(Blue) == 0.0) across combinations
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
            
        # Phase 3: 1-Step Screening across all remaining cells
        unrev = len([c for c in range(25) if c not in belief.revealed])
        step1_scores = []
        for c in remaining:
            p_b = probs[COLOR_BLUE][c]
            e_info = 0.0
            for col_id, p_list in probs.items():
                if col_id != COLOR_BLUE and p_list[c] > 0:
                    nb = belief.update(c, col_id)
                    v_ns = [s for s in nb.certain_safe_cells() if s in remaining and s != c]
                    e_info += p_list[c] * len(v_ns)
            s1 = -self.lam * p_b + (1.0 - self.lam) * e_info
            step1_scores.append((s1, c))
            
        step1_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Dynamic K-prune: In endgame (<=10 unrevealed), enable 2-step lookahead (takes <30ms)
        active_k_prune = 3 if unrev <= 10 else self.k_prune

        # Fast path if top candidate dominates, trivial choice, or active_k_prune <= 1
        if len(step1_scores) <= 1 or active_k_prune <= 1 or (step1_scores[0][0] - step1_scores[1][0] > 0.25):
            return step1_scores[0][1]
            
        # Phase 4: 2-Step Lookahead on Top-K candidates
        top_k = [c for _, c in step1_scores[:active_k_prune]]
        best_c = top_k[0]
        best_2step = -float('inf')
        
        for c in top_k:
            p_b = probs[COLOR_BLUE][c]
            imm_hazard = -self.lam * p_b
            
            future_val = 0.0
            for col_id, p_list in probs.items():
                p_c = p_list[c]
                if col_id != COLOR_BLUE and p_c > 0:
                    nb = belief.update(c, col_id)
                    ns = [s for s in nb.certain_safe_cells() if s in remaining and s != c]
                    n_safe = len(ns)
                    if n_safe > 0:
                        future_val += p_c * (n_safe * 1.4)
                    else:
                        rem_next = [x for x in remaining if x != c]
                        if rem_next and unrev <= 16:
                            p_next = nb.p_blue_all(use_exact_endgame=True)
                            min_next_pb = min(p_next[x] for x in rem_next)
                            future_val += p_c * (1.0 - min_next_pb)
                            
            tot_2step = imm_hazard + (1.0 - self.lam) * future_val
            tie = _score_safe_cell(belief, c)
            full_score = (tot_2step, tie[0], tie[1], tie[2])
            
            if full_score > (best_2step, -1, -1, -100):
                best_2step = tot_2step
                best_c = c
                
        return best_c

class RandomStrategy:
    def __init__(self):
        self.name = "Random Baseline"
        
    def __call__(self, belief, remaining: List[int]) -> int:
        return random.choice(remaining)
