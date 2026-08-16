"""
strategies.py

Strategies for OT mode.
"""

import random
from typing import List, Set

from ot.board_generator import COLOR_BLUE

class OTHybridStrategy:
    """
    Two-phase strategy:
    1. Deterministic: if any cell is certain to be non-blue, pick it.
    2. Probabilistic: pick the cell with the lowest probability of being blue.
    """
    def __init__(self, use_exact_endgame: bool = True, n_samples: int = 1000):
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
        # Phase 1: Certain safe cells
        safe_cells = belief.certain_safe_cells()
        valid_safe = [c for c in safe_cells if c in remaining]
        
        if valid_safe:
            # Pick one randomly among the certain safe cells
            return random.choice(valid_safe)
            
        # Phase 2: Lowest p_blue
        best_cell = -1
        min_p = 2.0
        
        probs = belief.p_blue_all(use_exact_endgame=self.use_exact_endgame, n_samples=self.n_samples)
        for c in remaining:
            if probs[c] < min_p:
                min_p = probs[c]
                best_cell = c
                
        return best_cell

class OTInfoGainStrategy:
    """
    Trade-off between risk and information gain.
    Score = -lam * p_blue + (1 - lam) * E[new_safe_cells]
    """
    def __init__(self, lam: float = 0.7, use_exact_endgame: bool = True, n_samples: int = 1000):
        self.lam = lam
        self.use_exact_endgame = use_exact_endgame
        self.n_samples = n_samples
        self.name = f"InfoGain(lam={lam:.2f}, {n_samples} MC)"
        
    def __call__(self, belief, remaining: List[int]) -> int:
        # Phase 1: Certain safe cells
        safe_cells = belief.certain_safe_cells()
        valid_safe = sorted([c for c in safe_cells if c in remaining])
        
        if valid_safe:
            return valid_safe[0]
            
        # Phase 2: InfoGain / Lowest p_blue
        probs = belief.p_color_all(use_exact_endgame=self.use_exact_endgame, n_samples=self.n_samples)
        
        best_cell = -1
        max_score = -float('inf')
        
        for cell in sorted(remaining):
            p_blue = probs[COLOR_BLUE][cell]
            
            e_info = 0.0
            if self.lam < 1.0:
                for c, p_c_list in probs.items():
                    if c != COLOR_BLUE:
                        p_c = p_c_list[cell]
                        if p_c > 0:
                            new_belief = belief.update(cell, c)
                            new_safe = new_belief.certain_safe_cells()
                            valid_new_safe = [s for s in new_safe if s in remaining and s != cell]
                            e_info += p_c * len(valid_new_safe)
                            
            score = -self.lam * p_blue + (1.0 - self.lam) * e_info
            
            if score > max_score:
                max_score = score
                best_cell = cell
                
        return best_cell

class RandomStrategy:
    def __init__(self):
        self.name = "Random Baseline"
        
    def __call__(self, belief, remaining: List[int]) -> int:
        return random.choice(remaining)
