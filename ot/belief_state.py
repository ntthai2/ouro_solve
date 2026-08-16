"""
belief_state.py

OT Belief State with constraint propagation and Monte Carlo probability estimation.
"""

from typing import Dict, List, Tuple, Set, Optional
import random
from collections import defaultdict

from ot.board_generator import (
    NUM_CELLS,
    COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW,
    COLOR_ORANGE, COLOR_WHITE, COLOR_BLACK,
    RUN_LENGTHS, RARE_COLORS,
    enumerate_line_placements
)

class FastCounterTwoPass:
    def __init__(self):
        self.memo = {}

    def count(self, subset_colors, candidate_masks, initial_used_mask=0):
        self.memo.clear()
        
        def get_ways(color_idx, used_mask):
            if color_idx == len(subset_colors): return 1
            state = (color_idx, used_mask)
            if state in self.memo: return self.memo[state]
            
            w = 0
            for p_mask, _ in candidate_masks[subset_colors[color_idx]]:
                if not (used_mask & p_mask):
                    w += get_ways(color_idx + 1, used_mask | p_mask)
            self.memo[state] = w
            return w
            
        total_ways = get_ways(0, initial_used_mask)
        if total_ways == 0:
            return 0, defaultdict(lambda: [0]*NUM_CELLS)
            
        path_counts = defaultdict(int)
        path_counts[(0, initial_used_mask)] = 1
        
        # marginals[color][cell] = ways
        marginals = defaultdict(lambda: [0]*NUM_CELLS)
        
        for color_idx in range(len(subset_colors)):
            color = subset_colors[color_idx]
            level_states = [(u_mask, count) for (c_idx, u_mask), count in path_counts.items() if c_idx == color_idx]
            
            for used_mask, p_count in level_states:
                if p_count == 0: continue
                for p_mask, p_cells in candidate_masks[color]:
                    if not (used_mask & p_mask):
                        next_mask = used_mask | p_mask
                        ways_to_finish = get_ways(color_idx + 1, next_mask)
                        if ways_to_finish > 0:
                            occurrences = p_count * ways_to_finish
                            path_counts[(color_idx + 1, next_mask)] += p_count
                            for c in p_cells:
                                marginals[color][c] += occurrences
                                
        return total_ways, marginals

class OTBeliefState:
    def __init__(self, candidate_placements: Optional[Dict[int, List[Tuple[int, ...]]]] = None, rare_active: Optional[Set[int]] = None, revealed: Optional[Dict[int, int]] = None):
        if candidate_placements is None:
            self.candidate_placements = {
                c: enumerate_line_placements(RUN_LENGTHS[c]) for c in RUN_LENGTHS
            }
        else:
            self.candidate_placements = {k: list(v) for k, v in candidate_placements.items()}
            
        self.rare_active = set(RARE_COLORS) if rare_active is None else set(rare_active)
        self.revealed = {} if revealed is None else dict(revealed)

    def update(self, cell: int, color: int) -> 'OTBeliefState':
        """Return a new belief state after observing cell=color."""
        new_revealed = dict(self.revealed)
        new_revealed[cell] = color
        
        new_candidates = {k: list(v) for k, v in self.candidate_placements.items()}
        new_rare_active = set(self.rare_active)
        
        if color == COLOR_BLUE:
            # Cell cannot be in ANY placement
            for c in list(new_candidates.keys()):
                new_candidates[c] = [p for p in new_candidates[c] if cell not in p]
        else:
            # Cell MUST be in the observed color's placement
            if color in new_candidates:
                new_candidates[color] = [p for p in new_candidates[color] if cell in p]
            # Cell CANNOT be in any OTHER color's placement
            for c in list(new_candidates.keys()):
                if c != color:
                    new_candidates[c] = [p for p in new_candidates[c] if cell not in p]
                    
        # If a rare color has no valid placements left, it cannot be active
        for rare in list(new_rare_active):
            if not new_candidates[rare]:
                new_rare_active.remove(rare)
                del new_candidates[rare]
                
        # Constraint propagation loop (naked singles)
        changed = True
        while changed:
            changed = False
            for c, placements in new_candidates.items():
                if len(placements) == 1:
                    # This color's placement is perfectly known
                    certain_cells = placements[0]
                    for other_c in new_candidates:
                        if other_c != c:
                            original_len = len(new_candidates[other_c])
                            # Remove placements that intersect with the certain placement
                            new_candidates[other_c] = [
                                p for p in new_candidates[other_c] 
                                if not any(x in certain_cells for x in p)
                            ]
                            if len(new_candidates[other_c]) < original_len:
                                changed = True
            
            # Check if any active rare colors were eliminated by propagation
            for rare in list(new_rare_active):
                if not new_candidates[rare]:
                    new_rare_active.remove(rare)
                    del new_candidates[rare]
                    changed = True

        return OTBeliefState(new_candidates, new_rare_active, new_revealed)

    def certain_safe_cells(self) -> Set[int]:
        """
        Returns cells that are DEFINITELY not blue, based on current constraints.
        A cell is definitely not blue if it is present in ALL valid candidate placements
        for some color that is definitely active.
        """
        safe = set()
        
        # Colors that are definitely active: Teal, Green, Yellow + revealed rare colors
        definitely_active = {COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW}
        for c in self.revealed.values():
            if c in RARE_COLORS:
                definitely_active.add(c)
                
        for c in definitely_active:
            placements = self.candidate_placements.get(c, [])
            if not placements:
                continue
            # Intersection of all placements
            common_cells = set(placements[0])
            for p in placements[1:]:
                common_cells.intersection_update(p)
                if not common_cells:
                    break
            safe.update(common_cells)
            
        # Also remove already revealed cells from the return set
        return safe - set(self.revealed.keys())

    def p_blue_all(self, use_exact_endgame: bool = True, n_samples: int = 1000) -> List[float]:
        """
        Calculate the probability of being Blue for ALL cells simultaneously.
        """
        probs = [0.0] * NUM_CELLS
        
        # Pre-fill already revealed cells
        unrevealed = []
        for c in range(NUM_CELLS):
            if c in self.revealed:
                probs[c] = 1.0 if self.revealed[c] == COLOR_BLUE else 0.0
            else:
                unrevealed.append(c)
                
        if use_exact_endgame and len(unrevealed) <= 8:
            return self._p_blue_exact(probs, unrevealed)
            
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW]
        blue_counts = [0] * NUM_CELLS
        successes = 0
        
        masks = {
            c: [(sum(1<<x for x in p), p) for p in self.candidate_placements.get(c, [])]
            for c in base_colors + list(self.rare_active)
        }
        
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)
                
        trials = 0
        while successes < n_samples and trials < n_samples * 10:
            trials += 1
            
            optional_rares = list(self.rare_active - must_have)
            num_must = len(must_have & set(RARE_COLORS))
            min_extra = max(0, 1 - num_must)
            max_extra = min(len(optional_rares), 3 - num_must)
            
            if min_extra > max_extra:
                continue
                
            n_extra = random.randint(min_extra, max_extra)
            chosen_extra = random.sample(optional_rares, n_extra)
            
            active_this_sample = base_colors + list(must_have & set(RARE_COLORS)) + chosen_extra
            
            used_mask = 0
            fail = False
            for c in active_this_sample:
                valid_p = [m for m, p in masks[c] if not (used_mask & m)]
                if not valid_p:
                    fail = True
                    break
                chosen_mask = random.choice(valid_p)
                used_mask |= chosen_mask
                
            if not fail:
                successes += 1
                for cell in unrevealed:
                    if not (used_mask & (1 << cell)):
                        blue_counts[cell] += 1
                        
        if successes == 0:
            for cell in unrevealed:
                probs[cell] = 1.0
            return probs
            
        for cell in unrevealed:
            probs[cell] = blue_counts[cell] / successes
            
        return probs

    def p_color_all(self, use_exact_endgame: bool = True, n_samples: int = 1000) -> Dict[int, List[float]]:
        """
        Calculate the probability of being each color for ALL cells.
        Returns marginals[color][cell] = P(cell is color).
        color = COLOR_BLUE is included.
        """
        probs = defaultdict(lambda: [0.0] * NUM_CELLS)
        
        # Pre-fill already revealed cells
        unrevealed = []
        for c in range(NUM_CELLS):
            if c in self.revealed:
                probs[self.revealed[c]][c] = 1.0
            else:
                unrevealed.append(c)
                
        if use_exact_endgame and len(unrevealed) <= 8:
            return self._p_color_exact(probs, unrevealed)
            
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW]
        color_counts = defaultdict(lambda: [0] * NUM_CELLS)
        successes = 0
        
        masks = {
            c: [(sum(1<<x for x in p), p) for p in self.candidate_placements.get(c, [])]
            for c in base_colors + list(self.rare_active)
        }
        
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)
                
        trials = 0
        while successes < n_samples and trials < n_samples * 10:
            trials += 1
            
            optional_rares = list(self.rare_active - must_have)
            num_must = len(must_have & set(RARE_COLORS))
            min_extra = max(0, 1 - num_must)
            max_extra = min(len(optional_rares), 3 - num_must)
            
            if min_extra > max_extra:
                continue
                
            n_extra = random.randint(min_extra, max_extra)
            chosen_extra = random.sample(optional_rares, n_extra)
            
            active_this_sample = base_colors + list(must_have & set(RARE_COLORS)) + chosen_extra
            
            used_mask = 0
            fail = False
            cell_colors = {}
            for c in active_this_sample:
                valid_p = [m for m, p in masks[c] if not (used_mask & m)]
                if not valid_p:
                    fail = True
                    break
                chosen_mask = random.choice(valid_p)
                used_mask |= chosen_mask
                # Find which cells this covers
                # masks[c] has (m, p)
                for m, p in masks[c]:
                    if m == chosen_mask:
                        for cell in p:
                            cell_colors[cell] = c
                        break
                
            if not fail:
                successes += 1
                for cell in unrevealed:
                    if used_mask & (1 << cell):
                        color_counts[cell_colors[cell]][cell] += 1
                    else:
                        color_counts[COLOR_BLUE][cell] += 1
                        
        if successes == 0:
            for cell in unrevealed:
                probs[COLOR_BLUE][cell] = 1.0
            return probs
            
        for c, counts in color_counts.items():
            for cell in unrevealed:
                probs[c][cell] = counts[cell] / successes
                
        return probs

    def _p_blue_exact(self, probs: List[float], unrevealed: List[int]) -> List[float]:
        if not unrevealed:
            return probs
            
        counter = FastCounterTwoPass()
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW]
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)
                
        optional_rares = list(self.rare_active - must_have)
        num_must = len(must_have & set(RARE_COLORS))
        min_extra = max(0, 1 - num_must)
        max_extra = min(len(optional_rares), 3 - num_must)
        
        # We must sum over all valid rare subsets, assuming generator uniform choice over 1..3
        # Wait, the true prior over subsets is a bit complex:
        # P(subset) depends on how generator picks. But for exact counting in endgame,
        # simply weighting all consistent configurations equally is standard and usually very accurate.
        # Let's do a uniform weighting over all globally consistent boards.
        
        total_valid = 0
        total_blue_counts = [0] * NUM_CELLS
        
        from itertools import combinations
        valid_subsets = []
        for n_extra in range(min_extra, max_extra + 1):
            for extra in combinations(optional_rares, n_extra):
                subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                valid_subsets.append(subset)
                
        masks_dict = {}
        for c in base_colors + list(self.rare_active):
            if c in self.candidate_placements:
                masks_dict[c] = [(sum(1<<x for x in p), p) for p in self.candidate_placements[c]]
            
        for subset in valid_subsets:
            # Filter subset to colors that have candidate placements
            if not all(c in masks_dict for c in subset):
                continue
            ways, marginals = counter.count(subset, masks_dict, 0)
            if ways > 0:
                total_valid += ways
                for i in range(NUM_CELLS):
                    covered_ways = sum(marginals[c][i] for c in subset)
                    total_blue_counts[i] += (ways - covered_ways)
                    
        if total_valid == 0:
            for c in unrevealed:
                probs[c] = 1.0
            return probs
            
        for c in unrevealed:
            probs[c] = total_blue_counts[c] / total_valid
            
        return probs

    def _p_color_exact(self, probs: Dict[int, List[float]], unrevealed: List[int]) -> Dict[int, List[float]]:
        if not unrevealed:
            return probs
            
        counter = FastCounterTwoPass()
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW]
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)
                
        optional_rares = list(self.rare_active - must_have)
        num_must = len(must_have & set(RARE_COLORS))
        min_extra = max(0, 1 - num_must)
        max_extra = min(len(optional_rares), 3 - num_must)
        
        total_valid = 0
        total_color_counts = defaultdict(lambda: [0] * NUM_CELLS)
        
        from itertools import combinations
        valid_subsets = []
        for n_extra in range(min_extra, max_extra + 1):
            for extra in combinations(optional_rares, n_extra):
                subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                valid_subsets.append(subset)
                
        masks_dict = {}
        for c in base_colors + list(self.rare_active):
            if c in self.candidate_placements:
                masks_dict[c] = [(sum(1<<x for x in p), p) for p in self.candidate_placements[c]]
            
        for subset in valid_subsets:
            if not all(c in masks_dict for c in subset):
                continue
            ways, marginals = counter.count(subset, masks_dict, 0)
            if ways > 0:
                total_valid += ways
                for i in range(NUM_CELLS):
                    covered_ways = 0
                    for c in subset:
                        w = marginals[c][i]
                        covered_ways += w
                        total_color_counts[c][i] += w
                    total_color_counts[COLOR_BLUE][i] += (ways - covered_ways)
                    
        if total_valid == 0:
            for c in unrevealed:
                probs[COLOR_BLUE][c] = 1.0
            return probs
            
        for c, counts in total_color_counts.items():
            for cell in unrevealed:
                probs[c][cell] = counts[cell] / total_valid
                
        return probs
