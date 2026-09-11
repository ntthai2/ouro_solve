"""
belief_state.py

OT Belief State with constraint propagation and Monte Carlo probability estimation.

Optimizations vs original:
- Internal `_cp_masks`: Dict[color, List[(bitmask, cells)]] — built once per state,
  reused by all methods. Avoids recomputing `sum(1<<x for x in p)` on every call.
- Bitmask filtering in `update()`: O(1) bitwise AND instead of O(run_length) `any(x in set)`.
- Bitmask AND in `certain_safe_cells()`: O(#placements) bitwise instead of set intersection.
- `candidate_placements` remains a public property for backward compat with experiments.py.
"""

from typing import Dict, List, Tuple, Set, Optional
import random
import hashlib
from collections import defaultdict

from ot.board_generator import (
    NUM_CELLS,
    COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW,
    COLOR_ORANGE, COLOR_WHITE, COLOR_BLACK,
    RUN_LENGTHS, RARE_COLORS, RARE_COLOR_WEIGHTS, M_PROBABILITIES,
    enumerate_line_placements
)

# ── Bitmask utilities ────────────────────────────────────────────────────────

def _cells_to_mask(cells) -> int:
    """Convert an iterable of cell indices to a bitmask."""
    mask = 0
    for c in cells:
        mask |= (1 << c)
    return mask

def _mask_to_set(mask: int) -> Set[int]:
    """Extract a set of cell indices from a bitmask."""
    result = set()
    idx = 0
    while mask:
        if mask & 1:
            result.add(idx)
        mask >>= 1
        idx += 1
    return result

# ── Helpers ───────────────────────────────────────────────────────────────────

def _deterministic_seed(revealed: Optional[Dict[int, int]]) -> int:
    """Generate a stable deterministic seed from the revealed cell-color mapping."""
    if not revealed:
        return 42
    rev_tuple = tuple(sorted(revealed.items()))
    return int(hashlib.md5(repr(rev_tuple).encode("utf-8")).hexdigest()[:8], 16)

def _sample_rares_subset(pool: List[int], k: int, rng: Optional[random.Random] = None) -> List[int]:
    """Sample k distinct rare colors from pool using weighted sampling without replacement."""
    if k <= 0 or not pool:
        return []
    if rng is None:
        rng = random
    # Copy to avoid mutation; use swap-and-pop for O(1) removal
    p_pool = list(pool)
    weights = [RARE_COLOR_WEIGHTS.get(c, 1.0) for c in p_pool]
    chosen = []
    for _ in range(min(k, len(p_pool))):
        total_w = sum(weights)
        if total_w <= 0:
            selected = rng.choice(p_pool)
            idx = p_pool.index(selected)
        else:
            # rng.choices returns a list; pick first
            selected = rng.choices(p_pool, weights=weights, k=1)[0]
            idx = p_pool.index(selected)
        # Swap-and-pop: O(1) removal
        last = len(p_pool) - 1
        p_pool[idx] = p_pool[last]
        weights[idx] = weights[last]
        p_pool.pop()
        weights.pop()
        chosen.append(selected)
    return chosen

def _sample_n_extra(num_must: int, max_avail: int, rng: Optional[random.Random] = None) -> int:
    """Sample number of additional rare colors given number of already confirmed rare colors."""
    if max_avail <= 0:
        return 0
    if rng is None:
        rng = random
    if num_must == 0:
        choices = [1, 2]
        weights = [M_PROBABILITIES[0], M_PROBABILITIES[1]]
    elif num_must == 1:
        choices = [0, 1]
        weights = [M_PROBABILITIES[0], M_PROBABILITIES[1]]
    else:
        return 0
    valid = [(c, w) for c, w in zip(choices, weights) if c <= max_avail]
    if not valid:
        return 0
    c_list, w_list = zip(*valid)
    return rng.choices(c_list, weights=w_list, k=1)[0]

def _subset_prior_weight(subset_rares: Set[int], fixed_rares: bool = False) -> float:
    """Compute relative prior weight of a specific combination of rare colors."""
    w_colors = 1.0
    for c in subset_rares:
        w_colors *= RARE_COLOR_WEIGHTS.get(c, 1.0)
    if fixed_rares:
        return w_colors
    m = len(subset_rares)
    if m == 1:
        wm = M_PROBABILITIES[0]
    elif m == 2:
        wm = M_PROBABILITIES[1]
    else:
        wm = 1.0
    return wm * w_colors

def _cells_from_mask(mask: int) -> Tuple[int, ...]:
    """Extract cell indices from a bitmask."""
    cells = []
    m = mask
    idx = 0
    while m:
        if m & 1:
            cells.append(idx)
        m >>= 1
        idx += 1
    return tuple(cells)

# ── FastCounterTwoPass ────────────────────────────────────────────────────────

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

# ── OTBeliefState ─────────────────────────────────────────────────────────────

class OTBeliefState:
    """
    Belief state for Ourotrace.

    Internal representation: `_cp_masks` stores placements as (bitmask, cells_tuple)
    for O(1) bitwise filtering. The public `candidate_placements` property exposes
    the original Dict[color, List[Tuple[int, ...]]] format for backward compatibility.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        _cp_masks: Optional[Dict[int, List[Tuple[int, Tuple[int, ...]]]]] = None,
        rare_active: Optional[Set[int]] = None,
        revealed: Optional[Dict[int, int]] = None,
        num_rares: Optional[int] = 1,
    ):
        """
        Internal constructor. Normally you'd start with `OTBeliefState()` and call
        `update()` to get child states. `_cp_masks` is the internal masked format.
        num_rares: Expected number of rare colors on board (1 for 6-color game, 2 for 7-color game).
        """
        if _cp_masks is None:
            # Initial state: build from enumerate_line_placements
            self._cp_masks = {
                c: [(_cells_to_mask(p), p) for p in enumerate_line_placements(RUN_LENGTHS[c])]
                for c in RUN_LENGTHS
            }
        else:
            self._cp_masks = _cp_masks

        self.rare_active = set(RARE_COLORS) if rare_active is None else set(rare_active)
        self.revealed = {} if revealed is None else dict(revealed)
        self.num_rares = num_rares

    @property
    def candidate_placements(self) -> Dict[int, List[Tuple[int, ...]]]:
        """
        Public backward-compatible view: Dict[color, List[cells_tuple]].
        Used by experiments.py and strategies._score_safe_cell.
        """
        return {c: [cells for _, cells in v] for c, v in self._cp_masks.items()}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, cell: int, color: int) -> 'OTBeliefState':
        """Return a new belief state after observing cell=color."""
        new_revealed = dict(self.revealed)
        new_revealed[cell] = color

        # Build new _cp_masks with bitmask filtering (O(1) per placement)
        cell_mask = 1 << cell
        new_cp: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = {}

        if color == COLOR_BLUE:
            # cell cannot be in ANY placement
            for c, placements in self._cp_masks.items():
                new_cp[c] = [(m, p) for (m, p) in placements if not (m & cell_mask)]
        else:
            for c, placements in self._cp_masks.items():
                if c == color:
                    # cell MUST be in this color's placement
                    new_cp[c] = [(m, p) for (m, p) in placements if m & cell_mask]
                else:
                    # cell CANNOT be in any other color's placement
                    new_cp[c] = [(m, p) for (m, p) in placements if not (m & cell_mask)]

        new_rare_active = set(self.rare_active)

        # If a rare color has no valid placements, it cannot be active
        for rare in list(new_rare_active):
            if not new_cp.get(rare):
                new_rare_active.discard(rare)
                new_cp.pop(rare, None)

        # Instant rare color deduction when num_rares is known:
        # If we have reached the exact quota of rare colors, all other rares are impossible!
        revealed_rares = {c for c in new_revealed.values() if c in RARE_COLORS}
        if self.num_rares is not None and len(revealed_rares) >= self.num_rares:
            for rare in list(new_rare_active):
                if rare not in revealed_rares:
                    new_rare_active.discard(rare)
                    new_cp.pop(rare, None)

        # Constraint propagation loop (naked singles) — bitmask accelerated
        changed = True
        while changed:
            changed = False
            for c, placements in list(new_cp.items()):
                if len(placements) == 1:
                    # This color's placement is perfectly known — build certain mask once
                    certain_mask = placements[0][0]  # already a bitmask
                    for other_c in list(new_cp.keys()):
                        if other_c == c:
                            continue
                        original_len = len(new_cp[other_c])
                        # O(1) bitwise check: keep placements that don't overlap
                        new_cp[other_c] = [
                            (m, p) for (m, p) in new_cp[other_c]
                            if not (m & certain_mask)
                        ]
                        if len(new_cp[other_c]) < original_len:
                            changed = True

            # Rare colors eliminated by propagation
            for rare in list(new_rare_active):
                if not new_cp.get(rare):
                    new_rare_active.discard(rare)
                    new_cp.pop(rare, None)
                    changed = True

        return OTBeliefState(new_cp, new_rare_active, new_revealed, num_rares=self.num_rares)

    # ── Certain Safe Cells ────────────────────────────────────────────────────

    def certain_safe_cells(self) -> Set[int]:
        """
        Returns cells that are DEFINITELY not blue.
        A cell is definitely safe if it appears in ALL valid placements of
        some definitely-active color.
        Uses bitmask AND for O(#placements) intersection instead of set operations.
        """
        safe_mask = 0

        # Colors that are definitely active: base + revealed rare colors
        definitely_active = {COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE}
        for c in self.revealed.values():
            if c in RARE_COLORS:
                definitely_active.add(c)

        # Deduction: If num_rares is known and remaining candidate rares exactly equals needed rares
        # (e.g. 1 rare needed and only 1 rare has valid placements left -> it MUST be active!)
        if self.num_rares is not None:
            revealed_rares = {c for c in self.revealed.values() if c in RARE_COLORS}
            unrevealed_needed = self.num_rares - len(revealed_rares)
            unrevealed_candidates = self.rare_active - revealed_rares
            if unrevealed_needed > 0 and len(unrevealed_candidates) == unrevealed_needed:
                for c in unrevealed_candidates:
                    definitely_active.add(c)

        for c in definitely_active:
            placements = self._cp_masks.get(c)
            if not placements:
                continue
            # Intersection via bitwise AND across all placements
            common = placements[0][0]
            for m, _ in placements[1:]:
                common &= m
                if common == 0:
                    break
            safe_mask |= common

        # Remove already-revealed cells
        revealed_mask = 0
        for cell in self.revealed:
            revealed_mask |= (1 << cell)
        safe_mask &= ~revealed_mask

        # Convert mask to set of cell indices
        return _mask_to_set(safe_mask)

    # ── Probability Estimation ────────────────────────────────────────────────

    def p_blue_all(self, use_exact_endgame: bool = True, n_samples: int = 1000) -> List[float]:
        """
        Calculate the probability of being Blue for ALL cells simultaneously.
        Uses precomputed _cp_masks to avoid redundant bitmask computation.
        """
        probs = [0.0] * NUM_CELLS

        # Pre-fill already revealed cells
        unrevealed = []
        for c in range(NUM_CELLS):
            if c in self.revealed:
                probs[c] = 1.0 if self.revealed[c] == COLOR_BLUE else 0.0
            else:
                unrevealed.append(c)

        exact_threshold = 18 if (self.num_rares == 1) else 16
        if use_exact_endgame and len(unrevealed) <= exact_threshold:
            return self._p_blue_exact(probs, unrevealed)

        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE]

        # Reuse _cp_masks directly — no recomputation
        masks = {c: self._cp_masks.get(c, []) for c in base_colors + list(self.rare_active)}

        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)

        blue_counts = [0] * NUM_CELLS
        successes = 0
        rng = random.Random(_deterministic_seed(self.revealed))

        trials = 0
        while successes < n_samples and trials < n_samples * 10:
            trials += 1

            optional_rares = list(self.rare_active - must_have)
            num_must = len(must_have & set(RARE_COLORS))

            if self.num_rares is not None:
                needed = self.num_rares - num_must
                if needed < 0 or needed > len(optional_rares):
                    continue
                chosen_extra = _sample_rares_subset(optional_rares, needed, rng=rng)
            else:
                min_extra = max(0, 1 - num_must)
                max_extra = min(len(optional_rares), 2 - num_must)
                if min_extra > max_extra:
                    continue
                n_extra = _sample_n_extra(num_must, max_extra, rng=rng)
                chosen_extra = _sample_rares_subset(optional_rares, n_extra, rng=rng)

            active_this_sample = base_colors + list(must_have & set(RARE_COLORS)) + chosen_extra

            used_mask = 0
            fail = False
            for c in active_this_sample:
                valid_p = [m for m, _ in masks.get(c, []) if not (used_mask & m)]
                if not valid_p:
                    fail = True
                    break
                chosen_mask = rng.choice(valid_p)
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
        Returns marginals[color][cell] = P(cell is color). COLOR_BLUE included.
        Uses precomputed _cp_masks to avoid redundant bitmask computation.
        """
        probs = defaultdict(lambda: [0.0] * NUM_CELLS)

        # Pre-fill already revealed cells
        unrevealed = []
        for c in range(NUM_CELLS):
            if c in self.revealed:
                probs[self.revealed[c]][c] = 1.0
            else:
                unrevealed.append(c)

        exact_threshold = 18 if (self.num_rares == 1) else 16
        if use_exact_endgame and len(unrevealed) <= exact_threshold:
            return self._p_color_exact(probs, unrevealed)

        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE]

        # Reuse _cp_masks directly — no recomputation
        masks = {c: self._cp_masks.get(c, []) for c in base_colors + list(self.rare_active)}

        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)

        color_counts = defaultdict(lambda: [0] * NUM_CELLS)
        successes = 0
        rng = random.Random(_deterministic_seed(self.revealed))

        trials = 0
        while successes < n_samples and trials < n_samples * 10:
            trials += 1

            optional_rares = list(self.rare_active - must_have)
            num_must = len(must_have & set(RARE_COLORS))

            if self.num_rares is not None:
                needed = self.num_rares - num_must
                if needed < 0 or needed > len(optional_rares):
                    continue
                chosen_extra = _sample_rares_subset(optional_rares, needed, rng=rng)
            else:
                min_extra = max(0, 1 - num_must)
                max_extra = min(len(optional_rares), 2 - num_must)
                if min_extra > max_extra:
                    continue
                n_extra = _sample_n_extra(num_must, max_extra, rng=rng)
                chosen_extra = _sample_rares_subset(optional_rares, n_extra, rng=rng)

            active_this_sample = base_colors + list(must_have & set(RARE_COLORS)) + chosen_extra

            used_mask = 0
            fail = False
            cell_colors = {}
            for c in active_this_sample:
                valid_pairs = [(m, p) for m, p in masks.get(c, []) if not (used_mask & m)]
                if not valid_pairs:
                    fail = True
                    break
                chosen_m, chosen_p = rng.choice(valid_pairs)
                used_mask |= chosen_m
                for cell in chosen_p:
                    cell_colors[cell] = c

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

    # ── Exact DP methods ──────────────────────────────────────────────────────

    def _p_blue_exact(self, probs: List[float], unrevealed: List[int]) -> List[float]:
        if not unrevealed:
            return probs

        counter = FastCounterTwoPass()
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE]
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)

        optional_rares = list(self.rare_active - must_have)
        num_must = len(must_have & set(RARE_COLORS))

        # Use _cp_masks directly — no recomputation
        masks_dict = {c: self._cp_masks[c] for c in base_colors + list(self.rare_active) if c in self._cp_masks}

        total_valid = 0
        total_blue_counts = [0] * NUM_CELLS

        from itertools import combinations
        valid_subsets = []
        if self.num_rares is not None:
            needed = self.num_rares - num_must
            if 0 <= needed <= len(optional_rares):
                for extra in combinations(optional_rares, needed):
                    subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                    valid_subsets.append(subset)
        else:
            min_extra = max(0, 1 - num_must)
            max_extra = min(len(optional_rares), 2 - num_must)
            for n_extra in range(min_extra, max_extra + 1):
                for extra in combinations(optional_rares, n_extra):
                    subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                    valid_subsets.append(subset)

        for subset in valid_subsets:
            if not all(c in masks_dict for c in subset):
                continue
            rares_in_subset = set(subset) & (set(RARE_COLORS) | self.rare_active)
            w_prior = _subset_prior_weight(rares_in_subset, fixed_rares=(self.num_rares is not None))
            ways, marginals = counter.count(subset, masks_dict, 0)
            if ways > 0:
                weighted_ways = ways * w_prior
                total_valid += weighted_ways
                for i in range(NUM_CELLS):
                    covered_ways = sum(marginals[c][i] for c in subset)
                    total_blue_counts[i] += (ways - covered_ways) * w_prior

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
        base_colors = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE]
        must_have = set(base_colors)
        for rv in self.revealed.values():
            if rv in RARE_COLORS:
                must_have.add(rv)

        optional_rares = list(self.rare_active - must_have)
        num_must = len(must_have & set(RARE_COLORS))

        # Use _cp_masks directly — no recomputation
        masks_dict = {c: self._cp_masks[c] for c in base_colors + list(self.rare_active) if c in self._cp_masks}

        total_valid = 0
        total_color_counts = defaultdict(lambda: [0] * NUM_CELLS)

        from itertools import combinations
        valid_subsets = []
        if self.num_rares is not None:
            needed = self.num_rares - num_must
            if 0 <= needed <= len(optional_rares):
                for extra in combinations(optional_rares, needed):
                    subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                    valid_subsets.append(subset)
        else:
            min_extra = max(0, 1 - num_must)
            max_extra = min(len(optional_rares), 2 - num_must)
            for n_extra in range(min_extra, max_extra + 1):
                for extra in combinations(optional_rares, n_extra):
                    subset = base_colors + list(must_have & set(RARE_COLORS)) + list(extra)
                    valid_subsets.append(subset)

        for subset in valid_subsets:
            if not all(c in masks_dict for c in subset):
                continue
            rares_in_subset = set(subset) & (set(RARE_COLORS) | self.rare_active)
            w_prior = _subset_prior_weight(rares_in_subset, fixed_rares=(self.num_rares is not None))
            ways, marginals = counter.count(subset, masks_dict, 0)
            if ways > 0:
                weighted_ways = ways * w_prior
                total_valid += weighted_ways
                for i in range(NUM_CELLS):
                    covered_ways = 0
                    for c in subset:
                        w = marginals[c][i] * w_prior
                        covered_ways += w
                        total_color_counts[c][i] += w
                    total_color_counts[COLOR_BLUE][i] += (ways * w_prior - covered_ways)

        if total_valid == 0:
            if COLOR_BLUE not in probs:
                probs[COLOR_BLUE] = [0.0] * NUM_CELLS
            for c in unrevealed:
                probs[COLOR_BLUE][c] = 1.0
            return probs

        for c, counts in total_color_counts.items():
            if c not in probs:
                probs[c] = [0.0] * NUM_CELLS
            for cell in unrevealed:
                probs[c][cell] = counts[cell] / total_valid

        return probs

