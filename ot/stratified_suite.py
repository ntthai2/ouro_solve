import os, sys, random, pickle, numpy as np
from typing import List

sys.path.insert(0, r"d:\Downloads\ouro_solve")

from ot.board_generator import (
    COLOR_BLUE, COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE,
    COLOR_WHITE, COLOR_BLACK, COLOR_RED, COLOR_RAINBOW,
    RUN_LENGTHS, PLACEMENTS_CACHE, NUM_CELLS
)

def generate_board_with_exact_rares(rares: List[int]) -> np.ndarray:
    """Generate a valid 5x5 board with exactly the specified rare colors."""
    colors_to_place = [COLOR_TEAL, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE] + rares
    board = np.full(NUM_CELLS, COLOR_BLUE, dtype=np.uint8)
    
    def backtrack(idx, used):
        if idx == len(colors_to_place):
            return True
        col = colors_to_place[idx]
        placements = [p for p in PLACEMENTS_CACHE[RUN_LENGTHS[col]] if not used.intersection(p)]
        random.shuffle(placements)
        for p in placements:
            for c in p:
                board[c] = col
                used.add(c)
            if backtrack(idx + 1, used):
                return True
            for c in p:
                board[c] = COLOR_BLUE
                used.remove(c)
        return False
        
    while True:
        if backtrack(0, set()):
            return board.copy()

def build_representative_benchmark_suite(total_n: int = 200, seed: int = 2026) -> List[np.ndarray]:
    """
    Build a mathematically stratified benchmark suite mirroring the true empirical distribution:
    - 67% m_extra = 1 (134 boards):
        * Orange + White: 58 boards (~43%)
        * Orange + Black: 51 boards (~38%)
        * Orange + Red: 13 boards (~9.5%)
        * Orange + Rainbow: 12 boards (~9.5%)
    - 33% m_extra = 2 (66 boards):
        * White + Black: 28 boards
        * White + Red: 8 boards
        * White + Rainbow: 8 boards
        * Black + Red: 7 boards
        * Black + Rainbow: 7 boards
        * Others / Rare combos: 8 boards
    """
    random.seed(seed)
    np.random.seed(seed)
    
    quota = [
        # m_extra = 1 (75% -> 150 boards, White 49%, Black 49%, Red 1%, Rainbow 1%)
        ([COLOR_WHITE], 74),
        ([COLOR_BLACK], 74),
        ([COLOR_RED], 1),
        ([COLOR_RAINBOW], 1),
        # m_extra = 2 (25% -> 50 boards, White+Black ~96%)
        ([COLOR_WHITE, COLOR_BLACK], 48),
        ([COLOR_WHITE, COLOR_RED], 1),
        ([COLOR_BLACK, COLOR_RAINBOW], 1),
    ]
    
    suite = []
    print(f"Generating Stratified Representative Benchmark Suite (N={total_n})...")
    for rares, count in quota:
        for _ in range(count):
            suite.append(generate_board_with_exact_rares(rares))
            
    print(f"Successfully constructed {len(suite)} representative boards.")
    return suite

if __name__ == "__main__":
    suite = build_representative_benchmark_suite(200)
    out_path = r"d:\Downloads\ouro_solve\cache\ot_representative_suite_200.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(suite, f)
    print(f"Saved to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
