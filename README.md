# Ourosphere Strategy Analysis

> Exhaustive evaluation of 16,800 valid board configurations under uniform red position distribution.

---

## 1. Game Overview

The Ourosphere minigame presents a 5×5 grid of colored spheres. The player has 5 sequential clicks to reveal spheres and maximize their total score. Each click reveals a color, and the revealed color provides geometric constraints on where the highest-value sphere — red — is located.

### Grid Composition

Every board contains the following spheres, placed according to strict geometric rules relative to red:

| Color | Count | Value | Placement Rule |
|---|---|---|---|
| Red | 1 | 150 pts | Any non-center cell |
| Orange | 2 | 90 pts | Immediate neighbors of red |
| Yellow | 3 | 55 pts | Full diagonal lines through red |
| Green | 4 | 35 pts | Same row or column as red |
| Teal | Varies | 20 pts | Remaining cells sharing row/col/diagonal with red |
| Blue | Varies | 10 pts | Cells sharing nothing with red |

The center cell (C3) never contains red. The theoretical maximum score per round is **440 points**: red (150) + orange×2 (180) + yellow×2 (110).

---

## 2. Deduction Rules

When a cell is revealed, its color constrains red's possible position. Constraints from multiple reveals combine.

| Revealed color | Constraint on red | Candidates after center reveal |
|---|---|---|
| Orange | Red is an immediate neighbor of the revealed cell | 4 |
| Yellow | Red is on the full diagonal lines through the revealed cell | 8 |
| Green | Red shares the same row or column as the revealed cell | 8 |
| Teal | Red shares the row, column, or diagonal of the revealed cell | Up to 16 |
| Blue | Red shares nothing with the revealed cell | 8 |

---

## 3. Board Statistics

### Valid Configurations

Exhaustive enumeration reveals **16,800 valid board configurations** — the complete state space.

### Red Position Distribution

Red is placed uniformly at random across the 24 non-center cells. However, the number of valid boards per red position varies due to geometric constraints:

| Region | Cells | Boards per position |
|---|---|---|
| Corners | 4 | 60 |
| Outer edges | 12 | 180 |
| Inner ring (surrounding center) | 8 | 1,800 |

This was verified empirically via chi-square test on 46 observed game outcomes (hypothesis A: uniform over red positions, not uniform over board configurations).

### Teal / Blue Counts

- Teal per board: min=3, max=5, mean=4.71
- Blue per board: min=10, max=12, mean=10.29

---

## 4. Strategy Analysis

All strategies evaluated by exact simulation across all 16,800 boards, weighted by the uniform red position distribution.

| Strategy | Expected score | vs Optimal | Score min | P(find red) | pkl size | Precompute |
|---|---|---|---|---|---|---|
| Exact POMDP | 336.98 | — | 200 | 100% | 789 MB | ~33 min |
| VOI Greedy (depth=3) | 336.97 | −0.01 | 200 | 100% | 16.6 MB | ~1 min |
| VOI Greedy (depth=2) | 335.84 | −1.14 | 95 | 99.9% | 1.3 MB | 5 sec |
| VOI Greedy (depth=1) | 328.61 | −8.37 | 70 | 99.5% | 0.1 MB | 0.2 sec |
| Entropy Minimization | 326.52 | −10.46 | 200 | 100% | ~1 MB | ~1 sec |
| Candidate Halving | 325.03 | −11.95 | 190 | 100% | ~1 MB | ~1 sec |
| Baseline (center+random) | 262.01 | −74.97 | ~198 | 98% | — | None |

### Key Findings

**VOI depth=5 is identical to POMDP.** With full-depth lookahead, VOI converges exactly to the POMDP solution — same expected score, same memo table size (394,735 states), same first click. This confirms both are computing the same optimal solution and that depth=5 VOI is mathematically equivalent to exact POMDP for a 5-click game. VOI depth=5 is therefore not listed separately above.

**VOI depth=3 is the practical optimum.** At 16.6 MB and only 0.01 points below POMDP, VOI d=3 is the chosen strategy for the live assistant. It finds red 100% of the time, has a score floor of 200 (never collapses), and is hostable on any free-tier platform.

**VOI depth=2's score minimum of 95 is a disqualifier.** Despite a strong mean score, d=2 occasionally makes a bad early decision that results in near-zero scoring games. The floor matters for a real player.

**Entropy minimization captures 97% of optimal value.** Despite being far simpler than POMDP, entropy minimization loses only 10.46 points per game. It also has a clean score floor of 200 and finds red 100% of the time, making it the best lightweight fallback.

**The baseline loses 75 points per game.** Switching from center-first random play to any informed strategy yields a ~28% improvement in expected score.

**The game has an inherent difficulty floor.** Even optimal play achieves only 337/440 (77% of theoretical maximum). This gap reflects boards where red's position cannot be determined in time to collect nearby orange and yellow spheres within 5 clicks.

### VOI Depth Scaling

| Depth | Memo states | pkl size | Expected score | Precompute |
|---|---|---|---|---|
| 1 | 1 | 0.1 MB | 328.61 | 0.2 sec |
| 2 | 150 | 1.3 MB | 335.84 | 5 sec |
| 3 | 6,265 | 16.6 MB | 336.97 | ~1 min |
| 4 | — | — | ~337 (est.) | ~10–30 min |
| 5 (= POMDP) | 394,735 | 789 MB | 336.98 | ~33 min |

The depth=3 → depth=4 jump would yield at most ~0.01 additional points based on the convergence pattern, making depth=4 not worth computing.

---

## 5. Optimal First Click

The optimal first click — determined by POMDP and confirmed by VOI depth=5 — is **cell B1 (row 1, column B)**, not the center cell C3.

| Strategy | First click | Grid position |
|---|---|---|
| POMDP / VOI depth=5 | Cell 1 | B1 (row 1, col B) |
| VOI depth=3 | Cell 15 | D3 (row 3, col D) |
| VOI depth=2 | Cell 3 | D1 (row 1, col D) |
| VOI depth=1 | Cell 6 | B2 (row 2, col B) |
| Entropy Minimization | Cell 8 | D2 (row 2, col D) |
| Candidate Halving | Cell 6 | B2 (row 2, col B) |
| Baseline | Cell 12 | C3 (center) |

B1 is preferred over center for three reasons. First, B1 can itself be red (center cannot), giving a direct chance of +150 on click 1. Second, B1's geometric reach partitions the board differently from center, producing more balanced candidate sets. Third, the optimal policy is computed under the correct distribution — center looks attractive intuitively but is not optimal under hypothesis A.

Due to the grid's 4-fold rotational symmetry, B1 is equivalent to D1, B5, and D5 as first clicks. Similarly B2, D2, B4, D4 are all equivalent. The reported cell is whichever representative the memo table stored first during precomputation.

Notably, the human-designed strategy documented independently by a player uses B2 as the first click — the same cell chosen by candidate halving and VOI d=1. This independent convergence is strong validation of the halving heuristic.

---

## 6. Practical Recommendations

### For Maximum Score (Automated / Bot)

Use VOI depth=3 policy via the SeeRed live assistant. Expected score: **337/440**, finds red 100% of the time, score never drops below 200.

### For Real-Time Play Without a Lookup Table

Use entropy minimization or candidate halving — both achieve 97% of optimal and can be approximated as a mental heuristic:

1. Start at **B2** (candidate halving opening, matches human expert strategy)
2. After each reveal, apply deduction rules to eliminate impossible red positions
3. Click the cell that minimizes the expected number of remaining red candidates
4. Once only one red candidate remains, click it immediately
5. Use remaining clicks on orange neighbors (90 pts), then yellow diagonals (55 pts)

### What to Avoid

Center-first with random subsequent picks loses 75 points per game vs optimal. The center-first opening is not wrong — it provides reasonable geometric coverage — but random follow-up wastes all the information gained from each reveal.

---

## 7. SeeRed Live Assistant

SeeRed is a browser-based assistant that drives the VOI d=3 policy in real time. You mirror your in-game clicks on the visual grid, enter the revealed color, and the assistant recommends the next optimal cell.

**Setup:**
```
python server.py        # starts local policy server on port 7734
open browser_guide.html # open in any browser
```

The server loads `cache/voi_d3_cache.pkl` (~17 MB, loads in seconds) and `cache/all_boards.npy`. The browser grid shows the full 5×5 board with column (A–E) and row (1–5) labels. The recommended cell pulses white. After each reveal the grid updates: revealed cells show their sphere and points, remaining red candidates are highlighted, and the next recommendation appears.

To switch strategy, edit `POLICY_CACHE` and `POLICY_DEPTH` at the top of `server.py`.

---

## 8. Technical Notes

### POMDP Formulation

The game is a Partially Observable Markov Decision Process (POMDP). The belief state is the set of board configurations consistent with all observed colors. The value function:

```
V(belief, t) = max_x Σ_c P(x=c | belief) × [reward(c) + V(update(belief, x, c), t−1)]
```

Memoization uses split keys: value memo on `(board_indices, clicks_left)` for computational reuse across paths; policy memo on `(board_indices, revealed, clicks_left)` for correctness — ensuring the returned cell is always unclicked in the current game state.

### Why the Split Key Matters

A single key on `(board_indices, clicks_left)` causes the value function to return a cached cell that may have already been revealed via a different path, leading to double-counting of rewards (observed during development: max score 690, mean 443 — both impossible). The split key fixes this at the cost of a larger policy memo.

### Validation

- Maximum observed score of exactly 440 confirms the simulation is correct
- POMDP and VOI depth=5 producing identical results to 6 decimal places confirms both compute the same optimal solution
- All strategies respect the 200–440 score range (min 200 = 5 clicks on low-value cells), except VOI d=1 and d=2 which can score lower due to the depth-limited approximation
- Chi-square test on 46 real game observations confirms hypothesis A (p > 0.05 vs hypothesis B)

### File Structure

```
cache/
  all_boards.npy          — 16,800 valid board configurations
  voi_d1_cache.pkl        — VOI depth=1 policy (0.1 MB)
  voi_d2_cache.pkl        — VOI depth=2 policy (1.3 MB)
  voi_d3_cache.pkl        — VOI depth=3 policy (16.6 MB) ← used by server
  pomdp_cache.pkl         — exact POMDP policy (789 MB)
  entropy_cache.pkl       — entropy minimization policy (~1 MB)
  halving_cache.pkl       — candidate halving policy (~1 MB)
  results.parquet         — raw simulation results
  summary.csv             — per-strategy summary statistics

board_generator.py        — exhaustive board enumeration, hypothesis-A weights
belief_state.py           — LightBeliefState + FullBeliefState (weighted)
strategies.py             — POMDP, VOI (all depths), entropy min, candidate halving, baseline
simulation.py             — exact evaluation across all boards with weighted statistics
analysis.py               — parquet export, score distribution and heatmap plots
main.py                   — entry point with cache management
server.py                 — local HTTP policy server
browser_guide.html        — browser-based live assistant UI
```
