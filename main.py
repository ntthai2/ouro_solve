"""
main.py

Entry point. Run with:
    python main.py

Evaluates VOI at depths 1, 2, 3, 4 alongside POMDP, entropy, halving, baseline.
All caches stored in cache/ folder.
"""

import os
import time
import pickle
import numpy as np

from board_generator import enumerate_boards, compute_board_weights
from belief_state import FullBeliefState
from strategies import ExactPOMDP, VOIGreedy, EntropyMinimization, CandidateHalving
from simulation import run_simulation, compute_summary, compute_by_center_color
from analysis import run_analysis

CACHE_DIR     = 'cache'
BOARDS_CACHE  = f'{CACHE_DIR}/all_boards.npy'
POMDP_CACHE   = f'{CACHE_DIR}/pomdp_cache.pkl'
ENTROPY_CACHE = f'{CACHE_DIR}/entropy_cache.pkl'
HALVING_CACHE = f'{CACHE_DIR}/halving_cache.pkl'

VOI_DEPTHS    = [1, 2, 3]

# bump when strategy logic changes
CACHE_VERSION = 5


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_ok(path):
    ver = path + '.ver'
    if not os.path.exists(path) or not os.path.exists(ver):
        return False
    with open(ver) as f:
        return f.read().strip() == str(CACHE_VERSION)

def _mark_cache(path):
    with open(path + '.ver', 'w') as f:
        f.write(str(CACHE_VERSION))

def _save(strategy, path):
    data = {'policy_memo': strategy._policy_memo}
    if hasattr(strategy, '_value_memo'):
        data['value_memo'] = strategy._value_memo
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    _mark_cache(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved -> {path}  ({size_mb:.1f} MB)")

def _load_pomdp(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    s = ExactPOMDP()
    s._value_memo  = d['value_memo']
    s._policy_memo = d['policy_memo']
    return s

def _load_voi(path, depth):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    s = VOIGreedy(depth=depth)
    s._value_memo  = d['value_memo']
    s._policy_memo = d['policy_memo']
    return s

def _load_entropy(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    s = EntropyMinimization()
    s._policy_memo = d['policy_memo']
    return s

def _load_halving(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    s = CandidateHalving()
    s._policy_memo = d['policy_memo']
    return s


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # step 1 — boards
    if os.path.exists(BOARDS_CACHE):
        print(f"Loading boards from {BOARDS_CACHE}...")
        all_boards = np.load(BOARDS_CACHE)
    else:
        print("Enumerating all valid boards...")
        t0 = time.time()
        all_boards = enumerate_boards()
        np.save(BOARDS_CACHE, all_boards)
        print(f"Done in {time.time()-t0:.1f}s")

    print(f"Total valid boards: {len(all_boards):,}")
    weights = compute_board_weights(all_boards)
    FullBeliefState.load_boards(all_boards, weights=weights)
    initial_belief = FullBeliefState()

    # step 2 — POMDP
    # if _cache_ok(POMDP_CACHE):
    #     print(f"\nLoading POMDP from cache...")
    #     pomdp = _load_pomdp(POMDP_CACHE)
    #     print(f"  Memo table: {len(pomdp._value_memo):,} states")
    # else:
    #     print("\nPrecomputing POMDP (~33 min)...")
    #     pomdp = ExactPOMDP()
    #     t0 = time.time()
    #     pomdp.precompute(initial_belief, max_clicks=5)
    #     print(f"  Done in {time.time()-t0:.1f}s")
    #     _save(pomdp, POMDP_CACHE)
    # print(f"  Expected score: {pomdp.value(initial_belief, 5):.4f}")
    # print(f"  First click: cell {pomdp(initial_belief, 5)}")

    # step 3 — VOI depths 1–4
    voi_strategies = []
    for d in VOI_DEPTHS:
        cache_path = f'{CACHE_DIR}/voi_d{d}_cache.pkl'
        if _cache_ok(cache_path):
            print(f"\nLoading VOI depth={d} from cache...")
            voi = _load_voi(cache_path, depth=d)
            print(f"  Memo table: {len(voi._value_memo):,} states")
        else:
            print(f"\nPrecomputing VOI depth={d}...")
            voi = VOIGreedy(depth=d)
            t0 = time.time()
            voi.precompute(initial_belief, max_clicks=5)
            print(f"  Done in {time.time()-t0:.1f}s")
            _save(voi, cache_path)
        print(f"  First click: cell {voi(initial_belief, 5)}")
        voi_strategies.append(voi)

    # step 4 — entropy minimization
    if _cache_ok(ENTROPY_CACHE):
        print(f"\nLoading entropy minimization from cache...")
        entropy = _load_entropy(ENTROPY_CACHE)
        print(f"  Memo table: {len(entropy._policy_memo):,} states")
    else:
        print("\nPrecomputing entropy minimization...")
        entropy = EntropyMinimization()
        t0 = time.time()
        entropy.precompute(initial_belief, max_clicks=5)
        print(f"  Done in {time.time()-t0:.1f}s")
        _save(entropy, ENTROPY_CACHE)
    print(f"  First click: cell {entropy(initial_belief, 5)}")

    # step 5 — candidate halving
    if _cache_ok(HALVING_CACHE):
        print(f"\nLoading candidate halving from cache...")
        halving = _load_halving(HALVING_CACHE)
        print(f"  Memo table: {len(halving._policy_memo):,} states")
    else:
        print("\nPrecomputing candidate halving...")
        halving = CandidateHalving()
        t0 = time.time()
        halving.precompute(initial_belief, max_clicks=5)
        print(f"  Done in {time.time()-t0:.1f}s")
        _save(halving, ENTROPY_CACHE)
    print(f"  First click: cell {halving(initial_belief, 5)}")

    # step 6 — simulate
    print("\nRunning simulation across all boards...")
    df = run_simulation(
        all_boards,
        # pomdp_strategy=pomdp,
        voi_strategies=voi_strategies,
        entropy_strategy=entropy,
        halving_strategy=halving,
        verbose=True,
    )

    # step 7 — export
    summary   = compute_summary(df)
    by_center = compute_by_center_color(df)
    run_analysis(df, by_center, summary)

    # print VOI depth comparison
    print("\n=== VOI Depth Comparison ===")
    voi_rows = summary[summary['strategy'].str.startswith('voi')]
    print(voi_rows[['strategy', 'expected_score', 'p_find_red']].to_string(index=False))

    print("\n=== Full Summary ===")
    print(summary[['strategy', 'expected_score', 'p_find_red', 'avg_clicks']].to_string(index=False))


if __name__ == '__main__':
    main()