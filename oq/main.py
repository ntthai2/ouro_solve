"""
main.py

Entry point for OQ mode.
Run with:
    python -m oq.main

VOI-only pipeline with cache load/save and full-board simulation.
"""

import os
import time
import pickle
import numpy as np

from oq.board_generator import enumerate_boards
from oq.belief_state import OQFullBeliefState
from oq.strategies import OQVOIGreedy, OQPurpleFirstGreedy
from oq.simulation import run_simulation_oq

CACHE_DIR = "cache"
BOARDS_CACHE = "cache/all_boards_oq.npy"
VOI_DEPTHS = [2]
CACHE_VERSION = 1


# -- cache helpers -------------------------------------------------------------

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


def _load_voi(path, depth):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    s = OQVOIGreedy(depth=depth)
    s._value_memo = d['value_memo']
    s._policy_memo = d['policy_memo']
    return s


# -- main ----------------------------------------------------------------------

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # step 1 -- boards
    if os.path.exists(BOARDS_CACHE):
        print(f"Loading boards from {BOARDS_CACHE}...")
        all_boards = np.load(BOARDS_CACHE)
    else:
        print("Enumerating all valid OQ boards...")
        t0 = time.time()
        all_boards = enumerate_boards()
        np.save(BOARDS_CACHE, all_boards)
        print(f"Done in {time.time() - t0:.1f}s")

    print(f"Total valid OQ boards: {len(all_boards):,}")
    OQFullBeliefState.load_boards(all_boards)
    initial_belief = OQFullBeliefState()

    # step 2 -- VOI depths
    voi_strategies = []
    for d in VOI_DEPTHS:
        cache_path = f"{CACHE_DIR}/voi_oq_d{d}_cache.pkl"
        if _cache_ok(cache_path):
            print(f"\nLoading OQ VOI depth={d} from cache...")
            voi = _load_voi(cache_path, depth=d)
            print(f"  Memo table: {len(voi._value_memo):,} states")
        else:
            print(f"\nPrecomputing OQ VOI depth={d}...")
            voi = OQVOIGreedy(depth=d)
            t0 = time.time()
            voi.precompute(initial_belief, max_clicks=7)
            print(f"  Done in {time.time() - t0:.1f}s")
            _save(voi, cache_path)

        first_cell = voi(initial_belief, 7, first_click=True)
        print(f"  First click: cell {first_cell}")
        voi_strategies.append(voi)

    # step 2b -- purple-first greedy (no precompute needed)
    # print("\nUsing purple-first greedy strategy (no precompute)...")
    # purple_first = OQPurpleFirstGreedy()
    # first_cell = purple_first(initial_belief, 7, first_click=True)
    # print(f"  First click: cell {first_cell}")
    # voi_strategies.append(purple_first)

    # step 3 -- simulation
    print("\nRunning OQ simulation across all boards...")
    df = run_simulation_oq(all_boards, voi_strategies, verbose=True)

    # step 4 -- summary
    summary = (df.groupby('strategy', as_index=False)
                 .agg(mean_score=('score', 'mean'),
                      std_score=('score', 'std'),
                      min_score=('score', 'min'),
                      max_score=('score', 'max'),
                      p_find_red=('found_red', 'mean'),
                      mean_paid_clicks=('paid_clicks_used', 'mean'),
                      mean_purples=('purples_clicked', 'mean')))
    summary = summary.sort_values('mean_score', ascending=False)

    print("\n=== OQ Summary ===")
    print(summary.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
