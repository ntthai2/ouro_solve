"""
analysis.py

Generates all output:
  - cache/results.parquet        raw per-(strategy, board) data
  - cache/summary.csv            per-strategy expected score, P(find red), etc.
  - plots/score_dist.png   score distribution per strategy
  - plots/by_center.png    breakdown by center cell color
  - plots/p_find_red.png   P(find red) comparison
  - plots/red_position.png expected score by red position heatmap
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

PLOT_DIR = 'plots'
STRATEGY_ORDER = ['exact_pomdp', 'voi_greedy', 'entropy_minimization',
                  'candidate_halving', 'center_first_random']
STRATEGY_LABELS = {
    'exact_pomdp':          'Exact POMDP',
    'voi_greedy':           'VOI Greedy (d=2)',
    'entropy_minimization': 'Entropy Min.',
    'candidate_halving':    'Candidate Halving',
    'center_first_random':  'Baseline (random)',
}
COLORS_PLOT = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
CENTER_COLOR_ORDER = ['orange', 'yellow', 'green', 'teal', 'blue']
CENTER_COLOR_HEX   = {
    'orange': '#F09500', 'yellow': '#C8B400',
    'green':  '#639922', 'teal':   '#1D9E75', 'blue': '#378ADD',
}


def save_parquet(df: pd.DataFrame, path: str = 'cache/results.parquet'):
    df = df.copy()
    df['found_red'] = df['found_red'].astype(float)
    df.to_parquet(path, index=False)
    print(f"Saved raw data → {path}  ({len(df):,} rows)")


def save_summary(summary: pd.DataFrame, path: str = 'cache/summary.csv'):
    summary.to_csv(path, index=False)
    print(f"Saved summary  → {path}")
    print("\n" + summary.to_string(index=False))


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


# ── plot 1: score distribution ────────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame):
    _ensure_plot_dir()
    strategies = [s for s in STRATEGY_ORDER if s in df['strategy'].unique()]
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(4 * len(strategies), 4), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat, color in zip(axes, strategies, COLORS_PLOT):
        data = df[df['strategy'] == strat]['score']
        ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_title(STRATEGY_LABELS.get(strat, strat), fontsize=11)
        ax.set_xlabel('Score')
        ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1,
                   label=f'Mean: {data.mean():.0f}')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Board count')
    fig.suptitle('Score distribution per strategy', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'score_dist.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── plot 2: breakdown by center color ────────────────────────────────────────

def plot_by_center_color(by_center: pd.DataFrame):
    _ensure_plot_dir()
    strategies = [s for s in STRATEGY_ORDER if s in by_center['strategy'].unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(CENTER_COLOR_ORDER))
    width = 0.15

    for i, (strat, color) in enumerate(zip(strategies, COLORS_PLOT)):
        sub = by_center[by_center['strategy'] == strat]
        sub = sub.set_index('center_color').reindex(CENTER_COLOR_ORDER)

        scores = sub['expected_score'].fillna(0).values
        probs  = sub['p_find_red'].fillna(0).values

        offset = (i - len(strategies)/2 + 0.5) * width
        ax1.bar(x + offset, scores, width, label=STRATEGY_LABELS.get(strat, strat),
                color=color, alpha=0.85)
        ax2.bar(x + offset, probs * 100, width, color=color, alpha=0.85)

    for ax, ylabel, title in [
        (ax1, 'Expected score', 'Expected score by center color'),
        (ax2, 'P(find red) %', 'P(find red) by center color'),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CENTER_COLOR_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        for cc, pos in zip(CENTER_COLOR_ORDER, x):
            ax.get_xticklabels()[list(CENTER_COLOR_ORDER).index(cc)].set_color(
                CENTER_COLOR_HEX.get(cc, 'black'))

    ax1.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'by_center.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── plot 3: P(find red) comparison bar ───────────────────────────────────────

def plot_p_find_red(summary: pd.DataFrame):
    _ensure_plot_dir()
    strategies = [s for s in STRATEGY_ORDER if s in summary['strategy'].values]
    labels  = [STRATEGY_LABELS.get(s, s) for s in strategies]
    probs   = [summary[summary['strategy'] == s]['p_find_red'].values[0] * 100
               for s in strategies]
    scores  = [summary[summary['strategy'] == s]['expected_score'].values[0]
                for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars1 = ax1.barh(labels, probs, color=COLORS_PLOT[:len(strategies)], alpha=0.85)
    ax1.set_xlabel('P(find red) %')
    ax1.set_title('P(find red) per strategy')
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(bars1, probs):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9)

    bars2 = ax2.barh(labels, scores, color=COLORS_PLOT[:len(strategies)], alpha=0.85)
    ax2.set_xlabel('Expected score')
    ax2.set_title('Expected score per strategy')
    for bar, val in zip(bars2, scores):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'p_find_red.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── plot 4: expected score by red position heatmap ────────────────────────────

def plot_red_position_heatmap(df: pd.DataFrame):
    _ensure_plot_dir()
    strategies = [s for s in STRATEGY_ORDER if s in df['strategy'].unique()]
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(4 * len(strategies), 4))
    if len(strategies) == 1:
        axes = [axes]

    all_scores = df['score'].values
    vmin, vmax = all_scores.min(), all_scores.max()

    for ax, strat in zip(axes, strategies):
        sub   = df[df['strategy'] == strat]
        grid  = np.full((5, 5), np.nan)
        for pos in range(25):
            pos_data = sub[sub['red_position'] == pos]['score']
            if len(pos_data) > 0:
                grid[pos // 5, pos % 5] = pos_data.mean()

        im = ax.imshow(grid, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(STRATEGY_LABELS.get(strat, strat), fontsize=10)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['A','B','C','D','E'], fontsize=8)
        ax.set_yticklabels(['1','2','3','4','5'], fontsize=8)

        for r in range(5):
            for c in range(5):
                val = grid[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f'{val:.0f}', ha='center', va='center',
                            fontsize=7, color='black')
        # mark center
        ax.add_patch(plt.Rectangle((-0.5 + 2, -0.5 + 2), 1, 1,
                                    fill=False, edgecolor='blue', linewidth=2))

    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='RdYlGn'),
                 ax=axes, shrink=0.8, label='Expected score')
    fig.suptitle('Expected score by red position (blue box = center, excluded)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'red_position.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, by_center: pd.DataFrame, summary: pd.DataFrame):
    save_parquet(df)
    save_summary(summary)
    plot_score_distribution(df)
    plot_by_center_color(by_center)
    plot_p_find_red(summary)
    plot_red_position_heatmap(df)
    print("\nAll outputs saved.")