import sys
import os
import numpy as np

# Add parent directory to path so we can import ot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ot.board_generator import generate_n_random_boards, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK, COLOR_VALUES
from ot.strategies import RandomStrategy, OTHybridStrategy, OTInfoGainStrategy
from ot.simulation import run_simulation_ot, sample_value

def main():
    n_boards = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Generating {n_boards} random OT boards for benchmark...")
    boards = generate_n_random_boards(n_boards)
    
    print(f"Successfully generated {len(boards)} boards.")
    
    # Calculate Oracle EV
    # Oracle knows exactly the board. It clicks all non-blue cells and up to 4 blue cells.
    # Since boards always have >= 9 blue cells, it always clicks exactly 4 blue cells.
    oracle_evs = []
    for b in boards:
        # Let's compute it properly:
        ev = 0
        for c in range(25):
            color = int(b[c])
            if color != COLOR_BLUE:
                ev += sample_value(color)
        ev += 40 # 4 Blue clicks
        oracle_evs.append(ev)
        
    avg_oracle_ev = np.mean(oracle_evs)
    
    strategies = [
        OTHybridStrategy(use_exact_endgame=True, n_samples=1000),
        OTInfoGainStrategy(lam=0.88, use_exact_endgame=True, n_samples=3500, k_prune=1),
        OTInfoGainStrategy(lam=0.7, use_exact_endgame=True, n_samples=1000),
        OTInfoGainStrategy(lam=0.5, use_exact_endgame=True, n_samples=1000)
    ]
    
    df = run_simulation_ot(boards, strategies, verbose=True)
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Oracle EV (Theoretical Max): {avg_oracle_ev:.2f}\n")
    
    # Prepare table format
    print(f"{'Strategy':<35} | {'EV':<8} | {'Std':<7} | {'95% CI':<17} | {'Range':<11} | {'% Oracle':<8} | {'Win Rate':<9} | {'% Non-Blue':<10}")
    print("-" * 125)
    
    for strat in strategies:
        strat_name = strat.name
        strat_df = df[df['strategy'] == strat_name]
        
        scores = strat_df['score'].values
        n_strat = len(scores)
        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if n_strat > 1 else 0.0
        se_score = std_score / np.sqrt(n_strat) if n_strat > 0 else 0.0
        ci_l = avg_score - 1.96 * se_score
        ci_u = avg_score + 1.96 * se_score
        min_s = float(np.min(scores)) if n_strat > 0 else 0.0
        max_s = float(np.max(scores)) if n_strat > 0 else 0.0

        win_rate = strat_df['win'].mean() * 100
        
        loss_df = strat_df[strat_df['win'] == False]
        avg_non_blue_percent_when_loss = (loss_df['cleared_non_blue'] / loss_df['total_non_blue']).mean() * 100 if len(loss_df) > 0 else 100.0
        
        # Calculate loss stages
        loss_early = len(loss_df[loss_df['unrevealed_when_lost'] > 15])
        loss_mid = len(loss_df[(loss_df['unrevealed_when_lost'] <= 15) & (loss_df['unrevealed_when_lost'] > 8)])
        loss_late = len(loss_df[loss_df['unrevealed_when_lost'] <= 8])
        total_losses = len(loss_df) if len(loss_df) > 0 else 1
        
        win_df = strat_df[strat_df['win'] == True]
        avg_blue_clicks_when_win = win_df['blue_clicks'].mean() if len(win_df) > 0 else 0.0
        
        pct_oracle = (avg_score / avg_oracle_ev) * 100
        ci_str = f"[{ci_l:.1f}, {ci_u:.1f}]"
        rng_str = f"[{min_s:.0f}, {max_s:.0f}]"
        
        print(f"{strat_name:<35} | {avg_score:<8.2f} | {std_score:<7.2f} | {ci_str:<17} | {rng_str:<11} | {pct_oracle:>6.2f}% | {win_rate:>6.2f}%   | {avg_non_blue_percent_when_loss:>9.2f}%")
        if total_losses > 1:
            print(f"    Loss distribution -> Early (>15 left): {loss_early/total_losses*100:.1f}%, Mid (9-15 left): {loss_mid/total_losses*100:.1f}%, Late (<=8 left): {loss_late/total_losses*100:.1f}%")

if __name__ == '__main__':
    main()
