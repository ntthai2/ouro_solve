import sys
import os
import numpy as np

# Add parent directory to path so we can import ot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ot.board_generator import generate_n_random_boards, COLOR_BLUE, COLOR_WHITE, COLOR_BLACK, COLOR_VALUES
from ot.strategies import RandomStrategy, OTHybridStrategy, OTInfoGainStrategy
from ot.simulation import run_simulation_ot, sample_value

def main():
    print("Generating 300 random OT boards for benchmark (Ablation study)...")
    boards = generate_n_random_boards(300)
    
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
        OTInfoGainStrategy(lam=0.9, use_exact_endgame=True, n_samples=1000),
        OTInfoGainStrategy(lam=0.7, use_exact_endgame=True, n_samples=1000),
        OTInfoGainStrategy(lam=0.5, use_exact_endgame=True, n_samples=1000)
    ]
    
    df = run_simulation_ot(boards, strategies, verbose=True)
    
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Oracle EV (Theoretical Max): {avg_oracle_ev:.2f}\n")
    
    # Prepare table format
    print(f"{'Strategy':<35} | {'EV':<8} | {'% Oracle':<8} | {'Win Rate':<9} | {'% Non-Blue (Loss)':<18} | {'Blue Clicks (Win)':<18}")
    print("-" * 110)
    
    for strat in strategies:
        strat_name = strat.name
        strat_df = df[df['strategy'] == strat_name]
        
        avg_score = strat_df['score'].mean()
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
        
        print(f"{strat_name:<35} | {avg_score:<8.2f} | {pct_oracle:>6.2f}% | {win_rate:>6.2f}%   | {avg_non_blue_percent_when_loss:>16.2f}% | {avg_blue_clicks_when_win:>16.2f}")
        if total_losses > 1:
            print(f"    Loss distribution -> Early (>15 left): {loss_early/total_losses*100:.1f}%, Mid (9-15 left): {loss_mid/total_losses*100:.1f}%, Late (<=8 left): {loss_late/total_losses*100:.1f}%")

if __name__ == '__main__':
    main()
