from qlearning_de import differential_evolution_qlearning
from config import *

for fname in SELECTED_FUNCTIONS:
    sys.stdout = Logger(f"logs/exploration___{fname}.txt")
    print("\n" + "=" * 60)
    print(f"TRAINING Q-table through exploration | Function: {fname}")
    func = getattr(functions, fname)
    optimum = OPTIMUM_VALUES.get(fname, None)

    # Reset Q-table and counters before exploration
    Q_global.fill(0)
    visited_states.clear()
    state_visit_counts.fill(0)

    for run in range(EXPLORATION_RUNS):
        print(f"\n--- Exploration {run + 1}/{EXPLORATION_RUNS} ---")
        differential_evolution_qlearning(func, fname, Q_global, config=EXPLORATION_CONFIG, optimum=optimum, caller_name="exploration.py")
        coverage = 100.0 * len(visited_states) / ((len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1))
        print(f"State space coverage after run {run + 1}: {coverage:.2f}%")

    print("\n" + "=" * 60)

    # Save Q-table
    np.save(f"q_tables/q_table_{fname}.npy", Q_global)