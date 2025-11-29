from qlearning_de import differential_evolution_qlearning
from config import *
import numpy as np

Q_global = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1, NUM_ACTIONS))

print("\n" + "=" * 60)
print("Starting exploration phase on multiple functions")
print(f"Number of states: {len(SUCCESS_BINS) - 1} x {len(DISTANCE_BINS) - 1} = {(len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1)}")
print(f"Number of actions: {NUM_ACTIONS}")
print("=" * 60)

for func_name in EXPLORATION_FUNCTIONS:
    sys.stdout = Logger(f"logs/exploration___{func_name}.txt")
    print(f"\n[FUNCTION] Training on function: {func_name}")
    func = getattr(functions, func_name)

    for run in range(EXPLORATION_RUNS):
        print(f"\n--- Run {run+1}/{EXPLORATION_RUNS} | Function: {func_name} ---")

        best_fitness, success_rate = differential_evolution_qlearning(
            func,
            func_name,
            Q_global,
            config=EXPLORATION_CONFIG,
            optimum=None,
            caller_name="exploration.py"
        )

        print(f"{func_name} | Run finished | Best fitness: {best_fitness:.6f}, Success rate: {success_rate:.4f}")

    # Zapisz Q-table po każdej funkcji
    np.save(q_file, Q_global)
    print(f"[SAVE] Q-table saved after training on {func_name}")

print("\n[INFO] Exploration completed on all selected functions.")
print("[INFO] Final Q-table saved at: q_tables/q_table_multi_function.npy")