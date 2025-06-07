from config import *
import numpy as np
from de_functions import differential_evolution

for func_name in SELECTED_FUNCTIONS:
    sys.stdout = Logger(f"logs/de___{func_name}.txt")
    func = getattr(functions, func_name)
    print(f"\n--- Running DE on {func_name} ---")

    results = []
    success_rates = []

    for run in range(1, DE_RUNS + 1):
        best_value, success_rate = differential_evolution(func, func_name, run)
        results.append(best_value)
        success_rates.append(success_rate)
        print(f"{func_name} | Run {run} finished. Best value: {best_value:.6f}, Success Rate: {success_rate:.4f}")

    mean_result = np.mean(results)
    std_result = np.std(results)
    mean_success_rate = np.mean(success_rates)

    print(f"\n{func_name} | Summary over {DE_RUNS} runs:")
    print(f"Mean best value: {mean_result:.6f}")
    print(f"Std deviation:   {std_result:.6f}")
    print(f"Mean mutation success rate: {mean_success_rate:.4f}")