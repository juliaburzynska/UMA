from config import *
from de_functions import differential_evolution


def de():
    all_results = {}
    for func_name in SELECTED_FUNCTIONS:
        sys.stdout = Logger(f"logs/de___{func_name}.txt")
        func = getattr(functions, func_name)
        print(f"\n--- Running DE on {func_name} ---")

        best_fitness_values = []
        success_rates = []

        for run in range(1, DE_RUNS + 1):
            best_value, success_rate = differential_evolution(func, func_name, run)
            best_fitness_values.append(best_value)
            success_rates.append(success_rate)
            print(f"{func_name} | Run {run} finished. Best value: {best_value:.6f}, Success Rate: {success_rate:.4f}")

        mean_best_fitness = np.mean(best_fitness_values)
        std_best_fitness = np.std(best_fitness_values)
        mean_success_rate = np.mean(success_rates)

        print(f"\n{func_name} | Summary over {DE_RUNS} runs:")
        print(f"Mean best fitness: {mean_best_fitness:.6f}")
        print(f"Standard deviation of best fitness:   {std_best_fitness:.6f}")
        print(f"Mean mutation success rate: {mean_success_rate:.4f}")

        all_results[func_name] = {
            "runs": {
                "best_fitness": best_fitness_values,
                "success_rate": success_rates
            },
            "summary": {
                "mean_best_fitness": mean_best_fitness,
                "std_best_fitness": std_best_fitness,
                "mean_success_rate": mean_success_rate
            }
        }

    return all_results
