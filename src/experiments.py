import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from exploitation import exploitation
from de import de
from config import q_file, SELECTED_FUNCTIONS
import json

for func_name in SELECTED_FUNCTIONS:
    de_results = de()
    q_results = exploitation()
    q_results_with_q_table = exploitation(q_file)

    full_results = {
        "DE": de_results,
        "Q-Learning": q_results,
        "Q-learning-with-q-table": q_results_with_q_table
    }

    with open(f"results/experiment_results_{func_name}", 'w') as f:
        json.dump(full_results, f, indent=4)

    methods = list(full_results.keys())
    mean_fitness = [full_results[m]["summary"]["mean_best_fitness"] for m in methods]
    std_fitness = [full_results[m]["summary"]["std_best_fitness"] for m in methods]
    success_rates = [full_results[m]["summary"]["mean_success_rate"] for m in methods]

    x = np.arange(len(methods))

    plt.figure(figsize=(8, 5))
    plt.bar(x, mean_fitness, yerr=std_fitness, capsize=5, color=sns.color_palette("viridis", len(methods)))
    plt.xticks(x, methods)
    plt.ylabel("Średnia wartość fitness")
    plt.title(f"{func_name} – Średni najlepszy wynik")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/fitness_comparison_{func_name}.png")
    plt.close()

    # Wykres 2: Procent sukcesów mutacji
    plt.figure(figsize=(8, 5))
    plt.bar(x, success_rates, capsize=5, color=sns.color_palette("muted", len(methods)))
    plt.xticks(x, methods)
    plt.ylabel("Procent sukcesów mutacji")
    plt.title(f"{func_name} – Skuteczność mutacji")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/success_rate_{func_name}.png")
    plt.close()