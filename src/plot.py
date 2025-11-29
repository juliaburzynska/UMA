import json
import matplotlib.pyplot as plt

with open('results/experiment_results_f25', 'r') as f:
    data = json.load(f)

models = ['DE', 'Q-Learning', 'Q-learning-with-q-table']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 6))

for model, color in zip(models, colors):
    fitness_values = data[model]['runs']['best_fitness']
    plt.plot(range(len(fitness_values)), fitness_values, label=model, color=color)

plt.xlabel('Iteracja')
plt.ylabel('Best Fitness')
plt.title('Porównanie Best Fitness dla funkcji f25')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/best_fitness_f25.png")
plt.show()
