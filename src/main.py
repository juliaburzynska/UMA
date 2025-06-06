import numpy as np
from cec2017 import functions
from itertools import product
# from logger import Logger
import sys
import matplotlib.pyplot as plt

# sys.stdout = Logger("logZf10.txt")

OPTIMUM_VALUES = {
    f'f{i}': 100.0 * i for i in range(1, 31)
}

DIM = 10
LOWER, UPPER = -100, 100
CR = 0.9
EVAL_WINDOW = 20

ALPHA = 0.1
GAMMA = 0.9

SELECTED_FUNCTIONS = ['f10']

EXPLORATION_RUNS = 300
EXPLORATION_CONFIG = {
    "epsilon": 0.9,
    "pop_size": 200,
    "generations": 1000,  # zmienione max_generations
}

EXPLOITATION_RUNS = 20
EXPLOITATION_CONFIG = {
    "epsilon": 0.1,
    "pop_size": 100,
    "generations": 2000,
}

SUCCESS_BIN_SIZE = 0.01   # zmniejszone z 0.05
DISTANCE_BIN_SIZE = 1     # zmniejszone z 5
DISTANCE_BINS = np.arange(0, 250 + DISTANCE_BIN_SIZE, DISTANCE_BIN_SIZE)
SUCCESS_BINS = np.arange(0, 1 + SUCCESS_BIN_SIZE, SUCCESS_BIN_SIZE)

STRATEGIES = ['rand', 'best', 'rand-to-best', 'rand/2', 'rand/3', 'best/2', 'current-to-best']
F_STEP = 0.25
F_VALUES = np.arange(F_STEP, 1.0 + F_STEP, F_STEP).round(3).tolist()

ACTIONS = list(product(range(len(STRATEGIES)), range(len(F_VALUES))))
NUM_ACTIONS = len(ACTIONS)

Q_global = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1, NUM_ACTIONS))
visited_states = set()
state_visit_counts = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1), dtype=int)

def discretize_state(success_rate, avg_distance):
    s_idx = np.digitize(success_rate, SUCCESS_BINS) - 1
    d_idx = np.digitize(avg_distance, DISTANCE_BINS) - 1
    s_idx = np.clip(s_idx, 0, len(SUCCESS_BINS) - 2)
    d_idx = np.clip(d_idx, 0, len(DISTANCE_BINS) - 2)
    return s_idx, d_idx

def select_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    s_idx, d_idx = state
    return np.argmax(Q[s_idx, d_idx])

def mutate(pop, best, i, strategy, F):
    idxs = [idx for idx in range(len(pop)) if idx != i]
    if strategy in ['rand', 'rand/1']:
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
        return a + F * (b - c)
    elif strategy == 'best':
        b, c = pop[np.random.choice(idxs, 2, replace=False)]
        return best + F * (b - c)
    elif strategy == 'rand-to-best':
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
        return a + F * (best - a) + F * (b - c)
    elif strategy == 'rand/2':
        a, b, c, d, e = pop[np.random.choice(idxs, 5, replace=False)]
        return a + F * (b - c + d - e)
    elif strategy == 'rand/3':
        a, b, c, d, e, f, g = pop[np.random.choice(idxs, 7, replace=False)]
        return a + F * (b - c + d - e + f - g)
    elif strategy == 'best/2':
        b, c, d, e = pop[np.random.choice(idxs, 4, replace=False)]
        return best + F * (b - c + d - e)
    elif strategy == 'current-to-best':
        a = pop[i]
        b, c = pop[np.random.choice(idxs, 2, replace=False)]
        return a + F * (best - a) + F * (b - c)
    else:
        raise ValueError(f"Nieznana strategia: {strategy}")

def crossover(target, mutant):
    cross_points = np.random.rand(DIM) < CR
    if not np.any(cross_points):
        cross_points[np.random.randint(0, DIM)] = True
    return np.where(cross_points, mutant, target)

def bound_check(v):
    return np.clip(v, LOWER, UPPER)

def evaluate_distance(pop):
    dists = []
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            dists.append(np.linalg.norm(pop[i] - pop[j]))
    return np.mean(dists)

def differential_evolution(func, func_name, Q, config, optimum=None):
    epsilon = config['epsilon']
    pop_size = config['pop_size']
    generations = config['generations']

    pop = np.random.uniform(LOWER, UPPER, (pop_size, DIM))
    fitness = func(pop)
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    success_count = 0
    total_count = 0

    for gen in range(generations):
        next_pop = []
        next_fitness = []

        successes = 0
        total = 0

        avg_dist = evaluate_distance(pop)
        success_rate = success_count / total_count if total_count > 0 else 0
        state = discretize_state(success_rate, avg_dist)
        visited_states.add(state)
        state_visit_counts[state] += 1

        action_idx = select_action(state, Q, epsilon)
        strategy_id, f_id = ACTIONS[action_idx]
        strategy = STRATEGIES[strategy_id]
        F = F_VALUES[f_id]

        for i in range(pop_size):
            mutant = mutate(pop, best, i, strategy, F)
            mutant = bound_check(mutant)
            trial = crossover(pop[i], mutant)
            trial = bound_check(trial)
            trial_fit = func(trial.reshape(1, -1))[0]

            if trial_fit < fitness[i]:
                next_pop.append(trial)
                next_fitness.append(trial_fit)
                successes += 1
            else:
                next_pop.append(pop[i])
                next_fitness.append(fitness[i])

            total += 1

        success_count += successes
        total_count += total

        if (gen + 1) % EVAL_WINDOW == 0:
            reward = successes / total if total > 0 else 0
            new_avg_dist = evaluate_distance(next_pop)
            new_state = discretize_state(reward, new_avg_dist)
            max_future_q = np.max(Q[new_state])
            s_idx, d_idx = state
            Q[s_idx, d_idx, action_idx] += ALPHA * (reward + GAMMA * max_future_q - Q[s_idx, d_idx, action_idx])
            success_count = 0
            total_count = 0

        # Powolny spadek epsilon dla eksploracji
        epsilon = max(0.1, epsilon * 0.996)

        pop = np.array(next_pop)
        fitness = np.array(next_fitness)
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        if optimum is not None:
            best_val = np.min(next_fitness)
            if best_val <= optimum + 0.1:
                print(f"{func_name} | Gen  {gen} | Osiągnięto optimum: {best_val:.5f} <= {optimum:.5f} + 0.1")
                break

        best_val = np.min(fitness)
        worst_val = np.max(fitness)
        if gen % 100 == 0 or gen == generations - 1:
            print(f"{func_name} | Gen {gen:>4} | Best: {best_val:.5f} | Worst: {worst_val:.5f}")
            if abs(best_val - worst_val) < 0.00001 and epsilon <= 0.1:
                print(f"{func_name} | Gen {gen} | Różnica między najlepszym a najgorszym < 0.00001, przerywam.")
                break

    return np.min(fitness)

for fname in SELECTED_FUNCTIONS:
    print("\n" + "=" * 60)
    print(f"UCZENIE tablicy Q przez eksplorację | Funkcja: {fname}")
    func = getattr(functions, fname)
    optimum = OPTIMUM_VALUES.get(fname, None)

    for run in range(EXPLORATION_RUNS):
        print(f"\n--- Exploracja {run + 1}/{EXPLORATION_RUNS} ---")
        differential_evolution(func, fname, Q_global, config=EXPLORATION_CONFIG, optimum=optimum)
        coverage = 100.0 * len(visited_states) / ((len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1))
        print(f"Pokrycie przestrzeni stanów po runie {run + 1}: {coverage:.2f}%")

    print("\n" + "=" * 60)
    print(f"WYKORZYSTANIE wytrenowanej tablicy Q | Funkcja: {fname}")
    results = []

    for run in range(EXPLOITATION_RUNS):
        print(f"\n--- Eksploatacja {run + 1}/{EXPLOITATION_RUNS} ---")
        result = differential_evolution(func, fname, Q_global, config=EXPLOITATION_CONFIG, optimum=optimum)
        results.append(result)
        coverage = 100.0 * len(visited_states) / ((len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1))
        print(f"Pokrycie przestrzeni stanów po runie {run + 1}: {coverage:.2f}%")


    print(f"\nFunkcja: {fname} | Średni wynik z eksploatacji: {np.mean(results):.6f} | Min: {np.min(results):.6f} | Max: {np.max(results):.6f}")

    total_possible_states = (len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1)
    print(f"\nLiczba odwiedzonych stanów Q: {len(visited_states)} / {total_possible_states}")
    print(f"Pokrycie przestrzeni stanów: {100.0 * len(visited_states) / total_possible_states:.2f}%")

    # Zapis tablicy Q
    np.save("q_table.npy", Q_global)

    # Histogram odwiedzin stanów
    plt.figure(figsize=(10, 6))
    plt.title("Histogram odwiedzin stanów (liczba odwiedzin w każdym stanie)")
    plt.xlabel("Indeks sukcesu (bin)")
    plt.ylabel("Indeks odległości (bin)")
    plt.imshow(state_visit_counts.T, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Liczba odwiedzin')
    plt.tight_layout()
    plt.show()
