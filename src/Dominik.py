import numpy as np
from cec2017 import functions
from itertools import product

OPTIMUM_VALUES = {
    'f1': 100.0, 'f2': 200.0, 'f3': 300.0, 'f4': 400.0, 'f5': 500.0,
    'f6': 600.0, 'f7': 700.0, 'f8': 800.0, 'f9': 900.0, 'f10': 1000.0,
    'f11': 1100.0, 'f12': 1200.0, 'f13': 1300.0, 'f14': 1400.0, 'f15': 1500.0,
    'f16': 1600.0, 'f17': 1700.0, 'f18': 1800.0, 'f19': 1900.0, 'f20': 2000.0,
    'f21': 2100.0, 'f22': 2200.0, 'f23': 2300.0, 'f24': 2400.0, 'f25': 2500.0,
    'f26': 2600.0, 'f27': 2700.0, 'f28': 2800.0, 'f29': 2900.0, 'f30': 3000.0
}

DIM = 10
POP_SIZE = 500
LOWER, UPPER = -100, 100
CR = 0.9
GENERATIONS = 1500
EVAL_WINDOW = 20

SELECTED_FUNCTIONS = ['f1', 'f3', 'f6', 'f7', 'f10', 'f13', 'f16', 'f20', 'f23', 'f24', 'f27']

ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.11

SUCCESS_BIN_SIZE = 0.1
DISTANCE_BIN_SIZE = 20
DISTANCE_BINS = np.arange(0, 200 + DISTANCE_BIN_SIZE, 11)

STRATEGIES = ['rand', 'best', 'rand-to-best']
F_STEP = 0.25
F_VALUES = np.arange(F_STEP, 1.0 + F_STEP, F_STEP).round(3).tolist()

SUCCESS_BINS = np.arange(0, 1 + SUCCESS_BIN_SIZE, SUCCESS_BIN_SIZE)
ACTIONS = list(product(range(len(STRATEGIES)), range(len(F_VALUES))))
NUM_ACTIONS = len(ACTIONS)

Q = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1, NUM_ACTIONS))


def discretize_state(success_rate, avg_distance):
    s_idx = np.digitize(success_rate, SUCCESS_BINS) - 1
    d_idx = np.digitize(avg_distance, DISTANCE_BINS) - 1
    s_idx = np.clip(s_idx, 0, len(SUCCESS_BINS) - 2)
    d_idx = np.clip(d_idx, 0, len(DISTANCE_BINS) - 2)
    return s_idx, d_idx


def select_action(state):
    if np.random.rand() < EPSILON:
        return np.random.randint(NUM_ACTIONS)
    s_idx, d_idx = state
    return np.argmax(Q[s_idx, d_idx])


def mutate(pop, best, i, strategy, F):
    idxs = [idx for idx in range(POP_SIZE) if idx != i]
    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
    if strategy == 'rand':
        return a + F * (b - c)
    elif strategy == 'best':
        return best + F * (b - c)
    elif strategy == 'rand-to-best':
        return a + F * (best - a) + F * (b - c)


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


def differential_evolution(func, func_name, optimum=None):
    global Q

    pop = np.random.uniform(LOWER, UPPER, (POP_SIZE, DIM))
    fitness = func(pop)
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    success_count = 0
    total_count = 0

    for gen in range(GENERATIONS):
        next_pop = []
        next_fitness = []

        successes = 0
        total = 0

        avg_dist = evaluate_distance(pop)
        success_rate = success_count / total_count if total_count > 0 else 0
        state = discretize_state(success_rate, avg_dist)

        action_idx = select_action(state)
        strategy_id, f_id = ACTIONS[action_idx]
        strategy = STRATEGIES[strategy_id]
        F = F_VALUES[f_id]

        for i in range(POP_SIZE):
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

            if optimum is not None:
                best_val = np.min(next_fitness)
                if best_val <= optimum + 0.0001:
                    print(f"{func_name} | Gen  {gen} | Osiągnięto optimum: {best_val:.5f} <= {optimum:.5f} + 0.0001")
                    break

        pop = np.array(next_pop)
        fitness = np.array(next_fitness)
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        best_val = np.min(fitness)
        worst_val = np.max(fitness)
        if gen % 100 == 0 or gen == GENERATIONS - 1:
            print(f"{func_name} | Gen {gen:>4} | Best: {best_val:.5f} | Worst: {worst_val:.5f}")

    return np.min(fitness)


for fname in SELECTED_FUNCTIONS:
    print("\n" + "=" * 60)
    print(f"Optymalizacja funkcji: {fname}")
    func = getattr(functions, fname)
    optimum = OPTIMUM_VALUES.get(fname, None)
    best_result = differential_evolution(func, fname, optimum)
    print(f"Najlepszy wynik: {best_result:.6f}")
    print("=" * 60)
