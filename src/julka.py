import numpy as np
from cec2017 import functions
import random

# Parametry DE
DIMENSION = 10
LOWER_BOUND = -100
UPPER_BOUND = 100
POP_SIZE = 100
CR = 0.9
GENERATIONS = 10000
WINDOW_SIZE = 20  # liczba generacji, po której oceniamy sukcesy

# Wybór funkcji do testów
SELECTED_FUNCTIONS = ['f1', 'f3', 'f6', 'f7', 'f10', 'f13', 'f16', 'f20', 'f23', f'24', 'f27']

# Strategie mutacji
MUTATION_STRATEGIES = [
    "rand/1",
    "rand/2",
    "best/1",
    "current-to-best/1"
]

# Możliwe wartości F
F_VALUES = [0.3, 0.5, 0.7]

# Dyskretyzacja stanu
SUCCESS_BINS = np.linspace(0, 1, 6)  # 0–1 podzielone na 5 części
DIVERSITY_BINS = np.linspace(0, 100, 6)  # Zakładana maksymalna różnorodność

# Q-learning hiperparametry
ALPHA = 0.1  # współczynnik uczenia
GAMMA = 0.9  # współczynnik dyskontowy
EPSILON = 0.1  # prawdopodobieństwo eksploracji

# Inicjalizacja tabeli Q
num_success_states = len(SUCCESS_BINS) - 1
num_diversity_states = len(DIVERSITY_BINS) - 1
num_actions = len(MUTATION_STRATEGIES) * len(F_VALUES)

Q_TABLE_SHAPE = (num_success_states, num_diversity_states, num_actions)
q_table = np.zeros(Q_TABLE_SHAPE)


# Funkcje pomocnicze

def get_state(success_rate, diversity):
    """Zwraca zdyskretyzowany stan."""
    success_idx = np.digitize(success_rate, SUCCESS_BINS) - 1
    diversity_idx = np.digitize(diversity, DIVERSITY_BINS) - 1
    success_idx = np.clip(success_idx, 0, num_success_states - 1)
    diversity_idx = np.clip(diversity_idx, 0, num_diversity_states - 1)
    return success_idx, diversity_idx


def choose_action(state):
    """Wybór akcji na podstawie Q-table (z eksploracją)."""
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, num_actions - 1)
    else:
        return int(np.argmax(q_table[state]))


def decode_action(action_index):
    """Dekoduje indeks akcji na strategię i F."""
    strategy_idx = action_index // len(F_VALUES)
    f_idx = action_index % len(F_VALUES)
    return MUTATION_STRATEGIES[strategy_idx], F_VALUES[f_idx]


def mutate(population, best_individual, current_individual, F, strategy):
    """Różne strategie mutacji."""
    idxs = [idx for idx in range(len(population))]
    a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]

    if strategy == "rand/1":
        mutant = a + F * (b - c)
    elif strategy == "rand/2":
        mutant = a + F * (b - c) + F * (d - e)
    elif strategy == "best/1":
        mutant = best_individual + F * (a - b)
    elif strategy == "current-to-best/1":
        mutant = current_individual + F * (best_individual - current_individual) + F * (a - b)
    else:
        raise ValueError("Unknown mutation strategy")

    return np.clip(mutant, LOWER_BOUND, UPPER_BOUND)


def diversity(population):
    """Oblicza średnią odległość między osobnikami."""
    dists = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dists.append(np.linalg.norm(population[i] - population[j]))
    return np.mean(dists) if dists else 0


def crossover(population, mutated, CR):
    children = []
    for target, mutant in zip(population, mutated):
        cross_points = np.random.rand(DIMENSION) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, DIMENSION)] = True
        trial = np.where(cross_points, mutant, target)
        children.append(trial)
    return np.array(children)


def bounds_check(children, best_individual):
    bound_ind = np.all((children < UPPER_BOUND) & (children > LOWER_BOUND), axis=1)
    children = children[bound_ind]
    while len(children) < POP_SIZE:
        children = np.concatenate((children, [best_individual]), axis=0)
    return children


def differential_evolution_qlearning(func, func_name):
    global q_table

    # Inicjalizacja populacji
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))
    fitness = func(population)
    best_fitness_history = []

    # Historia sukcesów
    recent_successes = []

    for generation in range(GENERATIONS):
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        best_fitness_history.append(best_fitness)

        # Oblicz diversyfikację
        div = diversity(population)

        # Określ stan
        success_rate = sum(recent_successes[-WINDOW_SIZE:]) / max(1, len(recent_successes[-WINDOW_SIZE:]))
        state = get_state(success_rate, div)

        # Wybierz akcję przez Q-learning
        action_idx = choose_action(state)
        strategy, F = decode_action(action_idx)

        # Mutacja
        mutants = []
        for i in range(POP_SIZE):
            mutant = mutate(population, best_individual, population[i], F, strategy)
            mutants.append(mutant)
        mutants = np.array(mutants)

        # Krzyżowanie
        trials = crossover(population, mutants, CR)
        trials = bounds_check(trials, best_individual)
        trial_fitness = func(trials)

        # Selekcja
        new_population = []
        new_fitness = []
        successes = 0
        for ind, fit, trial, trial_fit in zip(population, fitness, trials, trial_fitness):
            if trial_fit < fit:
                new_population.append(trial)
                new_fitness.append(trial_fit)
                successes += 1
            else:
                new_population.append(ind)
                new_fitness.append(fit)

        population = np.array(new_population)
        fitness = np.array(new_fitness)

        # Zapamiętaj sukcesy
        recent_successes.append(successes)
        if len(recent_successes) > WINDOW_SIZE * 2:
            recent_successes.pop(0)

        # Oblicz nagrodę (poprawa najlepszego fitnessu)
        reward = float(best_fitness_history[-2] - best_fitness_history[-1]) if len(best_fitness_history) >= 2 else 0

        # Aktualizuj Q-table
        next_success_rate = sum(recent_successes[-WINDOW_SIZE:]) / max(1, len(recent_successes[-WINDOW_SIZE:]))
        next_div = diversity(population)
        next_state = get_state(next_success_rate, next_div)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action_idx] = q_table[state][action_idx] + ALPHA * (
            reward + GAMMA * q_table[next_state][best_next_action] - q_table[state][action_idx])

        # Logowanie
        if generation % 100 == 0 or generation == GENERATIONS - 1:
            print(f'{func_name} | Gen {generation}: Best = {best_fitness:.4f}, Strategy: {strategy}, F={F:.2f}')

    return best_fitness


# Główna pętla
for func_name in SELECTED_FUNCTIONS:
    func = getattr(functions, func_name)
    print(f'\n--- Running DE with Q-learning on {func_name} ---')
    best_value = differential_evolution_qlearning(func, func_name)
    print(f'Best value for {func_name}: {best_value:.6f}')