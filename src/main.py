
import numpy as np
from cec2017 import functions

DIMENSION = 10
LOWER_BOUND = -100
UPPER_BOUND = 100
POP_SIZE = 500
lamb = POP_SIZE
CR = 0.6
GENERATIONS = 1000
SELECTED_FUNCTIONS = ['f7', 'f10']
PATIENCE = 50

# Możliwe strategie mutacji
strategies = ['rand', 'best', 'rand-to-best']
F_values = [0.3, 0.5, 0.8, 1.0]

# Prawdopodobieństwa użycia strategii (startowo równe)
strategy_prob = np.ones(len(strategies)) / len(strategies)
F_prob = np.ones(len(F_values)) / len(F_values)


def select_strategy():
    strategy_idx = np.random.choice(len(strategies), p=strategy_prob)
    F_idx = np.random.choice(len(F_values), p=F_prob)
    return strategies[strategy_idx], F_values[F_idx], strategy_idx, F_idx


def mutate(population, best, strategy, F):
    mutated = []
    for i in range(len(population)):
        idxs = [idx for idx in range(len(population)) if idx != i]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]

        if strategy == 'rand':
            mutant = a + F * (b - c)
        elif strategy == 'best':
            mutant = best + F * (b - c)
        elif strategy == 'rand-to-best':
            mutant = a + F * (best - a) + F * (b - c)

        mutated.append(mutant)
    return np.array(mutated)


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
    while len(children) < lamb:
        children = np.concatenate((children, [best_individual]), axis=0)
    return children


def update_probabilities(strategy_idx, F_idx, improvement, strategy_prob, F_prob, alpha=0.01):
    if improvement:
        strategy_prob[strategy_idx] += alpha
        F_prob[F_idx] += alpha
    else:
        strategy_prob[strategy_idx] -= alpha / 2
        F_prob[F_idx] -= alpha / 2

    # Normalizacja
    strategy_prob = np.clip(strategy_prob, 0.01, None)
    F_prob = np.clip(F_prob, 0.01, None)
    strategy_prob /= np.sum(strategy_prob)
    F_prob /= np.sum(F_prob)

    return strategy_prob, F_prob


def differential_evolution(func, func_name):
    global strategy_prob, F_prob

    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))
    fitness = func(population)

    best_overall = np.min(fitness)
    no_improvement_count = 0
    PATIENCE = 50  # Liczba generacji bez poprawy

    for generation in range(GENERATIONS):
        current_best = np.min(fitness)

        if current_best < best_overall:
            best_overall = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Jeśli długo nie ma poprawy - zwiększamy rozrzut osobników
        if no_improvement_count > PATIENCE:
            print(f"🔍 Brak postępu przez {PATIENCE} generacji. Zwiększam rozrzut osobników...")

            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            # Dodaj niewielką losową perturbację
            noise = np.random.uniform(-5, 5, size=population.shape)
            population = np.clip(population + noise, LOWER_BOUND, UPPER_BOUND)

            population[best_idx] = best_individual  # Zachowaj najlepszego

            # Odśwież wartości fitness
            fitness = func(population)

            no_improvement_count = 0

            # Zwiększ entropię wyboru strategii/F
            strategy_prob = np.ones(len(strategies)) / len(strategies)
            F_prob = np.ones(len(F_values)) / len(F_values)

            print("🧬 Rozrzut osobników został zwiększony.")

        # Reszta algorytmu bez zmian
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        strategy, F, strategy_idx, F_idx = select_strategy()

        mutated = mutate(population, best_individual, strategy, F)
        crossed = crossover(population, mutated, CR)
        children = bounds_check(crossed, best_individual)
        child_fitness = func(children)

        improvement = False
        new_population = []
        new_fitness = []

        for i in range(POP_SIZE):
            if child_fitness[i] < fitness[i]:
                new_population.append(children[i])
                new_fitness.append(child_fitness[i])
                improvement = True
            else:
                new_population.append(population[i])
                new_fitness.append(fitness[i])

        population = np.array(new_population)
        fitness = np.array(new_fitness)

        strategy_prob, F_prob = update_probabilities(strategy_idx, F_idx, improvement, strategy_prob, F_prob)

        if generation % 100 == 0 or generation == GENERATIONS - 1:
            print(
                f'{func_name} | Gen {generation:>4} | best = {np.min(fitness):.4f} | strategy = {strategy:<12} | F = {F}')

    return np.min(fitness)

# Główna pętla testowa z dodatkowymi informacjami
for func_name in SELECTED_FUNCTIONS:
    print("\n" + "=" * 60)
    print(f"🔍 Start testu funkcji: {func_name}")
    print("=" * 60)

    # Resetowanie rozkładów na początek testu funkcji
    strategy_prob = np.ones(len(strategies)) / len(strategies)
    F_prob = np.ones(len(F_values)) / len(F_values)

    # Wypisanie możliwych strategii i F
    print("🎲 Dostępne strategie mutacji:", strategies)
    print("🎯 Dostępne wartości F:", F_values)
    print(f"📊 Początkowe prawdopodobieństwa strategii: {strategy_prob}")
    print(f"📊 Początkowe prawdopodobieństwa F:         {F_prob}")

    func = getattr(functions, func_name)

    print(f"\n🚀 Rozpoczynam optymalizację funkcji {func_name}...\n")
    best_value = differential_evolution(func, func_name)

    print(f"\n✅ Zakończono optymalizację funkcji {func_name}")
    print(f"🏁 Najlepsza znaleziona wartość: {best_value:.6f}")
    print(f"📈 Końcowe prawdopodobieństwa strategii: {strategy_prob}")
    print(f"📈 Końcowe prawdopodobieństwa F:         {F_prob}")
    print("=" * 60)

