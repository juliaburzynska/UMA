import numpy as np
from cec2017 import functions

DIMENSION = 10
LOWER_BOUND = -100
UPPER_BOUND = 100
POP_SIZE = 500
CR = 0.6
PATIENCE = 50  # Liczba generacji bez poprawy
GENERATIONS = 1000
TOLERANCE = 1e-4
SELECTED_FUNCTIONS = ['f1', 'f2','f3','f4','f5','f6', 'f7','f8','f9','f10','f11','f12','f13','f14', 'f15','f16','f17','f18','f19',
                      'f20','f21','f22','f23','f24', 'f25','f26','f27','f28','f29', 'f30']

# Możliwe strategie
strategies = ['rand', 'best', 'rand-to-best']
F_values = [0.3, 0.5, 0.8, 1.0]

# Prawdopodobieństwa użycia strategii (startowo równe)
strategy_prob = np.ones(len(strategies)) / len(strategies)
F_prob = np.ones(len(F_values)) / len(F_values)


def mutate_adaptive(population, best, fitness):
    mutated = []
    avg_fitness = np.mean(fitness)

    for i in range(len(population)):
        if fitness[i] < avg_fitness:
            # Dobry osobnik → eksploatacja
            strategy = 'best'
            F = 0.5
        else:
            # Słabszy osobnik → eksploracja
            strategy = np.random.choice(['rand', 'rand-to-best'])
            F = np.random.choice(F_values)

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
    while len(children) < POP_SIZE:
        children = np.concatenate((children, [best_individual]), axis=0)
    return children


def differential_evolution(func, func_name):
    global strategy_prob, F_prob
    perturbation_count = 0
    
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))
    fitness = func(population)

    best_overall = np.min(fitness)
    no_improvement_count = 0
    last_best_value = best_overall

    for generation in range(GENERATIONS):
        current_best = np.min(fitness)

        if current_best < best_overall - TOLERANCE:
            best_overall = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Jeśli długo nie ma poprawy - zwiększamy rozrzut osobników
        if no_improvement_count > PATIENCE:
            print("Rozrzut osobników został zwiększony.")
            perturbation_count += 1

            if abs(best_overall - last_best_value) < TOLERANCE:
                if perturbation_count >= 4:
                    print("Brak postępu po 4 rozrzutach.")
                    break
            else:
                perturbation_count = 0

            last_best_value = best_overall

            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            noise = np.random.uniform(-5, 5, size=population.shape)
            population = np.clip(population + noise, LOWER_BOUND, UPPER_BOUND)
            population[best_idx] = best_individual

            # Odśwież wartości fitness
            fitness = func(population)

            no_improvement_count = 0

            # Zwiększ entropię wyboru strategii/F
            strategy_prob = np.ones(len(strategies)) / len(strategies)
            F_prob = np.ones(len(F_values)) / len(F_values)

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # Mutacja z adaptacyjnym doborem strategii per osobnik
        mutated = mutate_adaptive(population, best_individual, fitness)
        crossed = crossover(population, mutated, CR)
        children = bounds_check(crossed, best_individual)
        child_fitness = func(children)

        # Selekcja
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

        if generation % 100 == 0 or generation == GENERATIONS - 1:
            print(
                f'{func_name} | Gen {generation:>4} | best = {np.min(fitness):.4f}')

    return np.min(fitness)


# Główna pętla testowa
for func_name in SELECTED_FUNCTIONS:
    print("\n" + "=" * 60)
    print(f"Start testu funkcji: {func_name}")
    print("=" * 60)

    func = getattr(functions, func_name)

    best_value = differential_evolution(func, func_name)

    print(f"\n Zakończono optymalizację funkcji {func_name}")
    print(f" Najlepsza znaleziona wartość: {best_value:.6f}")
    print("=" * 60)
