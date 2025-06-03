import numpy as np
from cec2017 import functions

# Parametry algorytmu
DIMENSION = 10
LOWER_BOUND = -100
UPPER_BOUND = 100
POP_SIZE = 100
lamb = POP_SIZE
F = 0.5
CR = 0.9
GENERATIONS = 1000

# Wybór funkcji do testów
SELECTED_FUNCTIONS = ['f7', 'f10']

def mutate(population, F):
    mutated = []
    for i in range(len(population)):
        idxs = [idx for idx in range(len(population)) if idx != i]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
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

def differential_evolution(func, func_name):
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POP_SIZE, DIMENSION))
    fitness = func(population)

    for generation in range(GENERATIONS):
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        mutated = mutate(population, F)
        crossed = crossover(population, mutated, CR)
        children = bounds_check(crossed, best_individual)

        child_fitness = func(children)

        combined = np.vstack((population, children))
        combined_fitness = np.hstack((fitness, child_fitness))
        best_indices = np.argsort(combined_fitness)[:POP_SIZE]
        population = combined[best_indices]
        fitness = combined_fitness[best_indices]

        if generation % 100 == 0 or generation == GENERATIONS - 1:
            print(f'{func_name} | Generation {generation}: best = {np.min(fitness):.4f}')

    return np.min(fitness)

# Główna pętla
for func_name in SELECTED_FUNCTIONS:
    func = getattr(functions, func_name)
    print(f'\n--- Running DE on {func_name} ---')
    best_value = differential_evolution(func, func_name)
    print(f'Best value for {func_name}: {best_value:.6f}')
