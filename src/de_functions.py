from config import *

def mutate(pop, best, i, strategy, F):
    idxs = [idx for idx in range(len(pop)) if idx != i]
    if strategy == 'rand/1':
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
        return a + F * (b - c)
    elif strategy == 'best/1':
        b, c = pop[np.random.choice(idxs, 2, replace=False)]
        return best + F * (b - c)
    elif strategy == 'rand-to-best/1':
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
    elif strategy == 'current-to-best/1':
        a = pop[i]
        b, c = pop[np.random.choice(idxs, 2, replace=False)]
        return a + F * (best - a) + F * (b - c)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def crossover(target, mutant):
    cross_points = np.random.rand(DIM) < CR
    if not np.any(cross_points):
        cross_points[np.random.randint(0, DIM)] = True
    return np.where(cross_points, mutant, target)


def bound_check(v):
    return np.clip(v, LOWER, UPPER)


def differential_evolution(func, func_name, run_num):
    # Initialize population within bounds
    pop = np.random.uniform(LOWER, UPPER, size=(DE_CONFIG["pop_size"], DIM))

    # Evaluate initial population
    fitness = np.array([float(func(individual.reshape(1, -1))) for individual in pop])

    best_idx = np.argmin(fitness)
    best_individual = pop[best_idx].copy()
    best_fitness = fitness[best_idx]

    optimum = OPTIMUM_VALUES.get(func_name, None)

    print(f"\n{func_name} | Run {run_num}")

    for gen in range(DE_CONFIG["generations"]):
        new_pop = []
        new_fitness = []

        for i in range(DE_CONFIG["pop_size"]):
            # Mutation
            mutant = mutate(pop, best_idx, i, MUTATION_STRATEGY, F_VALUE)

            # Crossover (uses global CR)
            trial = crossover(pop[i], mutant)

            # Apply boundaries
            trial = bound_check(trial)

            # Evaluate trial solution
            f = float(func(trial.reshape(1, -1)))

            # Selection
            if f < fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])

        # Update population and fitness values
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]

        # Check if optimum is reached (if known)
        if optimum is not None:
            if best_val <= optimum + 0.0001:
                print(
                    f"{func_name} | Gen {gen} | Optimum reached: {best_val:.5f} <= {optimum:.5f} + 0.0001"
                )
                break

        # Logging every 100 generations or at the last one
        if gen % 100 == 0 or gen == DE_CONFIG["generations"] - 1:
            print(f'{func_name} | Generation {gen}: best = {best_val:.4f}')

    return best_val