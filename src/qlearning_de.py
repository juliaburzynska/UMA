from config import *
from de_functions import *


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

def evaluate_distance(pop):
    dists = []
    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            dists.append(np.linalg.norm(pop[i] - pop[j]))
    return np.mean(dists)


def differential_evolution_qlearning(func, func_name, Q, config, optimum=None, caller_name=None):
    epsilon = config['epsilon']
    pop_size = config['pop_size']
    generations = config['generations']

    pop = np.random.uniform(LOWER, UPPER, (pop_size, DIM))
    fitness = func(pop)
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    success_count = 0
    total_count = 0

    best_fitness_history = []

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

        current_best_fitness = np.min(fitness)
        best_fitness_history.append(current_best_fitness)

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

        # Gradual decay of epsilon for exploration
        if "exploration.py" in caller_name:
            epsilon = max(0.1, epsilon * 0.997)

        pop = np.array(next_pop)
        fitness = np.array(next_fitness)
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        if "exploitation.py" in caller_name:
            if optimum is not None:
                best_val = np.min(next_fitness)
                if best_val <= optimum + 0.0001:
                    print(f"{func_name} | Gen  {gen} | Optimum reached: {best_val:.5f} <= {optimum:.5f} + 0.000001")
                    break

        best_val = np.min(fitness)
        worst_val = np.max(fitness)
        if gen % 100 == 0 or gen == generations - 1:
            print(f"{func_name} | Gen {gen:>4} | Best: {best_val:.5f} | Worst: {worst_val:.5f}")

    total_success_rate = success_count / total_count if total_count > 0 else 0.0
    return np.min(fitness), total_success_rate
