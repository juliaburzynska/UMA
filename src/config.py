import sys
from itertools import product
import numpy as np
from cec2017 import functions
from logger import Logger

q_file = f"q_tables/q_table_multi_function.npy"

SELECTED_FUNCTIONS = ['f1', 'f7', 'f14']

OPTIMUM_VALUES = {
    f'f{i}': 100.0 * i for i in range(1, 31)
}

DIM = 10
LOWER, UPPER = -100, 100
CR = 0.9
EVAL_WINDOW = 20

ALPHA = 0.2
GAMMA = 0.9


#Differential evolution
MUTATION_STRATEGY = "best/1"
F_VALUE = 0.7

DE_RUNS = 25
DE_CONFIG = {
    "epsilon": 0.1,
    "pop_size": 100,
    "generations": 1000,
}

#Q-learning - exploration
EXPLORATION_RUNS = 300
EXPLORATION_CONFIG = {
    "epsilon": 0.9,
    "pop_size": 200,
    "generations": 500,
}
EXPLORATION_FUNCTIONS = ['f3', 'f10', 'f15', 'f20', 'f22']

#Q-learning - exploitation
EXPLOITATION_RUNS = 25
EXPLOITATION_CONFIG = {
    "epsilon": 0.1,
    "pop_size": 100,
    "generations": 600,
}

#Q-learning
SUCCESS_BIN_SIZE = 0.05
DISTANCE_BINS = np.array([
    0.0,
    0.2, 0.5, 1.0, 2.0, 4.0, 8.0,
    15.0, 30.0, 60.0, 100.0,
    150.0, 200.0, 250.0, 300.0
])
SUCCESS_BINS = np.arange(0, 1 + SUCCESS_BIN_SIZE, SUCCESS_BIN_SIZE)

STRATEGIES = ['rand/1', 'best/1', 'rand-to-best/1', 'rand/2', 'rand/3', 'best/2', 'current-to-best/1']
F_STEP = 0.1
F_VALUES = np.arange(F_STEP, 1.0 + F_STEP, F_STEP).round(3).tolist()

ACTIONS = list(product(range(len(STRATEGIES)), range(len(F_VALUES))))
NUM_ACTIONS = len(ACTIONS)

Q_global = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1, NUM_ACTIONS))
visited_states = set()
state_visit_counts = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1), dtype=int)
