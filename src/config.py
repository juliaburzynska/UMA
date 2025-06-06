import numpy as np
from cec2017 import functions
from itertools import product
from logger import Logger
import sys
import matplotlib.pyplot as plt

sys.stdout = Logger("logZ___f10.txt")

OPTIMUM_VALUES = {
    f'f{i}': 100.0 * i for i in range(1, 31)
}

DIM = 10
LOWER, UPPER = -100, 100
CR = 0.9
EVAL_WINDOW = 20

ALPHA = 0.2
GAMMA = 0.9

SELECTED_FUNCTIONS = ['f10']

EXPLORATION_RUNS = 300
EXPLORATION_CONFIG = {
    "epsilon": 0.9,
    "pop_size": 200,
    "generations": 500,
}

EXPLOITATION_RUNS = 20
EXPLOITATION_CONFIG = {
    "epsilon": 0.1,
    "pop_size": 100,
    "generations": 2000,
}

SUCCESS_BIN_SIZE = 0.01  
DISTANCE_BIN_SIZE = 1     
DISTANCE_BINS = np.arange(0, 250 + DISTANCE_BIN_SIZE, DISTANCE_BIN_SIZE)
SUCCESS_BINS = np.arange(0, 1 + SUCCESS_BIN_SIZE, SUCCESS_BIN_SIZE)

STRATEGIES = ['rand', 'best', 'rand-to-best', 'rand/2', 'rand/3', 'best/2', 'current-to-best']
F_STEP = 0.1
F_VALUES = np.arange(F_STEP, 1.0 + F_STEP, F_STEP).round(3).tolist()

ACTIONS = list(product(range(len(STRATEGIES)), range(len(F_VALUES))))
NUM_ACTIONS = len(ACTIONS)

Q_global = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1, NUM_ACTIONS))
visited_states = set()
state_visit_counts = np.zeros((len(SUCCESS_BINS) - 1, len(DISTANCE_BINS) - 1), dtype=int)