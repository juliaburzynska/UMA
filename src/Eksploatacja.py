from DE import *
from config import *
import numpy as np

print("\n" + "=" * 60)

for fname in SELECTED_FUNCTIONS:
    print(f"WYKORZYSTANIE wytrenowanej tablicy Q | Funkcja: {fname}")
    func = getattr(functions, fname)
    optimum = OPTIMUM_VALUES.get(fname, None)
    results = []

    # Wczytaj wytrenowaną tablicę Q z pliku
    q_file = f"q_table_{fname}.npy"
    try:
        Q_loaded = np.load(q_file)
        print(f"Wczytano Q-tabelę z pliku: {q_file}")
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {q_file}. Używam pustej Q-tabeli.")
        Q_loaded = np.zeros_like(Q_global)  # lub inny rozmiar zgodny z Q_global

    # Reset odwiedzonych stanów i liczników
    visited_states.clear()
    state_visit_counts.fill(0)

    for run in range(EXPLOITATION_RUNS):
        print(f"\n--- Eksploatacja {run + 1}/{EXPLOITATION_RUNS} ---")
        
        # Inicjalizuj Q_global z wczytanego Q przed każdym runem
        Q_global[:] = Q_loaded
        
        result = differential_evolution(func, fname, Q_global, config=EXPLOITATION_CONFIG, optimum=optimum)
        results.append(result)
        coverage = 100.0 * len(visited_states) / ((len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1))
        print(f"Pokrycie przestrzeni stanów po runie {run + 1}: {coverage:.2f}%")
