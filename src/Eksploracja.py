from DE import *
from config import *

for fname in SELECTED_FUNCTIONS:
    print("\n" + "=" * 60)
    print(f"UCZENIE tablicy Q przez eksplorację | Funkcja: {fname}")
    func = getattr(functions, fname)
    optimum = OPTIMUM_VALUES.get(fname, None)

    # Reset Q-tabeli i liczników przed eksploracją
    Q_global.fill(0)
    visited_states.clear()
    state_visit_counts.fill(0)

    for run in range(EXPLORATION_RUNS):
        print(f"\n--- Exploracja {run + 1}/{EXPLORATION_RUNS} ---")
        differential_evolution(func, fname, Q_global, config=EXPLORATION_CONFIG, optimum=optimum)
        coverage = 100.0 * len(visited_states) / ((len(SUCCESS_BINS) - 1) * (len(DISTANCE_BINS) - 1))
        print(f"Pokrycie przestrzeni stanów po runie {run + 1}: {coverage:.2f}%")

    print("\n" + "=" * 60)

# Zapis tablicy Q
    np.save(f"q_table_{fname}.npy", Q_global)

    # Histogram odwiedzin stanów
    plt.figure(figsize=(10, 6))
    plt.title("Histogram odwiedzin stanów (liczba odwiedzin w każdym stanie)")
    plt.xlabel("Indeks sukcesu (bin)")
    plt.ylabel("Indeks odległości (bin)")
    plt.imshow(state_visit_counts.T, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Liczba odwiedzin')
    plt.tight_layout()
    plt.show()
