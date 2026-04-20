# Funciones auxiliares para la simulación.

def count_states(state):
    counts = {"P": 0, "D": 0, "H": 0}
    
    for s in state.values():
        counts[s] += 1
    
    return counts

def max_state_counts(state):
    counts = count_states(state)

    return max(counts.values())
