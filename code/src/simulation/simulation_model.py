import random
import numpy as np

#Simulación del modelo PDH, el funcionamiento es el siguiente: 
#P: Nodo ocupado por la presa (Pseudomona aeruginosa)
#D: Nodo ocupado por la depredadora (Vampirovibrio)
#H: Nodo vacia
#Las bacterias presas son depredadas por las depredadoras, un nodo P en contacto con un nodo D se convierte en D con tasa de depredación alpha. Las bacterias presa pueden ocupar un nodo hueco H con tasa de ocupación beta. Las bacterias depredadoras pueden morir con tasa de mortalidad mu, convirtiendo el nodo en H.
#La red está definida por el parametro G, esta generada por la libreria networkx. A la hora de inicializar la red se asigna una proporcion de nodos como P, D y H, definidos por initial_proportions. En cada paso se actualizan los estados segun los pdh_parameters, que son alpha, beta y mu. La función devuelve un historial de los estados de cada nodo en cada paso de la simulacion.


def simulate_pdh(G, pdh_parameters, initial_proportions, steps):

    # Asigmoas variales iniciales
    p_proportion, d_proportion, h_proportion = initial_proportions
    beta, alpha, mu = pdh_parameters
    
    state = {}
    # Establecer proporciones iniciales
    for node in G.nodes():
        rand = random.random()
        if rand < p_proportion:
            state[node] = "P"
        elif rand < p_proportion + d_proportion:
            state[node] = "D"
        else:
            state[node] = "H"
    
    history = []

    for i in range(steps):
        new_state = state.copy()

        #Logica de actualización de estados
        for node in G.nodes():

            if state[node] == "H":
                # Ocupación de nodos vacíos
                for neighbor in G.neighbors(node):
                    if state[neighbor] == "P":
                        if random.random() < beta:
                            new_state[node] = "P"
                            break

            elif state[node] == "P":
                # Depredación
                for neighbor in G.neighbors(node):
                    if state[neighbor] == "D":
                        if random.random() < alpha:
                            new_state[node] = "D"
                            break
            
            # Mortalidad de depredadores
            elif state[node] == "D":
                if random.random() < mu:
                    new_state[node] = "H"

        state = new_state
        history.append(state.copy())

    return history

"""
    Resuelve la dinámica microscópica de Markov del modelo PDH
    sobre una red con matriz de adyacencia A.

    Estados:
        P -> presa
        D -> depredador
        H -> hueco

    Ecuaciones:
        Pi_PD = 1 - prod_j (1 - alpha * A_ij * d_j)
        Pi_HP = 1 - prod_j (1 - beta  * A_ij * p_j)

        p_i(t+1) = p_i(t) * (1 - Pi_PD) + h_i(t) * Pi_HP
        d_i(t+1) = d_i(t) * (1 - mu)    + p_i(t) * Pi_PD
        h_i(t+1) = h_i(t) * (1 - Pi_HP) + mu * d_i(t)
    """
"""
Este código simula cómo evolucionan tres tipos de nodos en una red (presas P, depredadores D y huecos H) usando probabilidades en lugar de estados fijos. A cada nodo no le asignamos un estado concreto, sino la probabilidad de estar en cada estado (p, d, h). En cada paso de tiempo, calculamos primero la probabilidad de que una presa sea atacada por algún vecino depredador (Pi_PD) y la probabilidad de que un hueco sea colonizado por alguna presa vecina (Pi_HP). Estas probabilidades se calculan teniendo en cuenta todos los vecinos del nodo mediante un producto (esto representa la probabilidad de que ninguno actúe y luego se resta a 1). Con estas probabilidades actualizamos las ecuaciones de Markov: las presas pueden sobrevivir o convertirse en depredadores, los depredadores pueden morir o generarse al comer presas, y los huecos pueden permanecer vacíos o llenarse con presas. Después normalizamos para asegurar que las probabilidades suman 1 (evitando errores numéricos). Finalmente, guardamos en cada paso el valor medio de presas, depredadores y huecos en toda la red para poder ver su evolución en el tiempo.
"""

def markov_pdh_dynamics(A, beta, alpha, mu, P0, D0, H0, T):

    N = A.shape[0]

    # Inicialización
    p = np.full(N, P0, dtype=float)
    d = np.full(N, D0, dtype=float)
    h = np.full(N, H0, dtype=float)

    P_mean = [np.mean(p)]
    D_mean = [np.mean(d)]
    H_mean = [np.mean(h)]

    for _ in range(T):

        # Probabilidad de ataque (P -> D)
        Pi_PD = 1.0 - np.prod(1.0 - alpha * A * d[np.newaxis, :], axis=1)

        # Probabilidad de colonización (H -> P)
        Pi_HP = 1.0 - np.prod(1.0 - beta * A * p[np.newaxis, :], axis=1)

        # Ecuaciones de Markov
        p_new = p * (1.0 - Pi_PD) + h * Pi_HP
        d_new = d * (1.0 - mu) + p * Pi_PD
        h_new = h * (1.0 - Pi_HP) + mu * d

        # Normalización (evita errores numéricos)
        total = p_new + d_new + h_new
        total[total == 0] = 1.0

        p = p_new / total
        d = d_new / total
        h = h_new / total

        P_mean.append(np.mean(p))
        D_mean.append(np.mean(d))
        H_mean.append(np.mean(h))

    return np.array(P_mean), np.array(D_mean), np.array(H_mean)


"""
    Calcula el estado estacionario medio del modelo PDH en redes ER
    para distintos valores del grado medio <k>, promediando sobre varias realizaciones.

    Parámetros:
    - generate_er_graph_func: función que ya usáis para generar la red ER
    - n_nodes: número de nodos
    - k_values: lista de grados medios a estudiar
    - beta, alpha, mu: parámetros del modelo
    - P0, D0, H0: condiciones iniciales
    - T: número de pasos temporales
    - n_realizations: número de redes distintas para cada <k>
    - tail: número de pasos finales usados para estimar el estado estacionario

    Devuelve:
    - P_inf_list, D_inf_list, H_inf_list: fracciones estacionarias medias
"""

def stationary_state_vs_degree(generate_er_graph_func, n_nodes, k_values, beta, alpha, mu,
                               P0, D0, H0, T, n_realizations=5, tail=20):

    P_inf_list = []
    D_inf_list = []
    H_inf_list = []

    P_std_list = []
    D_std_list = []
    H_std_list = []

    for k_avg in k_values:
        P_inf_runs = []
        D_inf_runs = []
        H_inf_runs = []

        for seed in range(n_realizations):
            _, A = generate_er_graph_func(n_nodes, k_avg, seed=seed)

            P, D, H = markov_pdh_dynamics(
                A=A,
                beta=beta,
                alpha=alpha,
                mu=mu,
                P0=P0,
                D0=D0,
                H0=H0,
                T=T
            )

            # Estado estacionario aproximado = promedio de los últimos "tail" pasos
            P_inf_runs.append(np.mean(P[-tail:]))
            D_inf_runs.append(np.mean(D[-tail:]))
            H_inf_runs.append(np.mean(H[-tail:]))

        P_inf_list.append(np.mean(P_inf_runs))
        D_inf_list.append(np.mean(D_inf_runs))
        H_inf_list.append(np.mean(H_inf_runs))

        P_std_list.append(np.std(P_inf_runs))
        D_std_list.append(np.std(D_inf_runs))
        H_std_list.append(np.std(H_inf_runs))

    return (
        np.array(P_inf_list),
        np.array(D_inf_list),
        np.array(H_inf_list),
        np.array(P_std_list),
        np.array(D_std_list),
        np.array(H_std_list)
    )