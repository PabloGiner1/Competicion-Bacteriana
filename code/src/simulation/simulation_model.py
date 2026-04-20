import random

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