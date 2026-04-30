# Configuracion basica para la sinulacion.
N_NODES = 400
STEPS = 150

# Parámetros red
ERDOS_P = 0.05 #Recordar que el grado medio es k = erdos_p*N_NODES, buscamos que sea un valor bajo para no estar en grado medio.
BARABASI_M = 3
WATTS_K = 6
WATTS_P = 0.1

# Parámetros PDH. La k_crit = mu / alpha
ALPHA = 0.2 #Colonización (H -> P)
BETHA = 0.4 #Depredación (P -> D) 
MU = 0.2 #Muerte natural (D -> H)

pdh_params = [ALPHA, BETHA, MU]

#Establecer proporciones iniciales normalizadas
P_PROPORTION = 0.2
D_PROPORTION = 0.05
H_PROPORTION = 0.75

initial_proportions = [P_PROPORTION, D_PROPORTION, H_PROPORTION]

#Establecemos que simulaciones plotearemos
RUN_ORIGINAL = False
RUN_MARKOV_INDIVIDUAL = False
RUN_MARKOV_COMBINED = False
RUN_STATIONARY = False 
RUN_HEATMAP = False
RUN_ANIMATION = True

# Tipo de grafo a simular
graph_type = "erdos"  # "erdos", "barabasi", "watts"