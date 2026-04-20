# Configuracion basica para la sinulacion.

N_NODES = 400

# Parámetros red
ERDOS_P = 0.01 #Recordar que el grado medio es erdos_p*N_NODES, buscamos que sea un valor bajo para no estar en grado medio.
BARABASI_M = 3
WATTS_K = 6
WATTS_P = 0.1

# Parámetros PDH
BETHA = 0.3 #Colonización (H -> P)
ALPHA = 0.15 #Depredación (P -> D)
MU = 0.05 #Muerte natural (D -> H)

pdh_params = [BETHA, ALPHA, MU]

#Establecer proporciones iniciales normalizadas
P_PROPORTION = 0.2
D_PROPORTION = 0.05
H_PROPORTION = 0.75

initial_proportions = [P_PROPORTION, D_PROPORTION, H_PROPORTION]


STEPS = 500

# Tipo de grafo a simular
graph_type = "erdos"  # "erdos", "barabasi", "watts"