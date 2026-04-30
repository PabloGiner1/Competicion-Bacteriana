from src.visualization.plots import *
from config import *


if __name__ == "__main__":

    if RUN_ORIGINAL:
        print("Ejecutando simulación original...")
        plot_original_simulation()

    if RUN_MARKOV_INDIVIDUAL:
        print("Ejecutando simulación Markov individual para distintos k...")
        plot_markov_individual()

    if RUN_MARKOV_COMBINED:
        print("Ejecutando simulación Markov combinada para distintos k...")
        plot_markov_combined()

    if RUN_STATIONARY:
        print("Ejecutando simulación del estado estacionario vs grado medio...")
        plot_stationary_state()

    if RUN_HEATMAP:
        print("Ejecutando simulación para generar heatmap...")
        plot_heatmap()
    
    if RUN_ANIMATION:
        print("Ejecutando animación de la simulación...")
        animate_pdh_simulation()