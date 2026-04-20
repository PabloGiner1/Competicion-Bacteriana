# Modelo de Competición Bacteriana (PDH) en Redes Complejas

Este repositorio contiene la implementación y simulación de un sistema biológico de tres estados ($P, D, H$) para estudiar la interacción entre bacterias patógenas y depredadoras en diferentes topologías de red.

## 1. Descripción del Modelo
El modelo simula la evolución de una población bacteriana en el pulmón, donde cada nodo de la red puede encontrarse en uno de los siguientes tres estados:
* **$P$ (Patógena):** Bacteria infecciosa que coloniza el tejido.
* **$D$ (Depredadora):** Bacteria introducida para combatir a la patógena.
* **$H$ (Huérfano/Vacío):** Nodo de tejido sano o espacio disponible.

### Reglas de Transición
La dinámica se rige por tres procesos estocásticos basados en contactos locales:
1. **Colonización ($\beta$):** Un nodo sano ($H$) se convierte en patógeno ($P$) por contacto con un vecino infectado.
2. **Depredación ($\alpha$):** Una bacteria depredadora ($D$) elimina a una patógena ($P$) y ocupa su lugar tras el contacto.
3. **Mortalidad ($\mu$):** Las bacterias depredadoras mueren de forma natural, dejando el nodo vacío ($H$).

## 2. Implementación Técnica
La simulación utiliza la **Lógica de Markov** para actualizaciones sincrónicas en cada paso de tiempo. A diferencia de los modelos de mezcla perfecta, aquí la estructura de la red es fundamental.

### Topologías Soportadas
* **Redes Erdos-Renyi:** Para estudiar el comportamiento en redes aleatorias con grado medio constante.
* **Redes Barabasi-Albert:** Para analizar el efecto de nodos con alta conectividad (*hubs*).
* **Redes Watts-Strogatz:** Para observar dinámicas en mundos pequeños (*small-world*).

## 3. Estructura del Código
* `main.py`: Punto de entrada para ejecutar la simulación y generar gráficas.
* `src/simulation/simulation_model.py`: Contiene la lógica central `simulate_pdh`.
* `src/utils/`: Funciones auxiliares para la generación de grafos y visualización de datos.

## 4. Requisitos
* Python 3.x
* NetworkX
* Matplotlib
* Numpy
