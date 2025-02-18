import random
import numpy as np
import pandas as pd  # Asegúrate de tener pandas instalado

# Variable global para almacenar los fitness de cada generación

fitness_matrix = [] # filas = generacion, columna = individuo 

# Clase que representa un individuo (una solución al TSP)
class Individuo:
    def __init__(self, ruta):
        self.ruta = ruta
        self.fitness = self.evaluar_fitness()

    def evaluar_fitness(self):
        global matriz_distancias  # Accedemos a la matriz de distancias global
        distancia_total = 0
        n_ciudades = len(self.ruta)
        for i in range(n_ciudades):
            ciudad_actual = self.ruta[i]
            ciudad_siguiente = self.ruta[(i + 1) % n_ciudades]  # Ciclo cerrado
            distancia_total += matriz_distancias[ciudad_actual, ciudad_siguiente]
        return distancia_total

# Clase que representa una población de individuos
class Poblacion:
    def __init__(self, tamaño, n_ciudades):
        self.individuos = self.inicializar(tamaño, n_ciudades)

    def inicializar(self, tamaño, n):
        poblacion = []
        for _ in range(tamaño):
            ruta = np.random.permutation(n).tolist()  # Generar una ruta aleatoria
            poblacion.append(Individuo(ruta))
        return poblacion

    def evaluar(self):
        for individuo in self.individuos:
            individuo.fitness = individuo.evaluar_fitness()

    def seleccion_torneo(self, n, k=30):
        seleccionados = []
        for _ in range(n):
            torneo = random.sample(self.individuos, k)  # Seleccionar k individuos al azar
            mejor = min(torneo, key=lambda x: x.fitness)  # El mejor en el torneo
            seleccionados.append(mejor)
        return seleccionados

    def recombinacion(self, padres):
        hijos = []
        while len(hijos) < len(padres):
            # Seleccionamos dos padres
            padre1, padre2 = random.sample(padres, 2)
            # Cruzamos los padres
            hijo_ruta = self.cruzar(padre1.ruta, padre2.ruta)
            hijos.append(Individuo(hijo_ruta))
        return hijos

    def cruzar(self, ruta1, ruta2):
        n = len(ruta1)
        inicio, fin = sorted(random.sample(range(n), 2))
        hijo_ruta = [-1] * n
        
        # Copiamos una parte de la ruta del primer padre
        hijo_ruta[inicio:fin] = ruta1[inicio:fin]
        
        # Llenamos el resto de la ruta con genes del segundo padre
        index = 0
        for ciudad in ruta2:
            if ciudad not in hijo_ruta:
                while hijo_ruta[index] != -1:
                    index += 1
                hijo_ruta[index] = ciudad
        return hijo_ruta

    def mutacion(self, prob_mutacion=0.01, n_swaps=5):
        n_mutar = int(len(self.individuos) * prob_mutacion)
        for i in random.sample(range(len(self.individuos)), n_mutar):
            hijo = self.individuos[i]
            for _ in range(n_swaps):
                idx1, idx2 = random.sample(range(len(hijo.ruta)), 2)
                # Intercambiamos las ciudades
                hijo.ruta[idx1], hijo.ruta[idx2] = hijo.ruta[idx2], hijo.ruta[idx1]
            hijo.fitness = hijo.evaluar_fitness()  # Actualizamos el fitness

    def obtener_mejores(self, n):
        # Devolver los n mejores individuos
        return sorted(self.individuos, key=lambda x: x.fitness)[:n]


# Función principal del algoritmo genético
def genetico(matriz, max_generaciones, cantidad):
    global matriz_distancias
    global fitness_matrix  # Usar la matriz de fitness global
    matriz_distancias = matriz
    n = len(matriz)  # Número de ciudades
    poblacion = Poblacion(cantidad, n)
    poblacion.evaluar()
    
    # Inicializar la matriz de fitness
    fitness_matrix.append([ind.fitness for ind in poblacion.individuos])

    for i in range(max_generaciones):
        padres = poblacion.seleccion_torneo(cantidad)
        hijos = poblacion.recombinacion(padres)
        poblacion.individuos.extend(hijos)  # Añadir hijos a la población
        poblacion.mutacion()  # Mutar una parte de la población
        poblacion.evaluar()  # Reevaluar la población
        
        # Guardar el fitness de la nueva población
        fitness_matrix.append([ind.fitness for ind in poblacion.individuos])

        poblacion.individuos = poblacion.obtener_mejores(cantidad)  # Mantener tamaño constante
    
    return poblacion.obtener_mejores(1)  # Devolver el mejor individuo

# Función para leer la matriz de distancias desde un archivo
def leer_matriz(file):
    return np.loadtxt(file, dtype=int)

# Función para guardar la matriz de fitness en un archivo CSV
def guardar_fitness_en_csv(nombre_archivo):
    global fitness_matrix
    df = pd.DataFrame(fitness_matrix)  # Convertir la matriz a un DataFrame de pandas
    df.to_csv(nombre_archivo, index=False, header=False)  # Guardar en CSV sin índice ni cabecera

# Ejemplo de uso
file = "Instancias/matrix-wi29.txt"
matriz_distancias = leer_matriz(file)
mejor_solucion = genetico(matriz_distancias, max_generaciones=70, cantidad=1000)

# Guardar la matriz de fitness en un archivo CSV
guardar_fitness_en_csv("fitness_generaciones.csv")

# Mostrar la mejor solución y su fitness
print("Mejor ruta:", mejor_solucion[0].ruta)
print("Mejor fitness (distancia total):", mejor_solucion[0].fitness)
