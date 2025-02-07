import numpy as np
import csv
import os

# Función para leer la matriz de distancias desde un archivo
def leer_matriz_distancias(filename):
    global n_ciudades
    matriz = np.loadtxt(filename, dtype=int)
    n_ciudades = len(matriz)  # Asignar el número de ciudades
    return matriz

# Función para calcular la función objetivo (longitud del recorrido)
def evaluar_funcion_objetivo(matriz_distancias, solucion):
    distancia_total = 0
    for i in range(n_ciudades):  # Usar la variable global n_ciudades
        ciudad_actual = solucion[i]
        ciudad_siguiente = solucion[(i + 1) % n_ciudades]  # Ciclo cerrado, regresa al inicio
        distancia_total += matriz_distancias[ciudad_actual, ciudad_siguiente]
    return distancia_total

# Función para generar una solución aleatoria (un recorrido aleatorio)
def generar_solucion_aleatoria(n=None):
    if n is None:
        n = n_ciudades  # Si no se proporciona un valor, usar el global
    return np.random.permutation(n)

# Función para realizar random search con 'n' soluciones por cada iteración
def generar_mejores_soluciones_y_guardar_csv(matriz_distancias, n_solutions, iteraciones, archivo_csv):
    mejores_distancias = []

    for iteracion in range(iteraciones):
        mejor_distancia = float('inf')  # Inicializar con un valor muy grande

        # Generar 'n_solutions' soluciones aleatorias y quedarse con la mejor
        for i in range(n_solutions):
            solucion_aleatoria = generar_solucion_aleatoria()
            distancia = evaluar_funcion_objetivo(matriz_distancias, solucion_aleatoria)

            if distancia < mejor_distancia:
                mejor_distancia = distancia
        
        # Guardar la mejor distancia de la iteración
        mejores_distancias.append([mejor_distancia])
    
    # Guardar las mejores distancias en un archivo CSV
    with open(archivo_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Escribir la cabecera del CSV
        writer.writerow(['Iteracion', 'Mejor Distancia'])
        # Escribir las mejores distancias de cada iteración
        for i, distancia in enumerate(mejores_distancias):
            writer.writerow([i + 1, distancia[0]])

# Ejemplo de uso
archivo_matriz = '../Instancias/matrix-rw1621.txt'  # Cambia esta ruta por la correcta
matriz_distancias = leer_matriz_distancias(archivo_matriz)

# Crear el nombre del archivo de salida basado en el nombre de la matriz
nombre_archivo_salida = f"{os.path.splitext(os.path.basename(archivo_matriz))[0]}-mejores.csv"

# Generar 200 soluciones por iteración, repetir 31 veces y guardar en un CSV
generar_mejores_soluciones_y_guardar_csv(matriz_distancias, 30000, 31, nombre_archivo_salida)
