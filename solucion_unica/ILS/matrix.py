import numpy as np
import math

# Función para leer las coordenadas de un archivo TSP
def leer_coordenadas_archivo(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Leer las coordenadas a partir de la sección NODE_COORD_SECTION
    coords = []
    start_reading = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            start_reading = True
            continue
        if "EOF" in line:
            break
        if start_reading:
            _, x, y = line.split()
            coords.append((float(x), float(y)))
    
    return np.array(coords)

# Función para calcular la distancia euclidiana truncada a enteros
def calcular_matriz_distancias(coords):
    n = len(coords)
    matriz_distancias = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i+1, n):
            # Calcular la distancia euclidiana
            dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            matriz_distancias[i, j] = matriz_distancias[j, i] = int(dist)  # Truncar y almacenar solo la parte entera
    
    return matriz_distancias

# Función para escribir la matriz de distancias a un archivo
def escribir_matriz_a_archivo(matriz_distancias, output_filename):
    np.savetxt(output_filename, matriz_distancias, fmt='%d', delimiter=' ')
    print(f"Matriz de distancias guardada en {output_filename}")

# Programa principal
def generar_matriz_distancias(archivo_tsp, archivo_salida):
    coords = leer_coordenadas_archivo(archivo_tsp)
    matriz_distancias = calcular_matriz_distancias(coords)
    escribir_matriz_a_archivo(matriz_distancias, archivo_salida)

# Ejemplo de uso
archivo_tsp = '/home/roko/Descargas/rw1621.tsp'  # Cambia a la ruta correcta del archivo
archivo_salida = 'matrix-rw1621.txt'
generar_matriz_distancias(archivo_tsp, archivo_salida)
