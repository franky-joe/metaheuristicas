import numpy as np
import pandas as pd
import csv
import os
import json
from math import sqrt
from itertools import combinations
import random
import time

# Variables globales
contador_funcion_objetivo = 0
n_ciudades = 0  # Inicializar como 0
max_evaluaciones = 3000000 # Valor por defecto
max_tamaño_vecindario = 1000
directorio_logs = '/home/roko/cursos/meta/lab/lab3/logs'


# ============================================================================== # 
# ================================ Utilidades ================================== #
# ============================================================================== # 

# Función para leer la matriz de distancias desde un archivo
def leer_matriz_distancias(filename):
    global n_ciudades
    matriz = np.loadtxt(filename, dtype=int)
    n_ciudades = len(matriz)  # Asignar el número de ciudades
    return matriz

# Función para escribir un logs
def escribir_log(log_file, n,  distancia):
    with open(log_file, 'a') as f:
        f.write(f"{n}, {distancia}\n")

# ============================================================================== # 
# ================================ Utilidades ================================== #
# ============================================================================== # 

# Función para calcular la función objetivo (longitud del recorrido)
def evaluar_funcion_objetivo(matriz_distancias, solucion):

    global contador_funcion_objetivo
    distancia_total = 0
    for i in range(n_ciudades):  # Usar la variable global n_ciudades
        ciudad_actual = solucion[i]
        ciudad_siguiente = solucion[(i + 1) % n_ciudades]  # Ciclo cerrado, regresa al inicio
        distancia_total += matriz_distancias[ciudad_actual, ciudad_siguiente]
    contador_funcion_objetivo += 1  # Contar la evaluación
    #print(distancia_total)
    escribir_log(directorio_logs+"/log-funcionObjetivo.txt", contador_funcion_objetivo, distancia_total)
    return distancia_total



# Función para generar una solución aleatoria (un recorrido aleatorio)
def generar_solucion_aleatoria(n=None):
    if n is None:
        n = n_ciudades  # Si no se proporciona un valor, usar el global
    return np.random.permutation(n)

# Genera solucion goloza a partir de una arista seleccionada aleatoriamente
def generar_solucion_golosa(matriz_distancias):
    # Seleccionar aleatoriamente un par de ciudades para iniciar
    ciudad1, ciudad2 = random.sample(range(n_ciudades), 2)

    # Comenzar la solución con el par de ciudades seleccionadas
    solucion_golosa = [ciudad1, ciudad2]
    ciudades_restantes = set(range(n_ciudades))
    ciudades_restantes.remove(ciudad1)
    ciudades_restantes.remove(ciudad2)

    # Mientras haya ciudades sin visitar
    while ciudades_restantes:
        ultima_ciudad_inicio = solucion_golosa[0]  # Ciudad al inicio del recorrido
        ultima_ciudad_final = solucion_golosa[-1]  # Ciudad al final del recorrido

        # Buscar la ciudad más cercana al inicio o al final del recorrido
        ciudad_mas_cercana = min(
            ciudades_restantes,
            key=lambda ciudad: min(
                matriz_distancias[ultima_ciudad_inicio, ciudad],
                matriz_distancias[ultima_ciudad_final, ciudad]
            )
        )

        # Decidir si agregarla al inicio o al final
        if matriz_distancias[ultima_ciudad_inicio, ciudad_mas_cercana] < matriz_distancias[ultima_ciudad_final, ciudad_mas_cercana]:
            solucion_golosa.insert(0, ciudad_mas_cercana)  # Agregar al inicio
        else:
            solucion_golosa.append(ciudad_mas_cercana)  # Agregar al final

        ciudades_restantes.remove(ciudad_mas_cercana)

    return np.array(solucion_golosa)



# ============================================================================== # 
# ================================ Vecindades ================================== #
# ============================================================================== # 

# Funcion para hacer el Swap de 2-opt a una ruta(solucion)
def TwoOptSwap(route, i, k):
    new_route = route.copy()
    # Invertir el segmento de i a k
    new_route[i:k+1] = new_route[i:k+1][::-1]
    
    return new_route

def generarVecindarioTwoOpt(route):
    neighbors = []
    iteraciones = 0
    
    for i in range(0, max_tamaño_vecindario):
        i = random.randint(0, n_ciudades)
        k = random.randint(0, n_ciudades)
        if i == k:
            k = random.randint(0, n_ciudades)

        # Genera una nueva ruta haciendo el intercambio de TwoOpt
        new_route = TwoOptSwap(route, i, k)
        
        neighbors.append(new_route)
        iteraciones += 1
    
    return neighbors


# ============================================================================== # 
# ============================== Perturbacion ================================== #
# ============================================================================== # 

def perturbar_solucion(solucion, nivel_perturbacion):
    nivel_perturbacion = 1 + int(n_ciudades * nivel_perturbacion)
    if nivel_perturbacion < 1:
        nivel_perturbacion = 1

    solucion_perturbada = solucion.copy()
    
    # Realizar 'nivel_perturbacion' intercambios
    for _ in range(nivel_perturbacion):
        # Elegir dos pares de posiciones aleatorias para cada perturbación
        i, j = np.random.choice(len(solucion_perturbada), 2, replace=False)
        # Intercambiar las posiciones
        solucion_perturbada[i], solucion_perturbada[j] = solucion_perturbada[j], solucion_perturbada[i]
    
    return solucion_perturbada

def perturbar_solucion_twoOpt(solucion, nivel_perturbacion):
    nivel_perturbacion = 1 + int(n_ciudades * nivel_perturbacion)
    if nivel_perturbacion < 1:
        nivel_perturbacion = 1
    solucion_perturbada = solucion.copy()
    for _ in range(nivel_perturbacion):
        i = np.random.randint(1, n_ciudades - 1)
        k = np.random.randint(i + 1, n_ciudades)
        solucion_perturbada = TwoOptSwap(solucion_perturbada, i, k)
    return solucion_perturbada



# ============================================================================== # 
# ================================== LS ======================================== #
# ============================================================================== # 

contador_ls = 0 

def busqueda_local(matriz_distancias, solucion_inicial, porcentaje_mejora=0.0001):
    global contador_funcion_objetivo, contador_ls
    mejor_solucion = solucion_inicial
    mejor_distancia = evaluar_funcion_objetivo(matriz_distancias, mejor_solucion)
    
    mejoras = [mejor_distancia]  # Guardamos las distancias de las últimas iteraciones

    mejora = True
    while mejora and contador_funcion_objetivo < max_evaluaciones:
        escribir_log(directorio_logs + "/log-LC.txt", contador_ls, mejor_distancia)
        mejora = False
        vecinos = generarVecindarioTwoOpt(mejor_solucion)        
        for vecino in vecinos:
            distancia_vecino = evaluar_funcion_objetivo(matriz_distancias, vecino)
            if distancia_vecino <= mejor_distancia:
                mejor_solucion = vecino
                mejor_distancia = distancia_vecino
                #print(len(mejor_solucion))
                mejora = True

        mejoras.append(mejor_distancia)  # Guardamos la distancia actual

        # Mantenemos solo las últimas 6 distancias
        if len(mejoras) > 6:
            mejoras.pop(0)

        # Verificamos si la última distancia ha mejorado respecto a la primera
        if len(mejoras) == 6:
            mejora_porcentaje = (1 - (mejoras[-1] / mejoras[0])) * 100
            if mejora_porcentaje < porcentaje_mejora:
                break  # Salimos si no mejora el porcentaje
        contador_ls += 1
    return mejor_solucion, mejor_distancia


# ============================================================================== # 
# ================================== ILS ======================================= #
# ============================================================================== # 


# Variables globales
mejor_solucion_de_todas = np.array([])
mejor_distancia_de_todas = 0

def ILS(ruta_matrix, solucion_inicial=None, max_eval=300000, porcentaje_mejora_min_LS=0.01, inicio_goloso=True, nivel_perturbacion=0.001, 
        tipo_de_perturbacion=1, dir_log='./logs', semilla=None, print_progreso=False, max_tam_vecindario=20000):

    # Leer la matriz de distancias
    matriz_distancias = leer_matriz_distancias(ruta_matrix)
    
    #ramdon_search(1000, matriz_distancias )
    # Extraer el nombre del archivo de la matriz de distancias (sin la extensión)
    nombre_matriz = os.path.splitext(os.path.basename(ruta_matrix))[0]
    
    # Set the max_tamaño_vecindario
    global max_tamaño_vecindario
    #if max_tam_vecindario > 10000:
    #    max_tam_vecindario = 10000
    max_tamaño_vecindario = max_tam_vecindario
    # Crear un nuevo directorio con el nombre de la matriz de distancias si no existe
    global directorio_logs
    directorio_logs = os.path.join(dir_log, nombre_matriz)
    if not os.path.exists(directorio_logs):
        os.makedirs(directorio_logs)
    
    # Abrir archivos de logs en el nuevo directorio
    with open(directorio_logs + "/log-LC.txt", 'w') as f:
        f.write("Iteración, distancia\n")
    with open(os.path.join(directorio_logs, "log-funcionObjetivo.txt"), 'w') as f:
        f.write("Iteración, ContadorFuncionObjetivo\n")
    with open(os.path.join(directorio_logs, "log-iteraciones-ciclo-principal.txt"), 'w') as f:
        f.write("Iteración ciclo principal, distancia\n")

    if semilla is None:
        semilla = int(time.time())
    np.random.seed(semilla)
    random.seed(semilla)

    if solucion_inicial is None:
        if inicio_goloso:
            solucion_inicial = generar_solucion_golosa(matriz_distancias)  # Solución golosa
        else:
            solucion_inicial = generar_solucion_aleatoria()  # Solución aleatoria

    global contador_funcion_objetivo, max_evaluaciones
    max_evaluaciones = max_eval

    # Inicializar el contador de evaluaciones y los logs
    mejor_solucion, mejor_distancia = busqueda_local(matriz_distancias, solucion_inicial, porcentaje_mejora=porcentaje_mejora_min_LS)
    escribir_log(os.path.join(directorio_logs, "log-iteraciones-ciclo-principal.txt"), 0, mejor_distancia)

    iteracion = 1

    # Verificar si es la mejor solución global
    global mejor_solucion_de_todas, mejor_distancia_de_todas
    mejor_solucion_de_todas = mejor_solucion.copy()
    mejor_distancia_de_todas = mejor_distancia

    while contador_funcion_objetivo < max_evaluaciones:
        # 1. Perturbar la mejor solución
        if tipo_de_perturbacion == 1:
            solucion_perturbada = perturbar_solucion(mejor_solucion, nivel_perturbacion)
        elif tipo_de_perturbacion == 2:
            solucion_perturbada = perturbar_solucion_twoOpt(mejor_solucion, nivel_perturbacion)
        
        # 2. Aplicar búsqueda local a la solución perturbada
        nueva_solucion, nueva_distancia = busqueda_local(matriz_distancias, solucion_perturbada, porcentaje_mejora=porcentaje_mejora_min_LS)

        # 3. Si la nueva solución es mejor, actualizar la mejor solución
        if nueva_distancia <= mejor_distancia:
            mejor_solucion = nueva_solucion
            mejor_distancia = nueva_distancia
        
        # 4. Reducir perturbacion si quedan pocas llamadas a la funcion objetivo
        if max_evaluaciones - contador_funcion_objetivo < 100000:
            nivel_perturbacion = int(nivel_perturbacion * 0.01)

        # Verificar si es la mejor solución global
        if mejor_distancia <= mejor_distancia_de_todas:
            mejor_solucion_de_todas = mejor_solucion.copy()
            mejor_distancia_de_todas = mejor_distancia

        # Actualizar el contador de evaluaciones y escribir logs
        iteracion += 1
        escribir_log(os.path.join(directorio_logs, "log-iteraciones-ciclo-principal.txt"), iteracion, mejor_distancia)
        if print_progreso:
            print(contador_funcion_objetivo / max_evaluaciones)
    #print(mejor_distancia_de_todas)
    #print(mejor_solucion_de_todas)
    # Guardar la mejor solución y sus datos en un archivo JSON
    resultado = {
        "instancia": nombre_matriz,
        "mejor_distancia_de_todas": float(mejor_distancia_de_todas),
        "semilla": semilla,
        "max_tam_vecindario": max_tam_vecindario,
        "tipo_de_perturbacion": tipo_de_perturbacion,
        "nivel_perturbacion": nivel_perturbacion,
        "porcentaje_mejora_min_LS": porcentaje_mejora_min_LS,
        "max_evaluaciones": max_evaluaciones,
        "mejor_solucion_de_todas": mejor_solucion_de_todas.tolist()
    }

    with open(os.path.join(directorio_logs, "mejor_solucion.json"), 'w') as f:
        json.dump(resultado, f, indent=4)

    return mejor_solucion_de_todas, mejor_distancia_de_todas, semilla

import argparse
import time


def elementos_repetidos(arr):
    # Obtener los elementos únicos y sus conteos
    elementos_unicos, conteo = np.unique(arr, return_counts=True)
    
    # Filtrar los elementos que tienen un conteo mayor a 1
    repetidos = elementos_unicos[conteo > 1]
    
    return repetidos


if __name__ == "__main__":
    # Configuración de argparse para recibir argumentos
    parser = argparse.ArgumentParser(description="Optimización del TSP usando ILS")
    parser.add_argument("--ruta_matrix", type=str, default='../Instancias/matrix-uy734.txt', help="Ruta al archivo de matriz de distancias")
    parser.add_argument("--nivel_perturbacion", type=float, default=0.01, help="Nivel de perturbación")
    parser.add_argument("--tipo_de_perturbacion", type=int, default=1, help="Tipo de perturbación")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla para la alea oriedad")
    parser.add_argument("--max_evaluaciones", type=int, default=50000000, help="Máximo de evaluaciones")
    parser.add_argument("--max_tamaño_vecindario", type=int, default=20000, help="Máximo tamaño del vecindario")
    parser.add_argument("--porcentaje_mejora_LS", type=float, default=0.00001, help="Porcentaje de mejora en la búsqueda local")
    parser.add_argument("--dir_log_out", type=str, default='./logs', help="Logs de salida")
    parser.add_argument("--print_progreso", type=bool, default=False, help="Imprimir progreso")

    args = parser.parse_args()
    # Llamar a la búsqueda local iterativa con los parámetros recibidos
    solucion_final, distancia_final, semilla = ILS(ruta_matrix=args.ruta_matrix,
                                                    nivel_perturbacion=args.nivel_perturbacion,
                                                    tipo_de_perturbacion=args.tipo_de_perturbacion,
                                                    porcentaje_mejora_min_LS=args.porcentaje_mejora_LS,
                                                    max_eval=args.max_evaluaciones,
                                                    print_progreso=args.print_progreso,
                                                    max_tam_vecindario=args.max_tamaño_vecindario,
                                                    dir_log=args.dir_log_out)
    
    #print(len(solucion_final))
    print(distancia_final)
    #print(len(elementos_repetidos(solucion_final)))


    #print(len(solucion_final))  # Esto funcionará