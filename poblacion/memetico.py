import argparse
import numpy as np
import random
import pandas as pd
import time
from scipy.stats import entropy

# Variable global para almacenar los fitness de cada generación

fitness_matrix = [] # filas = generacion, columna = individuo 

num_llamadas_objetivo = 0
# Clase que representa un individuo (una solución al TSP)
class Individuo:
    def __init__(self, ruta):
        self.ruta = ruta
        self.n_ciudades = len(ruta)
        self.fitness = self.evaluar_fitness_numpy()

    def evaluar_fitness(self):
        global matriz_distancias 
        global num_llamadas_objetivo
        distance_indices = np.roll(self.ruta, -1)
        num_llamadas_objetivo  = num_llamadas_objetivo + 1
        return np.sum(matriz_distancias[self.ruta, distance_indices])

    def evaluar_fitness_numpy(self):
        global matriz_distancias  # Accedemos a la matriz de distancias global
        global num_llamadas_objetivo
        num_llamadas_objetivo  = num_llamadas_objetivo + 1
        distancia_total = 0
        n_ciudades = len(self.ruta)
        for i in range(n_ciudades):
            ciudad_actual = self.ruta[i]
            ciudad_siguiente = self.ruta[(i + 1) % n_ciudades]  # Ciclo cerrado
            distancia_total += matriz_distancias[ciudad_actual, ciudad_siguiente]
        return distancia_total
    
    def mejorar_2opt(self, k=100):
       n = self.n_ciudades
       global num_llamadas_objetivo
       n_llamadas_ojetivo = int((2/n))*k # equivalencia a llamadas de la FO
       for _ in range(k):
            # Selección de dos índices aleatorios distintos
            i, j = random.sample(range(n), 2)
            
            # Asegurarse de que i < j intercambiándolos si es necesario
            if i > j:
                i, j = j, i

            # Validar valores extremos y evitar índices consecutivos
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # Saltar esta iteración
            
            # Obtener nodos antes y después de los puntos seleccionados
            ciudad1_prev = self.ruta[i - 1] if i > 0 else self.ruta[-1]
            ciudad1 = self.ruta[i]
            ciudad2 = self.ruta[j]
            ciudad2_next = self.ruta[(j + 1) % n]

            # Calcular costos de las aristas a remover y agregar
            costo_actual = (
                matriz_distancias[ciudad1_prev, ciudad1] +
                matriz_distancias[ciudad2, ciudad2_next]
            )
            costo_nuevo = (
                matriz_distancias[ciudad1_prev, ciudad2] +
                matriz_distancias[ciudad1, ciudad2_next]
            )
            delta_fitness = costo_nuevo - costo_actual

            # Aplicar el cambio si es una mejora
            if delta_fitness < 0:
                # Invertir la sección entre i y j
                self.ruta[i:j + 1] = reversed(self.ruta[i:j + 1])
                self.fitness += delta_fitness
                
    def mejorar_2opt2(self, k=100):
        n = len(self.ruta)
        for _ in range(k):
            # Seleccionar dos índices aleatorios para el intercambio 2-opt
            i, j = sorted(random.sample(range(n), 2))
            
            # Calcular el cambio en el fitness basado solo en las aristas afectadas
            ciudad_antes_i = self.ruta[i - 1]
            ciudad_i = self.ruta[i]
            ciudad_j = self.ruta[j]
            ciudad_despues_j = self.ruta[(j + 1) % n]

            # Restar las distancias de las aristas que vamos a remover
            cambio_fitness = -matriz_distancias[ciudad_antes_i, ciudad_i]
            cambio_fitness -= matriz_distancias[ciudad_j, ciudad_despues_j]
            
            # Sumar las distancias de las nuevas aristas que se crean con el intercambio
            cambio_fitness += matriz_distancias[ciudad_antes_i, ciudad_j]
            cambio_fitness += matriz_distancias[ciudad_i, ciudad_despues_j]

            # Solo actualizar la ruta si se encuentra una mejora
            if cambio_fitness < 0:
                # Invertir la sección entre i y j para aplicar el 2-opt
                self.ruta[i:j+1] = reversed(self.ruta[i:j+1])
                # Actualizar el fitness con el cambio calculado
                self.fitness += cambio_fitness
        

# Clase que representa una población de individuos
class Poblacion:
    def __init__(self, tamaño, n_ciudades, limite_generaciones, p_elite, p_diversos, tamaño_torneo,
                 prob_mutacion, nivel_mutacion, p_ls, profundidad_ls, frecuentia_ls):
        self.tamaño = tamaño
        self.n_ciudades = n_ciudades
        self.individuos = self.inicializar(tamaño, n_ciudades)
        self.ordenado = False  # Indica si los individuos están ordenados por fitness

        # Nuevos atributos
        self.limite_generaciones = limite_generaciones
        self.p_elite = p_elite
        self.p_diversos = p_diversos
        self.tamaño_torneo = tamaño_torneo
        self.prob_mutacion = prob_mutacion
        self.nivel_mutacion = nivel_mutacion
        self.p_ls = p_ls
        self.profundidad_ls = profundidad_ls
        self.frecuentia_ls = frecuentia_ls

    def inicializar(self, tamaño, n):
        poblacion = []
        for _ in range(tamaño):
            ruta = np.random.permutation(n).tolist()  # Generar una ruta aleatoria
            poblacion.append(Individuo(ruta))
        return poblacion
    
    def inicializar_goloso(self, matriz_distancias):
        """Inicializa la población usando soluciones golosas basadas en aristas aleatorias."""
        n_ciudades = matriz_distancias.shape[0]
        self.individuos = []

        while len(self.individuos) < self.tamaño:
            # Seleccionar una arista aleatoria
            ciudad1, ciudad2 = random.sample(range(n_ciudades), 2)
            
            # Generar una solución golosa basada en esta arista
            solucion_golosa = self.generar_solucion_golosa_desde_arista(matriz_distancias, ciudad1, ciudad2)
            
            # Crear el individuo y agregarlo a la población
            self.individuos.append(Individuo(solucion_golosa))

    def generar_solucion_golosa_desde_arista(self, matriz_distancias, ciudad1, ciudad2):
        """Genera una solución golosa iniciando con una arista específica."""
        n_ciudades = matriz_distancias.shape[0]
        solucion_golosa = [ciudad1, ciudad2]
        ciudades_restantes = set(range(n_ciudades)) - {ciudad1, ciudad2}
        
        while ciudades_restantes:
            ultima_ciudad_inicio = solucion_golosa[0]
            ultima_ciudad_final = solucion_golosa[-1]
            
            # Buscar la ciudad más cercana al inicio o al final del recorrido
            ciudad_mas_cercana, extremo = min(
                [(ciudad, extremo) for ciudad in ciudades_restantes for extremo in (0, 1)],
                key=lambda ciudad_extremo: matriz_distancias[
                    ultima_ciudad_inicio if ciudad_extremo[1] == 0 else ultima_ciudad_final,
                    ciudad_extremo[0]
                ]
            )
            
            # Decidir si agregarla al inicio o al final
            if extremo == 0:
                solucion_golosa.insert(0, ciudad_mas_cercana)
            else:
                solucion_golosa.append(ciudad_mas_cercana)
            
            ciudades_restantes.remove(ciudad_mas_cercana)
        
        return solucion_golosa

    def evaluar(self):
        for individuo in self.individuos:
            individuo.fitness = individuo.evaluar_fitness_numpy()

    def ordenar_individuos(self):
        """Ordena los individuos en la población por su fitness (ascendente)."""
        if not self.ordenado:
            self.individuos.sort(key=lambda x: x.fitness)
            self.ordenado = True


    def obtener_mejores(self, n):
        """Devuelve los n mejores individuos. Supone que ya están ordenados."""
        n = int(n * self.tamaño)
        if n < 1:
            n = 1
        if self.ordenado == True:
            return self.individuos[:n]
        
        self.ordenar_individuos()
        return self.individuos[:n]

    def mejor_individuo(self):
        """Devuelve el mejor individuo (menor fitness)."""
        if self.ordenado == True:
            return self.individuos[0]
        
        self.ordenar_individuos()
        return self.individuos[0]

    def seleccion_torneo(self, n, k=10):
        # Seleccionar n individuos
        n = int(n * self.tamaño)
        k = int(k * self.tamaño)
        if k < 2:
            k = 2
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
        self.ordenado = False
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

    def mutacion1(self, prob_mutacion=0.01, n_swaps=0.1):
        n_swaps = int(n_swaps * self.n_ciudades)
        n_mutar = int(len(self.individuos) * prob_mutacion)
        for i in random.sample(range(len(self.individuos)), n_mutar):
            hijo = self.individuos[i]
            for _ in range(n_swaps):
                idx1, idx2 = random.sample(range(len(hijo.ruta)), 2)
                # Intercambiamos las ciudades
                hijo.ruta[idx1], hijo.ruta[idx2] = hijo.ruta[idx2], hijo.ruta[idx1]
            hijo.fitness = hijo.evaluar_fitness_numpy()  # Actualizamos el fitness
        self.ordenado = False

    def mutacion(self, prob_mutacion=0.01, nivel_mutacion=0.3):
        n_mutar = int(len(self.individuos) * prob_mutacion)  # Número de individuos a mutar
        nivel_mutacion = min(nivel_mutacion, 0.3)  # Asegurarse de que no supere el 30%
        
        for i in random.sample(range(len(self.individuos)), n_mutar):
            hijo = self.individuos[i]
            n = len(hijo.ruta)
            
            # Determinar el número de intercambios basado en el nivel de mutación
            n_swaps = max(1, int(n * nivel_mutacion))  # Al menos 1 intercambio

            # Realizar los intercambios en la ruta
            for _ in range(n_swaps):
                idx1, idx2 = np.random.choice(range(n), size=2, replace=False)  # Selección de índices únicos
                hijo.ruta[idx1], hijo.ruta[idx2] = hijo.ruta[idx2], hijo.ruta[idx1]  # Intercambio
            
            # Actualizar el fitness del individuo después de la mutación
            hijo.fitness = hijo.evaluar_fitness_numpy()
            hijo.mejorar_2opt(1000)
        self.ordenado = False

    def aplicar_busqueda_local(self, k=0.1, profundida=100):
        k = int(k * self.tamaño)
        mejorados = random.sample(self.individuos, k)
        for individuo in mejorados:
            individuo.fitness = individuo.evaluar_fitness_numpy()
            individuo.mejorar_2opt(profundida)  # Aplicar el 2-opt en cada individuo
        self.ordenado = False

    def generar_rutas_aleatorias(self, n):
        n = int(self.tamaño * n)
        rutas = []
        for _ in range(n):
            ruta = np.random.permutation(self.n_ciudades).tolist()
            rutas.append(Individuo(ruta))
        return rutas

    def imprimir_poblacion(self):
        """Imprime todos los individuos de la población con sus rutas y valores de fitness."""
        print(f"Población de tamaño {len(self.individuos)}:")
        for i, individuo in enumerate(self.individuos, start=1):
            print(f"Individuo {i}: Ruta = {individuo.ruta}, Fitness = {individuo.fitness}, Len = {len(individuo.ruta)}")

    def extraer_individuos(self, n):
        """Extrae `n` mejores individuos y eliminar los n peores de la población."""
        self.ordenar_individuos()
        #print("extraer")
        #print([ind.fitness for ind in self.individuos])
        #extraidos = self.individuos[-n:]
        #self.individuos = self.individuos[:-n]
        #self.ordenar_individuos()
        extraidos = self.individuos[:n]
        self.individuos = self.individuos[n:]
        #print([ind.fitness for ind in self.individuos])
        return extraidos
    
    def reintegrar_individuos(self, individuos):
        """Añade individuos a la población y la reequilibra."""
        #print("regresar")
        #print([ind.fitness for ind in self.individuos])
        self.individuos.extend(individuos)
        #print([ind.fitness for ind in self.individuos])
        self.ordenado = False
    
    def evolucionar(self):
        """Evoluciona la población por el número de generaciones especificado."""
        p_padres = 1 - self.p_diversos - self.p_elite
        for i in range(self.limite_generaciones):
            elite = self.obtener_mejores(self.p_elite)
            diversos = self.generar_rutas_aleatorias(self.p_diversos)
            padres = self.seleccion_torneo(p_padres, k=self.tamaño_torneo)
            padres.extend(diversos)
            hijos = self.recombinacion(padres)  # Cruzar la población 

            self.individuos = hijos
            self.mutacion(self.prob_mutacion, self.nivel_mutacion)
            self.individuos.extend(elite) 
            # Aplicar búsqueda local
            if (i+1) % self.frecuentia_ls == 0:
                self.aplicar_busqueda_local(self.p_ls, self.profundidad_ls)
                #elite[1].mejorar_2opt(15000)
            # Guardar el fitness de la nueva población
            #print(len(self.individuos))
            #elite = self.obtener_mejores(0.5)
            #fitness_matrix.append([ind.fitness for ind in elite])

        self.individuos.extend(elite)

def memetico2(matriz, num_ciclos_principal=100, num_generaciones=10, tam_pobla=100, 
              limit_llamadas_funcion_objetivo=None, p_elite=0.05, p_diversos=0.05, tamaño_torneo=0.1,
              prob_mutacion=0.01, nivel_mutacion=0.1, p_ls=0.1, profundidad_ls=1000, frecuentia_ls=10):
    global matriz_distancias
    global fitness_matrix 
    global num_llamadas_objetivo
    matriz_distancias = matriz
    n = len(matriz)  # Número de ciudades

    # Inicializar las tres subpoblaciones
    # Asumiendo 40 de poblacion
    pobla1 = Poblacion(
        tamaño= int(tam_pobla // 3), 
        n_ciudades=n,
        limite_generaciones=num_generaciones,
        p_elite=p_elite,
        p_diversos=p_diversos,
        tamaño_torneo=tamaño_torneo,
        prob_mutacion=prob_mutacion,
        nivel_mutacion=nivel_mutacion,
        p_ls=p_ls,
        profundidad_ls=profundidad_ls,
        frecuentia_ls=frecuentia_ls
    )

    pobla2 = Poblacion(
        tamaño=int(tam_pobla // 3),
        n_ciudades=n,
        limite_generaciones=num_generaciones,
        p_elite=p_elite,
        p_diversos=p_diversos,
        tamaño_torneo=tamaño_torneo,
        prob_mutacion=prob_mutacion,
        nivel_mutacion=nivel_mutacion,
        p_ls=p_ls,
        profundidad_ls=profundidad_ls,
        frecuentia_ls=frecuentia_ls
    )
    #pobla2.inicializar_goloso(matriz_distancias)
    #pobla2.mutacion(1,0.15)

    pobla3 = Poblacion(
        tamaño=int(tam_pobla // 3),
        n_ciudades=n,
        limite_generaciones=num_generaciones,
        p_elite=p_elite,
        p_diversos=p_diversos,
        tamaño_torneo=tamaño_torneo,
        prob_mutacion=prob_mutacion,
        nivel_mutacion=nivel_mutacion,
        p_ls=p_ls,
        profundidad_ls=profundidad_ls,
        frecuentia_ls=frecuentia_ls
    )
    i = 0 
    while num_llamadas_objetivo < limit_llamadas_funcion_objetivo:
        i = i + 1

        # Evolución de cada subpoblación
        pobla1.evolucionar()
        pobla2.evolucionar()
        pobla3.evolucionar()

        # Extraer elites de las subpoblaciones
        mejor1 = pobla1.extraer_individuos(5)
        mejor2 = pobla2.extraer_individuos(5)
        mejor3 = pobla3.extraer_individuos(5)

        #mejor1[2].mejorar_2opt(k=100)
        #mejor2[2].mejorar_2opt(k=100)
        #mejor3[2].mejorar_2opt(k=100)

        # Reiniciar subpoblaciones con la elite combinada y otras soluciones
        pobla1.reintegrar_individuos(mejor3)
        pobla2.reintegrar_individuos(mejor1)
        pobla3.reintegrar_individuos(mejor2)



    # Combinar las subpoblaciones al final
    poblacion_final = pobla1.individuos + pobla2.individuos + pobla3.individuos

    mejor_individuo = min(poblacion_final, key=lambda x: x.fitness)
    return mejor_individuo

# Función para leer la matriz de distancias desde un archivo
def leer_matriz(file):
    return np.loadtxt(file, dtype=int)

# Función para guardar la matriz de fitness en un archivo CSV
def guardar_fitness_en_csv(nombre_archivo):
    global fitness_matrix
    df = pd.DataFrame(fitness_matrix)  # Convertir la matriz a un DataFrame de pandas
    df.to_csv(nombre_archivo, index=False, header=False)  # Guardar en CSV sin índice ni cabecera

if __name__ == "__main__":
    # Configuración de argparse para recibir argumentos desde la línea de comandos
    parser = argparse.ArgumentParser(description="Optimización del TSP usando un Algoritmo Memético")
    parser.add_argument("--ruta", type=str, default='Instancias/matrix-uy734.txt', help="Ruta al archivo de matriz de distancias")
    parser.add_argument("--n_generaciones", type=int, default=300, help="Número de generaciones")
    parser.add_argument("--tam_pobla", type=int, default=120, help="Tamaño de la población")
    parser.add_argument("--limit_llamadas_funcion_objetivo", type=int, default=300000, help="Número de llamadas a la función objetivo límite")
    parser.add_argument("--p_elite", type=float, default=0.1, help="Proporción de individuos élite")
    parser.add_argument("--p_diversos", type=float, default=0.15, help="Proporción de individuos diversos")
    parser.add_argument("--tamaño_torneo", type=float, default=0.01, help="Tamaño del torneo de selección")
    parser.add_argument("--prob_mutacion", type=float, default=0.05, help="Probabilidad de mutación")
    parser.add_argument("--nivel_mutacion", type=float, default=0.3, help="Nivel de mutación")
    parser.add_argument("--p_ls", type=float, default=0.2, help="Proporción de individuos en los que se aplica búsqueda local")
    parser.add_argument("--profundidad_ls", type=int, default=200, help="Profundidad de búsqueda local")
    parser.add_argument("--frecuentia_ls", type=int, default=5, help="Frecuencia de aplicación de la búsqueda local")
    parser.add_argument("--output", type=str, default="fitness_generaciones.csv", help="Nombre del archivo de salida para la matriz de fitness")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla para la aleatoriedad")
    parser.add_argument("--optimo", type=float, default=None, help="Valor óptimo conocido para calcular el error relativo")

    args = parser.parse_args()

    # Configurar la semilla de aleatoriedad
    if args.semilla is not None:
        np.random.seed(args.semilla)
        random.seed(args.semilla)

    # Leer la matriz de distancias
    matriz_distancias = leer_matriz(args.ruta)

    # Medir el tiempo de ejecución
    start_time = time.time()

    # Ejecutar el algoritmo memético con los parámetros proporcionados

    mejor_solucion = memetico2(
        matriz=matriz_distancias,
        num_generaciones=args.n_generaciones,
        num_ciclos_principal= 100,
        tam_pobla=args.tam_pobla,
        limit_llamadas_funcion_objetivo=args.limit_llamadas_funcion_objetivo,
        p_elite=args.p_elite,
        p_diversos=args.p_diversos,
        tamaño_torneo=args.tamaño_torneo,
        prob_mutacion=args.prob_mutacion,
        nivel_mutacion=args.nivel_mutacion,
        p_ls=args.p_ls,
        profundidad_ls=args.profundidad_ls,
        frecuentia_ls=args.frecuentia_ls
    )


    # Guardar la matriz de fitness en el archivo especificado
    guardar_fitness_en_csv(args.output)

    # Imprimir la mejor solución, su fitness y el error relativo si corresponde
    end_time = time.time()
    tiempo_total = end_time - start_time

    error_relativo = abs(mejor_solucion.fitness - 79114)/ 79114 * 100
    #print(f"{error_relativo:.4f}")

    # Imprimir la mejor solución y su fitness
    #print("Mejor ruta:", mejor_solucion.ruta)
    #print("Mejor fitness (distancia total):", mejor_solucion.fitness)

    # Imprimir el tiempo de ejecución para irace
    #print(mejor_solucion.fitness , end_time-start_time)
    
    print(f"Len: {len(mejor_solucion.ruta)}")
    print(f"Mejor fitness encontrado: {mejor_solucion.fitness}")
    print("Mejor ruta:", mejor_solucion.ruta)
    print(f"Tiempo de ejecución: {tiempo_total:.2f} segundos")

    if args.optimo is not None:
        error_relativo = abs(mejor_solucion.fitness - args.optimo) / args.optimo * 100
        print(f"Error relativo: {error_relativo:.2f}%")


# python3 memetico2.py --ruta Instancias/matrix-wi29.txt --optimo 27603
# python3 memetico2.py --ruta Instancias/matrix-dj38.txt --optimo 6656
# python3 memetico2.py --ruta Instancias/matrix-qa194.txt --optimo 9352
# python3 memetico2.py --ruta Instancias/matrix-lu980.txt --optimo 11340
# python3 memetico2.py --ruta Instancias/matrix-zi929.txt --optimo 95345
# python3 memetico2.py --ruta Instancias/matrix-uy734.txt --optimo 79114
# python3 memetico2.py --ruta Instancias/matrix-rw1621.txt --optimo 26051