import subprocess
import csv
import concurrent.futures
import time

# ============================================================================== # 
# =========== Este programa permite ejecutar n veces la metaheuristica ========= #
# ============================================================================== # 

def ejecutar_comando():
    comando = ["python3", "ILS.py", "--ruta", "../Instancias/matrix-rw1621.txt", "--dir_log_out", "logs/"]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    return int(resultado.stdout.strip())

def escribir_resultado(file, ejecucion, resultado):
    with open(file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ejecucion, resultado])

def main():
    num_ejecuciones = 31
    archivo_csv = 'resultados.csv'
    tiempo_retraso = 2  # Retraso en segundos

    # Creamos el archivo CSV y escribimos la cabecera
    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ejecución", "Resultado"])

    # Usamos ThreadPoolExecutor para ejecutar en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_ejecuciones) as executor:
        futures = {}
        
        for i in range(num_ejecuciones):
            # Programamos la ejecución con un retraso
            future = executor.submit(ejecutar_comando)
            futures[future] = i + 1
            time.sleep(tiempo_retraso)  # Introducimos un retraso

        for future in concurrent.futures.as_completed(futures):
            ejecucion = futures[future]
            try:
                resultado = future.result()
                escribir_resultado(archivo_csv, ejecucion, resultado)
                print(f"Ejecución {ejecucion} completada con resultado: {resultado}")
            except Exception as e:
                print(f"Ocurrió un error en la ejecución {ejecucion}: {e}")

if __name__ == "__main__":
    main()
