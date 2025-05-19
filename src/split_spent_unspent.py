import os
import shutil

# Configuración
input_dir = "/home/fernando/dev/utxo-experiments/output/"
filename = f"{input_dir}utxo-history-248.csv"

# Encontrar el archivo máximo
print("Detectando archivo máximo...")
max_index = 248
while os.path.exists(f"{input_dir}utxo-history-{max_index}.csv"):
    max_index += 1
max_index -= 1  # El último existente

print(f"Archivo máximo detectado: utxo-history-{max_index}.csv")

# Desplazar archivos hacia arriba, desde el último hasta 249
print("Desplazando archivos hacia arriba...")
for i in range(max_index, 248, -1):
    old_filename = f"{input_dir}utxo-history-{i}.csv"
    new_filename = f"{input_dir}utxo-history-{i + 1}.csv"
    shutil.move(old_filename, new_filename)
    print(f"Renombrado {old_filename} a {new_filename}")

# Leer y separar Spent y Unspent manualmente
print("Leyendo archivo 248 y separando Spent/Unspent...")
spent_lines = []
unspent_lines = []

with open(filename, 'r') as file:
    header = file.readline()  # Leer la cabecera
    for line in file:
        if ",Unspent," in line:
            unspent_lines.append(line)
        else:
            spent_lines.append(line)

# Verificar
print(f"Total Spent: {len(spent_lines)}")
print(f"Total Unspent: {len(unspent_lines)}")

# Guardar los archivos resultantes
with open(f"{input_dir}utxo-history-248.csv", 'w') as spent_file:
    spent_file.write(header)
    spent_file.writelines(spent_lines)
print("Guardado archivo Spent en utxo-history-248.csv")

with open(f"{input_dir}utxo-history-249.csv", 'w') as unspent_file:
    unspent_file.write(header)
    unspent_file.writelines(unspent_lines)
print("Guardado archivo Unspent en utxo-history-249.csv")

print("Proceso completado.")
