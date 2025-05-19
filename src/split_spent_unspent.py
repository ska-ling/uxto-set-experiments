import os
import pandas as pd
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

# Leer el archivo original (248)
print("Leyendo archivo 248...")
df = pd.read_csv(filename, names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"])

# Separar Spent y Unspent
df_spent = df[df["spent_block"] != "Unspent"]
df_unspent = df[df["spent_block"] == "Unspent"]

# Verificar
print(f"Total Spent: {len(df_spent)}")
print(f"Total Unspent: {len(df_unspent)}")

# Guardar el archivo Spent en el mismo nombre (248)
df_spent.to_csv(f"{input_dir}utxo-history-248.csv", index=False, header=False)
print("Guardado archivo Spent en utxo-history-248.csv")

# Guardar el archivo Unspent como 249 (que ya fue desplazado)
df_unspent.to_csv(f"{input_dir}utxo-history-249.csv", index=False, header=False)
print("Guardado archivo Unspent en utxo-history-249.csv")

print("Proceso completado.")
