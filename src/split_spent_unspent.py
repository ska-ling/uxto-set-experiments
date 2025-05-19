import os
import pandas as pd
import shutil

# Configuración
input_dir = "/home/fernando/dev/utxo-experiments/output/"
filename = f"{input_dir}utxo-history-248.csv"

# Desplazar los archivos siguientes (249, 250, ...) en +1
print("Shifting following files...")
file_index = 249
while os.path.exists(f"{input_dir}utxo-history-{file_index}.csv"):
    old_filename = f"{input_dir}utxo-history-{file_index}.csv"
    new_filename = f"{input_dir}utxo-history-{file_index + 1}.csv"
    shutil.move(old_filename, new_filename)
    print(f"Renamed {old_filename} to {new_filename}")
    file_index += 1

# Leer el archivo original
print("Reading original file...")
df = pd.read_csv(filename, names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"])

# Separar Spent y Unspent
df_spent = df[df["spent_block"] != "Unspent"]
df_unspent = df[df["spent_block"] == "Unspent"]

# Verificar
print(f"Total Spent: {len(df_spent)}")
print(f"Total Unspent: {len(df_unspent)}")

# Guardar el archivo Spent en el mismo nombre (248)
df_spent.to_csv(f"{input_dir}utxo-history-248.csv", index=False, header=False)
print("Saved Spent transactions in utxo-history-248.csv")

# Guardar el archivo Unspent como 249 (que ya fue desplazado)
df_unspent.to_csv(f"{input_dir}utxo-history-249.csv", index=False, header=False)
print("Saved Unspent transactions in utxo-history-249.csv")

print("Process completed.")
