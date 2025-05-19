import pandas as pd
import glob
import re
import os

# Directorios
input_dir = "/home/fernando/dev/utxo-experiments/output/"
backup_dir = "/home/fernando/dev/utxo-experiments/backup/"
os.makedirs(backup_dir, exist_ok=True)
csv_files = glob.glob(f"{input_dir}utxo-history-*.csv")

# Ordenar archivos correctamente por número
csv_files.sort(key=lambda x: int(re.search(r'utxo-history-(\d+)\.csv', x).group(1)))

def fix_csv_file(file):
    print(f"Processing file: {file}")

    # Crear respaldo del archivo original
    backup_file = os.path.join(backup_dir, os.path.basename(file))
    os.rename(file, backup_file)
    
    # Leer el archivo y determinar si contiene Unspent
    with open(backup_file, "r") as f:
        lines = f.readlines()
    
    corrected_lines = [lines[0].strip()]  # Header line
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) == 5:
            # Verificar si es una fila Unspent
            if parts[2] == "Unspent":
                # Cambiar el orden de las columnas para Unspent
                corrected_line = f"{parts[0]},{parts[2]},{parts[1]},{parts[3]},-\n"
                corrected_lines.append(corrected_line)
            else:
                # Mantener las filas normales sin cambio
                corrected_lines.append(line)
    
    # Guardar el archivo corregido
    with open(file, "w") as f:
        f.writelines(corrected_lines)
    print(f"File corrected: {file}")

# Corregir los archivos relevantes (a partir del archivo 248)
for file in csv_files:
    file_number = int(re.search(r'utxo-history-(\d+)\.csv', file).group(1))
    if file_number >= 248:
        fix_csv_file(file)

print("All files corrected.")
