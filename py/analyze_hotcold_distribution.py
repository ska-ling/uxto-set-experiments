import pandas as pd
import glob
import re
from pathlib import Path

# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
THRESHOLD = 1000  # bloques

# === Ordenar archivos por número
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Contadores globales
hot_count = 0
cold_count = 0

for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)
        df['hot'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(int)
        counts = df['hot'].value_counts()
        hot_count += counts.get(1, 0)
        cold_count += counts.get(0, 0)

        print("\n🔥 Distribución parcial de clases 'hot':")
        print(f"Cold (0): {cold_count:,}")
        print(f"Hot  (1): {hot_count:,}")

    except Exception as e:
        print(f"❌ Error en {path}: {e}")

# === Mostrar resultado
print("\n🔥 Distribución global de clases 'hot':")
print(f"Cold (0): {cold_count:,}")
print(f"Hot  (1): {hot_count:,}")
