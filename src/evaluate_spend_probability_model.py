import pandas as pd
import glob
import lightgbm as lgb
import numpy as np
from pathlib import Path
import re
from sklearn.metrics import mean_squared_error
from math import sqrt


# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-spend-probability.txt"
SAMPLE_SIZE = 1_000_000
THRESHOLD = 1000  # para calcular el target real (no afecta al modelo)

# === Ordenar archivos por índice
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Muestreo aleatorio desde los parquet
samples = []

for path in files:
    try:
        df = pd.read_parquet(path)

        # Calcular el target real: gasto pronto o no
        df['p_spend_soon'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(float)

        # Tomar muestra aleatoria parcial
        frac = SAMPLE_SIZE / (len(files) * len(df))
        frac = min(frac, 0.1)  # no tomar más del 10% por archivo
        sample = df.sample(frac=frac, random_state=42)
        samples.append(sample)
    except Exception as e:
        print(f"❌ Error procesando {path}: {e}")

# === Consolidar muestras
df_eval = pd.concat(samples, ignore_index=True)
if len(df_eval) > SAMPLE_SIZE:
    df_eval = df_eval.sample(n=SAMPLE_SIZE, random_state=42)

print(f"✅ Total muestras aleatorias cargadas: {len(df_eval):,}")

# === Cargar modelo
model = lgb.Booster(model_file=MODEL_PATH)
print(f"✅ Modelo cargado desde: {MODEL_PATH}")

# === Features y target
features = [
    'value', 'locking_script_size', 'unlocking_script_size',
    'tx_coinbase', 'op_return', 'epoch', 'creation_block'
]
X = df_eval[features]
y_true = df_eval['p_spend_soon']

# === Predicción y evaluación
y_pred = model.predict(X)

# rmse = mean_squared_error(y_true, y_pred, squared=False)
rmse = sqrt(mean_squared_error(y_true, y_pred))

print(f"\n📊 RMSE sobre muestra aleatoria: {rmse:.5f}")

# === Resumen de las probabilidades predichas
print("\n🔎 Distribución de probabilidades predichas:")
print(pd.Series(y_pred).describe())

# === Guardar histograma (opcional)
import matplotlib.pyplot as plt
plt.hist(y_pred, bins=30, edgecolor='black')
plt.title("Distribución de probabilidad de gasto pronto")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Cantidad de UTXOs")
plt.tight_layout()
plt.savefig("spend_probability_histogram.png", dpi=300)
print("📈 Histograma guardado en spend_probability_histogram.png")
