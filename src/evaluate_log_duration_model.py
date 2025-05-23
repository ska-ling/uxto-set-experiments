import pandas as pd
import glob
import lightgbm as lgb
import numpy as np
from pathlib import Path
import re
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-log-duration.txt"
SAMPLE_SIZE = 1_000_000
THRESHOLD = 1000
MIN_AGE_UNSPENT = THRESHOLD
EPSILON = 0.5  # mismo que en entrenamiento

# === Recolectar archivos ordenados
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Cargar muestras aleatorias
samples = []

for path in files:
    try:
        df = pd.read_parquet(path)
        df = df[df['duration'] > 0]
        df = df[~((df['event'] == False) & (df['duration'] < MIN_AGE_UNSPENT))]
        if df.empty:
            continue
        frac = SAMPLE_SIZE / (len(files) * len(df))
        frac = min(frac, 0.1)
        sample = df.sample(frac=frac, random_state=42)
        samples.append(sample)
    except Exception as e:
        print(f"❌ Error en {path}: {e}")

df_eval = pd.concat(samples, ignore_index=True)
if len(df_eval) > SAMPLE_SIZE:
    df_eval = df_eval.sample(n=SAMPLE_SIZE, random_state=42)

print(f"✅ Total muestras evaluadas: {len(df_eval):,}")

# === Target real: log(duration)
df_eval['target'] = np.log(df_eval['duration']) + (~df_eval['event']) * EPSILON

# === Cargar modelo
model = lgb.Booster(model_file=MODEL_PATH)
print(f"✅ Modelo cargado desde: {MODEL_PATH}")

# === Inferencia
features = [
    'value', 'locking_script_size', 'unlocking_script_size',
    'tx_coinbase', 'op_return', 'epoch', 'creation_block'
]
X = df_eval[features]
y_true = df_eval['target']
y_pred = model.predict(X)

# === Evaluación RMSE (solo en datos reales, no censurados)
mask = df_eval['event']
rmse = mean_squared_error(y_true[mask], y_pred[mask], squared=False)
print(f"\n📊 RMSE (solo Spent): {rmse:.5f}")

# === Estadísticas de duración esperada
expected_duration = np.exp(y_pred)
print("\n🔎 Duración esperada predicha (exp):")
print(pd.Series(expected_duration).describe())

# === Histograma
plt.hist(expected_duration, bins=50, edgecolor='black')
plt.title("Duración esperada predicha")
plt.xlabel("Bloques hasta gasto (esperado)")
plt.ylabel("Cantidad de UTXOs")
plt.tight_layout()
plt.savefig("log_duration_prediction_histogram.png", dpi=300)
print("📈 Histograma guardado en log_duration_prediction_histogram.png")
