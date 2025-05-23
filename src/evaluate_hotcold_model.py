import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re
from sklearn.metrics import classification_report
import numpy as np

# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-hotcold-batched.txt"
SAMPLE_SIZE = 1_000_000
THRESHOLD = 1000  # bloques para definir "hot"

# === Ordenar archivos
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Cargar muestras aleatorias
samples = []

for path in files:
    try:
        df = pd.read_parquet(path)

        # Etiquetar hot = 1 si gasto en corto plazo
        df['hot'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(int)

        # Muestra parcial aleatoria
        frac = SAMPLE_SIZE / (len(files) * len(df))
        frac = min(frac, 0.1)
        sample = df.sample(frac=frac, random_state=42)
        samples.append(sample)
    except Exception as e:
        print(f"❌ Error en {path}: {e}")

df_eval = pd.concat(samples, ignore_index=True)
if len(df_eval) > SAMPLE_SIZE:
    df_eval = df_eval.sample(n=SAMPLE_SIZE, random_state=42)

print(f"✅ Total muestras aleatorias cargadas: {len(df_eval):,}")

# === Cargar modelo
model = lgb.Booster(model_file=MODEL_PATH)
print(f"✅ Modelo cargado desde: {MODEL_PATH}")

# === Preparar features y target
features = [
    'value', 'locking_script_size', 'unlocking_script_size',
    'tx_coinbase', 'op_return', 'epoch', 'creation_block'
]
X = df_eval[features]
y_true = df_eval['hot']

# === Predicción y evaluación
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 Evaluación de clasificación (hot/cold):")
print(classification_report(y_true, y_pred, digits=4))

# === Opcional: ver distribución de probabilidades
print("\n🔎 Resumen de probabilidades predichas:")
print(pd.Series(y_pred_prob).describe())
print("Ejemplos con prob > 0.5:", np.sum(y_pred_prob > 0.5))
print("Ejemplos con prob > 0.2:", np.sum(y_pred_prob > 0.2))
