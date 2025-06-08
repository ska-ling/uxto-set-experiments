import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re
import random

# === Configuración general ===
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-spend-probability.txt"
THRESHOLD = 1000  # bloques considerados como "gasto pronto"
MIN_AGE_UNSPENT = THRESHOLD  # edad mínima que debe tener un Unspent para usarse

# === Recolectar y barajar archivos ===
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)
random.seed(42)
random.shuffle(files)

# === Parámetros de LightGBM
params = {
    "objective": "binary",             # usamos salida en [0, 1] (probabilidad)
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 60,
    "verbosity": -1
}

model = None

# === Entrenamiento incremental por archivo
for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)

        # === Filtrar UTXOs poco confiables
        df = df[df['duration'] > 0]
        df = df[~((df['event'] == False) & (df['duration'] < MIN_AGE_UNSPENT))]

        if df.empty:
            print("⚠️  Archivo vacío después del filtrado, se salta.")
            continue

        # 🎯 Target: probabilidad binaria de gasto pronto
        df['p_spend_soon'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(float)

        # 📥 Features disponibles en tiempo real
        features = [
            'value', 'locking_script_size', 'unlocking_script_size',
            'tx_coinbase', 'op_return', 'epoch', 'creation_block'
        ]
        X = df[features]
        y = df['p_spend_soon']

        # 🧠 Entrenamiento incremental
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=20,
            init_model=model,
            keep_training_booster=True
        )

    except Exception as e:
        print(f"❌ Error procesando {path}: {e}")

# === Guardar modelo final
model.save_model(MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")
