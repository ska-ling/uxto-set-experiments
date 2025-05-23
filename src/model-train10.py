import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re

# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-spend-probability.txt"
THRESHOLD = 1000  # bloques considerados como "gasto pronto"

# === Ordenar archivos
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Parámetros de LightGBM para regresión
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 8,
    "verbosity": -1
}

model = None

# === Entrenamiento incremental por archivo
for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)

        # 🎯 Target: probabilidad simulada de gasto pronto
        df['p_spend_soon'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(float)

        # Features disponibles en tiempo real
        features = [
            'value', 'locking_script_size', 'unlocking_script_size',
            'tx_coinbase', 'op_return', 'epoch', 'creation_block'
        ]
        X = df[features]
        y = df['p_spend_soon']

        # Entrenamiento incremental con LightGBM
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

# === Guardar modelo entrenado
model.save_model(MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")
