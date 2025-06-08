import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re
import numpy as np
import random

# === Configuración
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
MODEL_PATH = "modelo-log-duration.txt"
THRESHOLD = 1000              # bloques considerados "gasto pronto"
MIN_AGE_UNSPENT = THRESHOLD   # mínima edad para usar un Unspent
EPSILON = 0.5                 # valor agregado a Unspent para diferenciar censura

# === Recolectar y barajar archivos
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)
random.seed(42)
random.shuffle(files)

# === Parámetros LightGBM
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 60,
    "verbosity": -1
}

model = None

# === Entrenamiento incremental
for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)

        # === Filtrar UTXOs poco confiables
        df = df[df['duration'] > 0]
        df = df[~((df['event'] == False) & (df['duration'] < MIN_AGE_UNSPENT))]
        if df.empty:
            print("⚠️  Archivo vacío tras filtrado, se salta.")
            continue

        # === Target: log(duration) o log(duration)+ε si Unspent
        df['target'] = np.log(df['duration']) + (~df['event']) * EPSILON

        # === Features disponibles en tiempo real
        features = [
            'value', 'locking_script_size', 'unlocking_script_size',
            'tx_coinbase', 'op_return', 'epoch', 'creation_block'
        ]
        X = df[features]
        y = df['target']

        # === Entrenamiento incremental
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
