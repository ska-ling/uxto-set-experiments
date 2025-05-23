import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re
from sklearn.metrics import classification_report
import numpy as np


# === CONFIGURACIÓN ===
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
THRESHOLD = 1000  # bloques para considerar "hot"
EVAL_ROWS = 1_000_000  # cuántas filas usar para evaluación final
MODEL_PATH = "modelo-hotcold-batched.txt"

# === Ordenar archivos por índice numérico
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === LightGBM config
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 60,
    "verbosity": -1,
    "scale_pos_weight": 0.4793 # 319_023_419 / 665_471_470 ≈ 0.4793 👈 ajustar según la distribución
}

model = None
X_eval_list = []
y_eval_list = []
rows_seen = 0

# === Entrenamiento incremental por archivo
for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)

        # ✅ Target: si fue gastado y en corto plazo (<= threshold)
        df['hot'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(int)

        # ✅ Features: solo campos disponibles en tiempo real
        features = [
            'value',
            'locking_script_size',
            'unlocking_script_size',
            'tx_coinbase',
            'op_return',
            'epoch',
            'creation_block'
        ]
        X = df[features]
        y = df['hot']

        # ✅ Acumular evaluación (opcional)
        if rows_seen < EVAL_ROWS:
            X_eval_list.append(X)
            y_eval_list.append(y)
            rows_seen += len(X)

        # ✅ Entrenamiento incremental
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=20,
            init_model=model,
            keep_training_booster=True
        )

    except Exception as e:
        print(f"❌ Error con {path}: {e}")

# === Evaluación final
print("\n📊 Evaluando sobre subset de evaluación...")
X_eval = pd.concat(X_eval_list)
y_eval = pd.concat(y_eval_list)

y_pred = model.predict(X_eval)
y_pred_class = (y_pred > 0.5).astype(int)

print(classification_report(y_eval, y_pred_class, digits=4))

print("\n🔎 Resumen de probabilidades predichas:")
print(pd.Series(y_pred).describe())
print("Ejemplos con probabilidad > 0.1:", np.sum(y_pred > 0.1))
print("Ejemplos con probabilidad > 0.5:", np.sum(y_pred > 0.5))


# === Guardar modelo
model.save_model(MODEL_PATH)
print(f"✅ Modelo entrenado guardado en: {MODEL_PATH}")
