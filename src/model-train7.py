import pandas as pd
import glob
import lightgbm as lgb
from pathlib import Path
import re
from sklearn.metrics import classification_report

# === CONFIG ===
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
THRESHOLD = 1000
EVAL_ROWS = 1_000_000
MODEL_PATH = "modelo-hotcold-batched.txt"

# === Utilidad para ordenar por índice numérico
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

# === Cargar archivos ordenados
files = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)

# === Parámetros LightGBM
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 8,
    "verbosity": -1
}

model = None
X_eval_list = []
y_eval_list = []
rows_seen = 0

# === Entrenamiento por archivo
for path in files:
    print(f"Procesando: {path}")
    try:
        df = pd.read_parquet(path)
        print(df.columns.tolist())

        df = df.copy()
        df['hot'] = ((df['duration'] <= THRESHOLD) & df['event']).astype(int)

        features = ['value', 'locking_script_size', 'unlocking_script_size',
                    'tx_coinbase', 'op_return', 'epoch']
        X = df[features]
        y = df['hot']

        # Evaluación acumulada
        if rows_seen < EVAL_ROWS:
            X_eval_list.append(X)
            y_eval_list.append(y)
            rows_seen += len(X)

        # Entrenar con este batch
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_set, num_boost_round=20, init_model=model, keep_training_booster=True)

    except Exception as e:
        print(f"❌ Error con {path}: {e}")

# === Evaluación final
print("\nEvaluando sobre subset de evaluación...")
X_eval = pd.concat(X_eval_list)
y_eval = pd.concat(y_eval_list)

y_pred = model.predict(X_eval)
y_pred_class = (y_pred > 0.5).astype(int)

print(classification_report(y_eval, y_pred_class, digits=4))

# === Guardar modelo
model.save_model(MODEL_PATH)
print(f"✅ Modelo guardado en: {MODEL_PATH}")
