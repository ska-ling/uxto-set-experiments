import pandas as pd
import glob
import lightgbm as lgb
import numpy as np
import re
from sklearn.metrics import mean_absolute_error

# === Configuración ===
CSV_DIR = "/home/fernando/dev/utxo-experiments/output"
CHUNK_SIZE = 1_000_000
EVAL_ROWS = 1_000_000

# === Ordenar archivos numéricamente ===
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.csv$', path)
    return int(match.group(1)) if match else -1

SPENT_FILES = sorted(
    glob.glob(f"{CSV_DIR}/utxo-history-*.csv"),
    key=extract_index
)[:353]

# === LightGBM params ===
params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.1,
    "max_depth": 8,
    "verbosity": -1,
    "num_threads": 4
}

model = None
X_eval_total = []
y_eval_total = []
rows_seen = 0

for path in SPENT_FILES:
    print(f"Procesando: {path}")
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        try:
            # Convertir columnas
            chunk['creation_block'] = chunk['creation_block'].astype(int)
            chunk['spent_block'] = chunk['spent_block'].astype(int)
            chunk['value'] = chunk['value'].astype(int)
            chunk['locking_script_size'] = chunk['locking_script_size'].astype(int)
            chunk['unlocking_script_size'] = pd.to_numeric(chunk['unlocking_script_size'], errors='coerce').fillna(-1).astype(int)
            chunk['tx_coinbase'] = chunk['tx_coinbase'].str.lower() == 'true'
            chunk['op_return'] = chunk['op_return'].str.lower() == 'true'
            chunk['lifetime'] = chunk['spent_block'] - chunk['creation_block']
        except Exception as e:
            print(f"Error procesando chunk: {e}")
            continue

        X = chunk[['creation_block', 'value', 'locking_script_size',
                   'unlocking_script_size', 'tx_coinbase', 'op_return']].astype('float32')
        y = chunk['lifetime'].astype('float32')

        # Evaluación sobre primeros 1M
        if rows_seen < EVAL_ROWS:
            X_eval_total.append(X)
            y_eval_total.append(y)
            rows_seen += len(X)

        # Entrenamiento incremental
        train_set = lgb.Dataset(X, label=y, free_raw_data=False)
        model = lgb.train(params, train_set, init_model=model, keep_training_booster=True)

# === Evaluación final ===
print("Evaluando modelo...")
X_eval = pd.concat(X_eval_total)
y_eval = pd.concat(y_eval_total)
y_pred = model.predict(X_eval)

mae = mean_absolute_error(y_eval, y_pred)
print(f"MAE sobre subset de {len(X_eval)} filas: {mae:.2f} bloques")

# === Guardar modelo
model.save_model("modelo-utxo.txt")
print("Modelo guardado en modelo-utxo.txt")
