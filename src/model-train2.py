import pandas as pd
import glob
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib   
import re

def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.csv$', path)
    return int(match.group(1)) if match else -1


# === Configuración ===
# SPENT_FILES = sorted(glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-*.csv"))[:353]


SPENT_FILES = sorted(
    glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-*.csv"),
    key=extract_index
)[:353]

print(f"Archivos SPENT: {SPENT_FILES}")


CHUNK_SIZE = 1_000_000  # líneas por batch

# === Modelo y escalador ===
model = SGDRegressor(max_iter=5, tol=1e-3, random_state=42)
scaler = StandardScaler()

X_total = []
y_total = []

for path in SPENT_FILES:
    print(f"Procesando: {path}")
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        rows = []
        for index, row in chunk.iterrows():
            try:
                creation_block = int(row['creation_block'])
                spent_block = int(row['spent_block'])
                lifetime = spent_block - creation_block
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = int(row['unlocking_script_size']) if row['unlocking_script_size'] else -1
                tx_coinbase = row['tx_coinbase'].strip().lower() == 'true'
                op_return = row['op_return'].strip().lower() == 'true'
                rows.append([
                    creation_block, value, locking_script_size,
                    unlocking_script_size, int(tx_coinbase), int(op_return), lifetime
                ])
            except Exception as e:
                continue

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=[
            'creation_block', 'value', 'locking_script_size',
            'unlocking_script_size', 'tx_coinbase', 'op_return', 'lifetime'
        ])

        X = df.drop('lifetime', axis=1)
        y = df['lifetime']

        # Acumular para evaluación
        if len(X_total) < 1_000_000:
            X_total.append(X)
            y_total.append(y)

        # Escalar
        X_scaled = scaler.fit_transform(X)

        # Entrenar incrementalmente
        model.partial_fit(X_scaled, y)

# === Evaluación final ===
X_eval = pd.concat(X_total)
y_eval = pd.concat(y_total)
X_eval_scaled = scaler.transform(X_eval)
y_pred = model.predict(X_eval_scaled)

print(f"MAE final sobre subset: {mean_absolute_error(y_eval, y_pred):.2f} bloques")

#Guardar el modelo
joblib.dump(model, 'sgd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
Cargar el modelo
model = joblib.load('sgd_model.pkl')
scaler = joblib.load('scaler.pkl')
y_pred = model.predict(X_eval_scaled)
print(f"MAE final sobre subset: {mean_absolute_error(y_eval, y_pred):.2f} bloques")
