import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === Rutas ===
SPENT_FILES = sorted(glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-[0-3][0-9][0-9].csv"))[:353]

def load_spent_csvs(file_paths):
    rows = []
    for path in file_paths:
        df = pd.read_csv(path, dtype=str)
        for index, row in df.iterrows():
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
                    unlocking_script_size, tx_coinbase, op_return, lifetime
                ])
            except Exception as e:
                print(f"Error en archivo {path}, línea {index}, row {row}: {e}")
    return rows

# Cargar datos solo de SPENT
spent_data = load_spent_csvs(SPENT_FILES)

df = pd.DataFrame(spent_data, columns=[
    'creation_block', 'value', 'locking_script_size',
    'unlocking_script_size', 'tx_coinbase', 'op_return', 'lifetime'
])

X = df.drop('lifetime', axis=1)
y = df['lifetime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} bloques")
