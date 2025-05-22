import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === Rutas ===
SPENT_FILES = sorted(glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-[0-3][0-9][0-9].csv"))[:353]
UNSPENT_FILES = sorted(glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-[3][5-9][3-9].csv"))

print(f"Archivos gastados: {SPENT_FILES}")
print(f"Archivos no gastados: {UNSPENT_FILES}")

def load_csvs(file_paths, spent_flag):
    rows = []
    for path in file_paths:
        print(f"Procesando archivo: {path}")
        df = pd.read_csv(path, dtype=str)
        for index, row in df.iterrows():
            try:
                creation_block = int(row['creation_block'])
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = int(row['unlocking_script_size']) if row['unlocking_script_size'] else -1
                tx_coinbase = row['tx_coinbase'].strip().lower() == 'true'
                op_return = row['op_return'].strip().lower() == 'true'
                spent = spent_flag
                rows.append([
                    creation_block, value, locking_script_size,
                    unlocking_script_size, tx_coinbase, op_return, spent
                ])
            except Exception as e:
                print(f"Error en archivo {path}, línea {index}: {e}")
    return rows

# Cargar datos
spent_data = load_csvs(SPENT_FILES, 1)
unspent_data = load_csvs(UNSPENT_FILES, 0)

# Combinar en un solo DataFrame
all_data = spent_data + unspent_data
df = pd.DataFrame(all_data, columns=[
    'creation_block', 'value', 'locking_script_size',
    'unlocking_script_size', 'tx_coinbase', 'op_return', 'spent'
])

# === Entrenar modelo ===
X = df.drop('spent', axis=1)
y = df['spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
