import pandas as pd
import glob
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re

# === CONFIGURACIÓN ===
THRESHOLD = 1000
DATA_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
OUT_MODEL = "modelo-hotcold.txt"

# === Definir epoch como feature
def assign_epoch(block_height):
    if block_height < 100_000:
        return 0
    elif block_height < 300_000:
        return 1
    elif block_height < 500_000:
        return 2
    elif block_height < 600_000:
        return 3
    elif block_height < 700_000:
        return 4
    else:
        return 5

# === Cargar todos los .parquet
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

all_parquets = sorted(glob.glob(str(DATA_DIR / "utxo-history-*.parquet")), key=extract_index)

dfs = []
print("Cargando archivos...")
for path in all_parquets:
    print(f" -> {path}")
    try:
        df = pd.read_parquet(path)
        dfs.append(df)
    except Exception as e:
        print(f"❌ Error: {path}: {e}")

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total UTXOs cargados: {len(df_all)}")

# === Calcular columna epoch
df_all['epoch'] = df_all['duration'].astype(int) + 1  # asegurar tipo
df_all['creation_block'] = 789_999 - df_all['duration']  # recrear creación
df_all['epoch'] = df_all['creation_block'].apply(assign_epoch)

# === Calcular columna target 'hot'
df_all['hot'] = ((df_all['duration'] <= THRESHOLD) & df_all['event']).astype(int)

# === Features y Target
features = ['value', 'locking_script_size', 'unlocking_script_size', 'tx_coinbase', 'op_return', 'epoch']
X = df_all[features]
y = df_all['hot']

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Entrenar modelo LightGBM
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "num_threads": 8,
    "verbosity": -1
}

train_set = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_set, num_boost_round=100)

# === Evaluación
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

print("=== Evaluación ===")
print(classification_report(y_test, y_pred_class, digits=4))

# === Guardar modelo
model.save_model(OUT_MODEL)
print(f"✅ Modelo guardado en: {OUT_MODEL}")
