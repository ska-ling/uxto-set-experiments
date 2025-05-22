import pandas as pd
import glob
import re
from lifelines import KaplanMeierFitter

FINAL_BLOCK_HEIGHT = 789_999

# === Archivos ===
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.csv$', path)
    return int(match.group(1)) if match else -1

ALL_FILES = sorted(glob.glob("/home/fernando/dev/utxo-experiments/output/utxo-history-*.csv"), key=extract_index)
# SPENT_FILES = ALL_FILES[:353]
# UNSPENT_FILES = ALL_FILES[353:396]
SPENT_FILES = ALL_FILES[:10]
UNSPENT_FILES = ALL_FILES[353:363]

# === Loader ===
def load_spent_data(files):
    rows = []
    for path in files:
        print(f"Procesando (spent): {path}")
        df = pd.read_csv(path, dtype=str)
        for _, row in df.iterrows():
            try:
                creation_block = int(row['creation_block'])
                spent_block = int(row['spent_block'])
                lifetime = spent_block - creation_block
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = int(row['unlocking_script_size']) if row['unlocking_script_size'] else -1
                tx_coinbase = row['tx_coinbase'].strip().lower() == 'true'
                op_return = row['op_return'].strip().lower() == 'true'
                rows.append({
                    'duration': lifetime,
                    'event': True,
                    'value': value,
                    'locking_script_size': locking_script_size,
                    'unlocking_script_size': unlocking_script_size,
                    'tx_coinbase': tx_coinbase,
                    'op_return': op_return
                })
            except:
                continue
    return pd.DataFrame(rows)

def load_unspent_data(files):
    rows = []
    for path in files:
        print(f"Procesando (unspent): {path}")
        df = pd.read_csv(path, dtype=str)
        for _, row in df.iterrows():
            try:
                creation_block = int(row['creation_block'])
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = 0  # siempre falta
                tx_coinbase = row['tx_coinbase'].strip().lower() == 'true'
                op_return = row['op_return'].strip().lower() == 'true'
                lifetime = FINAL_BLOCK_HEIGHT - creation_block
                rows.append({
                    'duration': lifetime,
                    'event': False,
                    'value': value,
                    'locking_script_size': locking_script_size,
                    'unlocking_script_size': unlocking_script_size,
                    'tx_coinbase': tx_coinbase,
                    'op_return': op_return
                })
            except:
                continue
    return pd.DataFrame(rows)

# === Cargar y unir datos ===
print("Cargando datos...")
df_spent = load_spent_data(SPENT_FILES)
df_unspent = load_unspent_data(UNSPENT_FILES)
df_all = pd.concat([df_spent, df_unspent], ignore_index=True)

print(f"Total UTXOs: {len(df_all)}")

# === Modelo Kaplan-Meier ===
kmf = KaplanMeierFitter()
kmf.fit(durations=df_all['duration'], event_observed=df_all['event'])

# === Guardar modelo ===
kmf.save_model("modelo-km.pkl")
print("Modelo guardado en modelo-km.pkl")


# === Mostrar curva de supervivencia ===
import matplotlib.pyplot as plt

kmf.plot_survival_function()
plt.title("Supervivencia de UTXOs")
plt.xlabel("Bloques de vida")
plt.ylabel("Probabilidad de seguir sin gastar")
plt.grid(True)
plt.show()
