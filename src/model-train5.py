import pandas as pd
import glob
import re
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from pathlib import Path

FINAL_BLOCK_HEIGHT = 789_999
PARQUET_DIR = Path("/home/fernando/dev/utxo-experiments/parquet")

# === Archivos .parquet ===
def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

ALL_PARQUETS = sorted(glob.glob(str(PARQUET_DIR / "utxo-history-*.parquet")), key=extract_index)
SPENT_PARQUETS = ALL_PARQUETS[:353][:10]      # Cambiá el slice para escalar
UNSPENT_PARQUETS = ALL_PARQUETS[353:396][:10] # Idem

# === Loaders ===
def load_spent_data(files):
    rows = []
    for path in files:
        print(f"Procesando (spent): {path}")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            try:
                creation_block = int(row['creation_block'])
                spent_block = int(row['spent_block'])
                lifetime = spent_block - creation_block
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = int(row['unlocking_script_size']) if row['unlocking_script_size'] else -1
                tx_coinbase = str(row['tx_coinbase']).strip().lower() == 'true'
                op_return = str(row['op_return']).strip().lower() == 'true'
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
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            try:
                creation_block = int(row['creation_block'])
                value = int(row['value'])
                locking_script_size = int(row['locking_script_size'])
                unlocking_script_size = 0  # o -1 si preferís
                tx_coinbase = str(row['tx_coinbase']).strip().lower() == 'true'
                op_return = str(row['op_return']).strip().lower() == 'true'
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

# === Cargar y unir ===
print("Cargando datos...")
df_spent = load_spent_data(SPENT_PARQUETS)
df_unspent = load_unspent_data(UNSPENT_PARQUETS)
df_all = pd.concat([df_spent, df_unspent], ignore_index=True)

print(f"Total UTXOs: {len(df_all)}")

# === Kaplan-Meier
kmf = KaplanMeierFitter()
kmf.fit(durations=df_all['duration'], event_observed=df_all['event'])

# === Guardar curva estimada
kmf.survival_function_.to_parquet("survival_km.parquet")
print("Guardado survival_km.parquet")

# === Plot
kmf.plot_survival_function()
plt.title("Supervivencia de UTXOs")
plt.xlabel("Bloques de vida")
plt.ylabel("Probabilidad de seguir sin gastar")
plt.grid(True)
plt.show()
