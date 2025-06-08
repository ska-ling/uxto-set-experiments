import pandas as pd
import glob
import re
from pathlib import Path

FINAL_BLOCK_HEIGHT = 789_999
SRC_DIR = Path("/home/fernando/dev/utxo-experiments/parquet")
DST_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
DST_DIR.mkdir(parents=True, exist_ok=True)

def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.parquet$', path)
    return int(match.group(1)) if match else -1

# === Listas de archivos ===
all_parquets = sorted(glob.glob(str(SRC_DIR / "utxo-history-*.parquet")), key=extract_index)
spent_files = all_parquets[:353]
unspent_files = all_parquets[353:396]

# === Epoch helper ===
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

# === Función procesadora ===
def process_parquet_files(files, is_spent):
    for path in files:
        print(f"Procesando {'spent' if is_spent else 'unspent'}: {path}")
        try:
            df = pd.read_parquet(path)
            df = df.copy()

            df['creation_block'] = df['creation_block'].astype(int)
            df['value'] = df['value'].astype(int)
            df['locking_script_size'] = df['locking_script_size'].astype(int)
            df['tx_coinbase'] = df['tx_coinbase'].astype(str).str.lower() == 'true'
            df['op_return'] = df['op_return'].astype(str).str.lower() == 'true'

            if is_spent:
                df['spent_block'] = df['spent_block'].astype(int)
                df['duration'] = df['spent_block'] - df['creation_block']
                df['event'] = True
                df['unlocking_script_size'] = pd.to_numeric(df['unlocking_script_size'], errors='coerce').fillna(-1).astype(int)
            else:
                df['duration'] = FINAL_BLOCK_HEIGHT - df['creation_block']
                df['event'] = False
                df['unlocking_script_size'] = 0
                df['spent_block'] = pd.NA  # mantener columna aunque esté vacía

            df['epoch'] = df['creation_block'].apply(assign_epoch)

            out_path = DST_DIR / Path(path).name
            df.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"❌ Error procesando {path}: {e}")

process_parquet_files(spent_files, is_spent=True)
process_parquet_files(unspent_files, is_spent=False)

print(f"✅ Archivos procesados guardados en: {DST_DIR}")
