import pandas as pd
import glob
from pathlib import Path
import re

input_dir = Path("/home/fernando/dev/utxo-experiments/output")
output_dir = Path("/home/fernando/dev/utxo-experiments/parquet")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_index(path):
    match = re.search(r'utxo-history-(\d+)\.csv$', path)
    return int(match.group(1)) if match else -1

csv_files = sorted(glob.glob(str(input_dir / "utxo-history-*.csv")), key=extract_index)

for csv_path in csv_files:
    # print(f"Procesando: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        parquet_path = output_dir / (Path(csv_path).stem + ".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"✔️  {csv_path} → {parquet_path}")
    except Exception as e:
        print(f"❌  Error procesando {csv_path}: {e}")
