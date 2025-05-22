import pandas as pd
import glob
import re
import os
import os
from pathlib import Path


input_dir = Path("/home/fernando/dev/utxo-experiments/output/")  # <-- CAMBIA ESTA RUTA

for i in range(395):  # 0 a 394 inclusive
    file = input_dir / f"utxo-history-{i}.csv"
    with open(file) as f:
        rows = f.readlines()

    if not rows:
        continue

    for row in rows:
        row = row.strip()
        if ",Unspent" in row:
            print(f"Primer archivo con Unspent: {file}")
            # end of the program
            exit(0)

print(f"No se encontró 'Unspent' en ningún archivo.")
