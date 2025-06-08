import pandas as pd
import glob
import re
import os
import os
from pathlib import Path


input_dir = Path("/home/fernando/dev/utxo-experiments/output/")  # <-- CAMBIA ESTA RUTA
output_dir = Path("/home/fernando/dev/utxo-experiments/output-fixed/")
output_dir.mkdir(exist_ok=True)

# el primer archivo con Unspent es el 353
# el último archivo es el 395

for i in range(353, 396):  # [353, 396)
    file = input_dir / f"utxo-history-{i}.csv"
    with open(file) as f:
        lines = f.readlines()

    if not lines:
        continue

    header = lines[0].strip()
    rows = lines[1:]

    fixed_lines = [header + '\n']

    for row in rows:
        row = row.strip()

        parts = row.split(',')
        if len(parts) == 6:
            # Caso incorrecto: falta una coma
            # parts = [creation_block, spent_block, value, locking_script_size, unlocking_script_size+tx_coinbase, op_return]
            unlocking_plus_coinbase = parts[4]
            if '-' in unlocking_plus_coinbase:
                unlocking, tx_coinbase = unlocking_plus_coinbase.split('-', 1)
                parts[4] = unlocking
                parts.insert(5, tx_coinbase)
        fixed_row = ','.join(parts)

        fixed_lines.append(fixed_row + '\n')

    with open(output_dir / f"utxo-history-{i}.csv", "w") as f:
        f.writelines(fixed_lines)
