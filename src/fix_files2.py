import pandas as pd
import glob
import re
import os
import os
from pathlib import Path


input_dir = Path("/home/fernando/dev/utxo-experiments/output/")  # <-- CAMBIA ESTA RUTA
output_dir = Path("/home/fernando/dev/utxo-experiments/output-fixed/")
output_dir.mkdir(exist_ok=True)

found_unspent = False

for i in range(395):  # 0 a 394 inclusive
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
        if not found_unspent and "Unspent" in row:
            print(f"Primer archivo con Unspent: {file}")
            found_unspent = True

        if found_unspent:
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
        else:
            fixed_row = row

        fixed_lines.append(fixed_row + '\n')

    if found_unspent:
        with open(output_dir / f"utxo-history-{i}.csv", "w") as f:
            f.writelines(fixed_lines)
