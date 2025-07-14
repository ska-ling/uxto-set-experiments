import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar CSVs
utxoz_csv = '/Users/fernando/utxo_z_benchmark_summary_20250710_150136.csv'
leveldb_csv = '/Users/fernando/utxo_z_benchmark_summary_20250710_150503.csv'

utxoz_df = pd.read_csv(utxoz_csv)
leveldb_df = pd.read_csv(leveldb_csv)

# Crear gráfico
plt.figure(figsize=(12, 6))

plt.plot(utxoz_df['batch'], utxoz_df['tps'], label='UTXO-Z', linewidth=2)
plt.plot(leveldb_df['batch'], leveldb_df['tps'], label='LevelDB', linewidth=2)

plt.xlabel('Batch (2M transactions per batch)')
plt.ylabel('Transactions per Second (TPS)')
plt.title('TPS per Batch: UTXO-Z vs LevelDB')
plt.legend()
plt.grid(True)

# Guardar con fecha en el nombre
fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig(f'tps_per_batch_{fecha}.png', dpi=300)

# Calcular cuántas veces UTXO-Z es más rápido que LevelDB (promedio de TPS)
utxoz_avg_tps = utxoz_df['tps'].mean()
leveldb_avg_tps = leveldb_df['tps'].mean()
if leveldb_avg_tps > 0:
    speedup = utxoz_avg_tps / leveldb_avg_tps
    print(f"UTXO-Z es {speedup:.2f}x más rápido que LevelDB (promedio TPS)")
else:
    print("No se puede calcular el speedup: LevelDB TPS promedio es 0")

# Calcular máx, min y median de TPS para ambos
utxoz_max = utxoz_df['tps'].max()
utxoz_min = utxoz_df['tps'].min()
utxoz_median = utxoz_df['tps'].median()
leveldb_max = leveldb_df['tps'].max()
leveldb_min = leveldb_df['tps'].min()
leveldb_median = leveldb_df['tps'].median()

print(f"UTXO-Z TPS -> max: {utxoz_max:.2f}, min: {utxoz_min:.2f}, median: {utxoz_median:.2f}")
print(f"LevelDB TPS -> max: {leveldb_max:.2f}, min: {leveldb_min:.2f}, median: {leveldb_median:.2f}")

# Calcular el nX batch a batch
merged = pd.merge(utxoz_df[['batch', 'tps']], leveldb_df[['batch', 'tps']], on='batch', suffixes=('_utxoz', '_leveldb'))
merged['nX'] = merged['tps_utxoz'] / merged['tps_leveldb']

print("\nSpeedup UTXO-Z vs LevelDB por batch:")
for idx, row in merged.iterrows():
    print(f"Batch {row['batch']}: {row['nX']:.2f}x")

nx_max = merged['nX'].max()
nx_min = merged['nX'].min()
nx_mean = merged['nX'].mean()
nx_median = merged['nX'].median()

print(f"\nnX stats -> max: {nx_max:.2f}, min: {nx_min:.2f}, mean: {nx_mean:.2f}, median: {nx_median:.2f}")

# plt.show()
