import pandas as pd
import glob
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from pathlib import Path

# === Rutas ===
NORMALIZED_DIR = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")

# === Cargar todos los .parquet normalizados ===
print("Cargando archivos parquet normalizados...")
all_files = sorted(glob.glob(str(NORMALIZED_DIR / "utxo-history-*.parquet")))

dfs = []
for path in all_files:
    print(f" -> {path}")
    try:
        df = pd.read_parquet(path)
        dfs.append(df[['duration', 'event']])
    except Exception as e:
        print(f"❌ Error en {path}: {e}")

df_all = pd.concat(dfs, ignore_index=True)
print(f"✔️  Total UTXOs cargados: {len(df_all)}")

# === Kaplan-Meier ===
print("Entrenando modelo Kaplan-Meier...")
kmf = KaplanMeierFitter()
kmf.fit(durations=df_all['duration'], event_observed=df_all['event'])

# === Guardar curva estimada
out_path = "survival_km.parquet"
kmf.survival_function_.to_parquet(out_path)
print(f"✔️  Curva de supervivencia guardada en: {out_path}")

# === Mostrar gráfico
kmf.plot_survival_function()
plt.title("Supervivencia de UTXOs")
plt.xlabel("Bloques de vida")
plt.ylabel("Probabilidad de seguir sin gastar")
plt.grid(True)
plt.show()
