import pandas as pd
import matplotlib.pyplot as plt
import random

# === Cargar curva de supervivencia estimada ===
print("Cargando curva de supervivencia...")
df_km = pd.read_parquet("survival_km.parquet")

# Convertimos a Series para facilidad de acceso
survival_curve = df_km['KM_estimate']
timeline = df_km.index.to_series()

# === Guardar gráfico
print("Guardando gráfico de supervivencia...")
plt.figure()
plt.plot(timeline, survival_curve, label="Kaplan-Meier estimate")
plt.title("Supervivencia de UTXOs")
plt.xlabel("Bloques de vida")
plt.ylabel("Probabilidad de seguir sin gastar")
plt.grid(True)
plt.legend()
plt.savefig("survival_km.png", dpi=300)
print("✅ Gráfico guardado como survival_km.png")


# === Consultas random ===
print("Consultas a la curva de supervivencia:")
for _ in range(5):
    d = random.randint(1, 200_000)  # cambiar rango si querés otro
    # Buscar la estimación más cercana por duración
    closest_index = timeline.searchsorted(d, side='right') - 1
    if closest_index >= 0:
        prob = survival_curve.iloc[closest_index]
        print(f"Duración {d} bloques → probabilidad de no haber sido gastado: {prob:.4f}")
    else:
        print(f"Duración {d} bloques → fuera del rango de la curva.")

