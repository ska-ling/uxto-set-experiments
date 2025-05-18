import pandas as pd
import glob
import matplotlib.pyplot as plt

# Directorio de los archivos CSV
input_dir = "/home/fernando/dev/utxo-experiments/output/"
csv_files = glob.glob(f"{input_dir}utxo-history-*.csv")

# Leer y concatenar todos los archivos en un solo DataFrame
df = pd.concat([pd.read_csv(f, names=["txid", "index", "creation_block", "spent_block"]) for f in csv_files], ignore_index=True)

# Limpiar y convertir los datos
df["spent_block"] = pd.to_numeric(df["spent_block"], errors="coerce")
df["creation_block"] = pd.to_numeric(df["creation_block"])
df["lifespan"] = df["spent_block"] - df["creation_block"]
df.dropna(subset=["lifespan"], inplace=True)  # Eliminar UTXOs no gastados

# Calcular estadísticas generales
average_lifespan = df["lifespan"].mean()
median_lifespan = df["lifespan"].median()
min_lifespan = df["lifespan"].min()
max_lifespan = df["lifespan"].max()

print(f"Average lifespan: {average_lifespan:.2f} blocks")
print(f"Median lifespan: {median_lifespan} blocks")
print(f"Min lifespan: {min_lifespan} blocks")
print(f"Max lifespan: {max_lifespan} blocks")

# Distribución de vida útil por rangos
lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
df["lifespan_range"] = pd.cut(df["lifespan"], bins=lifespan_bins)
lifespan_distribution = df["lifespan_range"].value_counts().sort_index()
print("\nDistribution of Lifespan:")
print(lifespan_distribution)

# Distribución por año de creación
blocks_per_year = 52560  # Approx. 1 block every 10 minutes
df["creation_year"] = df["creation_block"] // blocks_per_year + 2009
creation_distribution = df["creation_year"].value_counts().sort_index()
print("\nDistribution of UTXOs by Creation Year:")
print(creation_distribution)

# Visualización de las distribuciones
plt.figure(figsize=(12, 6))
plt.hist(df["lifespan"], bins=100, log=True, color="skyblue", edgecolor="black")
plt.title("UTXO Lifespan Distribution (Log Scale)")
plt.xlabel("Lifespan (blocks)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
creation_distribution.plot(kind="bar", color="skyblue")
plt.title("UTXOs Created per Year")
plt.xlabel("Year")
plt.ylabel("Number of UTXOs")
plt.show()
