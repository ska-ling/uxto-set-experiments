import pandas as pd
import glob
import matplotlib.pyplot as plt

# Directorio de los archivos CSV
input_dir = "/home/fernando/dev/utxo-experiments/output/"
csv_files = glob.glob(f"{input_dir}utxo-history-0.csv")

# Variables para estadísticas
total_utxos = 0
total_spent = 0
total_unspent = 0
lifespans = []

# Procesar archivos uno por uno
for file in csv_files:
    print(f"Processing file: {file}")
    for chunk in pd.read_csv(file, names=["txid_index", "creation_block", "spent_block"], 
                             usecols=[1, 2], skiprows=1, chunksize=1_000_000):
        # Convertir a numérico y calcular lifespan
        chunk["spent_block"] = pd.to_numeric(chunk["spent_block"], errors="coerce")
        chunk["creation_block"] = pd.to_numeric(chunk["creation_block"])
        chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]

        # Estadísticas acumuladas
        total_utxos += len(chunk)
        total_spent += chunk["lifespan"].notna().sum()
        total_unspent += chunk["lifespan"].isna().sum()
        lifespans.extend(chunk["lifespan"].dropna().values)

# Convertir los tiempos de vida a pandas Series para análisis
lifespans = pd.Series(lifespans)
average_lifespan = lifespans.mean()
median_lifespan = lifespans.median()
min_lifespan = lifespans.min()
max_lifespan = lifespans.max()

print("\n=== UTXO Statistics ===")
print(f"Total UTXOs: {total_utxos}")
print(f"Total Spent: {total_spent}")
print(f"Total Unspent: {total_unspent}")
print(f"Average Lifespan: {average_lifespan:.2f} blocks")
print(f"Median Lifespan: {median_lifespan} blocks")
print(f"Min Lifespan: {min_lifespan} blocks")
print(f"Max Lifespan: {max_lifespan} blocks")

# Distribución de vida útil por rangos
lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
lifespan_distribution = pd.cut(lifespans, bins=lifespan_bins).value_counts().sort_index()
print("\nDistribution of Lifespan:")
print(lifespan_distribution)

# Distribución por año de creación
blocks_per_year = 52560  # Approx. 1 block every 10 minutes
creation_years = (lifespans.index // blocks_per_year) + 2009
creation_distribution = creation_years.value_counts().sort_index()
print("\nDistribution of UTXOs by Creation Year:")
print(creation_distribution)

# Visualización de las distribuciones
plt.figure(figsize=(12, 6))
plt.hist(lifespans, bins=100, log=True, color="skyblue", edgecolor="black")
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
