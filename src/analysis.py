import pandas as pd
import re
import glob
import matplotlib.pyplot as plt
import os

# Directorios
input_dir = "/home/fernando/dev/utxo-experiments/output/"
output_dir = "/home/fernando/dev/utxo-experiments/output/graphs/"
os.makedirs(output_dir, exist_ok=True)
csv_files = glob.glob(f"{input_dir}utxo-history-*.csv")

# Ordenar archivos correctamente por número
csv_files.sort(key=lambda x: int(re.search(r'utxo-history-(\d+)\.csv', x).group(1)))

# Configuración
segment_size = 50_000  # Tamaño de cada segmento
lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
amount_bins = [0, 10_000, 1_000_000, 100_000_000, 1_000_000_000, 10_000_000_000, float('inf')]
script_bins = [0, 20, 50, 100, 200, 500, float('inf')]

# Variables para estadísticas
segment_stats = {}
total_lifespans = []
total_lifespans_zero = 0
total_amounts = []
total_locking_sizes = []
total_unlocking_sizes = []

# file_index = 0

# Procesar archivos uno por uno
for file in csv_files:
    print(f"Processing file: {file}")
    if file_index >= 3:
        print("Skipping file due to index limit.")
        break
    file_index += 1    

    for chunk in pd.read_csv(file, names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
                             usecols=[0, 1, 2, 3, 4], skiprows=1, chunksize=1_000_000):
        # Filtrar las filas con "Unspent"
        chunk = chunk[chunk["spent_block"] != "Unspent"]

        # Convertir a numérico y calcular lifespan
        chunk["spent_block"] = pd.to_numeric(chunk["spent_block"], errors="coerce")
        chunk["creation_block"] = pd.to_numeric(chunk["creation_block"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk["locking_script_size"] = pd.to_numeric(chunk["locking_script_size"], errors="coerce")
        chunk["unlocking_script_size"] = pd.to_numeric(chunk["unlocking_script_size"], errors="coerce")
        chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]

        # Segmentar por rango de bloques
        for _, row in chunk.iterrows():
            segment = (row['creation_block'] // segment_size) * segment_size
            if segment not in segment_stats:
                segment_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [],
                    'amounts': [], 
                    'locking_sizes': [], 
                    'unlocking_sizes': []
                }

            # Contabilizar en el segmento
            if row['lifespan'] == 0:
                segment_stats[segment]['lifespan_0'] += 1
                total_lifespans_zero += 1
            else:
                segment_stats[segment]['spent'] += 1
                segment_stats[segment]['lifespans'].append(row['lifespan'])
                segment_stats[segment]['amounts'].append(row['value'])
                segment_stats[segment]['locking_sizes'].append(row['locking_script_size'])
                segment_stats[segment]['unlocking_sizes'].append(row['unlocking_script_size'])

                # Totales
                total_lifespans.append(row['lifespan'])
                total_amounts.append(row['value'])
                total_locking_sizes.append(row['locking_script_size'])
                total_unlocking_sizes.append(row['unlocking_script_size'])

# Generar estadísticas segmentadas y totales
with open(f"{output_dir}utxo_statistics.txt", "w") as f:
    f.write("\n=== UTXO Statistics (Total, Lifespan > 0) ===\n")
    total_lifespans_series = pd.Series(total_lifespans)
    f.write(f"Total TXOs (Spent): {len(total_lifespans)}\n")
    f.write(f"Total TXOs (Lifespan = 0): {total_lifespans_zero}\n")

    # Lifespan
    f.write(f"Average Lifespan: {total_lifespans_series.mean():.2f} blocks\n")
    f.write(f"Median Lifespan: {total_lifespans_series.median()} blocks\n")
    f.write(f"Min Lifespan: {total_lifespans_series.min()} blocks\n")
    f.write(f"Max Lifespan: {total_lifespans_series.max()} blocks\n\n")
    # Amount    
    f.write(f"Average Amount: {total_amounts.mean()} Satoshis\n")
    f.write(f"Median Amount: {pd.Series(total_amounts).median()} Satoshis\n")
    f.write(f"Min Amount: {pd.Series(total_amounts).min()} Satoshis\n")
    f.write(f"Max Amount: {pd.Series(total_amounts).max()} Satoshis\n\n")

    # Locking Script Size
    f.write(f"Average Locking Script Size: {pd.Series(total_locking_sizes).mean()} bytes\n")
    f.write(f"Median Locking Script Size: {pd.Series(total_locking_sizes).median()} bytes\n")
    f.write(f"Min Locking Script Size: {pd.Series(total_locking_sizes).min()} bytes\n")
    f.write(f"Max Locking Script Size: {pd.Series(total_locking_sizes).max()} bytes\n\n")
    
    # Unlocking Script Size
    f.write(f"Average Unlocking Script Size: {pd.Series(total_unlocking_sizes).mean()} bytes\n")
    f.write(f"Median Unlocking Script Size: {pd.Series(total_unlocking_sizes).median()} bytes\n")
    f.write(f"Min Unlocking Script Size: {pd.Series(total_unlocking_sizes).min()} bytes\n")
    f.write(f"Max Unlocking Script Size: {pd.Series(total_unlocking_sizes).max()} bytes\n\n")


    # Distribución por lifespan
    lifespan_distribution = pd.cut(total_lifespans_series, bins=lifespan_bins).value_counts().sort_index()
    f.write("Distribution of Lifespan (Spent Only, >0):\n")
    f.write(str(lifespan_distribution) + "\n\n")

    # Distribución por monto
    for i in range(len(amount_bins) - 1):
        filtered = [v for v in total_amounts if amount_bins[i] <= v < amount_bins[i + 1]]
        avg_lifespan = pd.Series([total_lifespans[j] for j, v in enumerate(total_amounts) 
                                  if amount_bins[i] <= v < amount_bins[i + 1]]).mean()
        f.write(f"Amount Range {amount_bins[i]} - {amount_bins[i+1]} Satoshis: Average Lifespan: {avg_lifespan:.2f}\n")

    # Distribución por tamaño de script
    for i in range(len(script_bins) - 1):
        lock_avg = pd.Series([total_lifespans[j] for j, v in enumerate(total_locking_sizes) 
                              if script_bins[i] <= v < script_bins[i + 1]]).mean()
        unlock_avg = pd.Series([total_lifespans[j] for j, v in enumerate(total_unlocking_sizes) 
                                if script_bins[i] <= v < script_bins[i + 1]]).mean()
        f.write(f"Locking Script Size {script_bins[i]} - {script_bins[i+1]} bytes: Average Lifespan: {lock_avg:.2f}\n")
        f.write(f"Unlocking Script Size {script_bins[i]} - {script_bins[i+1]} bytes: Average Lifespan: {unlock_avg:.2f}\n")

    # Estadísticas segmentadas
    for segment, data in segment_stats.items():
        f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
        f.write(f"Total TXOs (Spent): {data['spent']}\n")
        f.write(f"Total TXOs (Lifespan = 0): {data['lifespan_0']}\n")

        # Lifespan
        lifespans = pd.Series(data['lifespans'])
        if len(lifespans) > 0:
            f.write(f"Average Lifespan: {lifespans.mean():.2f} blocks\n")
            f.write(f"Median Lifespan: {lifespans.median()} blocks\n")
            f.write(f"Min Lifespan: {lifespans.min()} blocks\n")
            f.write(f"Max Lifespan: {lifespans.max()} blocks\n\n")

            lifespan_distribution = pd.cut(lifespans, bins=lifespan_bins).value_counts().sort_index()
            f.write("Distribution of Lifespan (Spent Only, >0):\n")
            f.write(str(lifespan_distribution) + "\n\n")

            # Graficar distribución del segmento
            plt.figure(figsize=(12, 6))
            plt.hist(lifespans, bins=100, log=True, color="skyblue", edgecolor="black")
            plt.title(f"TXO Lifespan Distribution (Log Scale) - Segment {segment}")
            plt.xlabel("Lifespan (blocks)")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}segment_{segment}_lifespan_distribution.png")
            plt.close()
        else:
            f.write("No spent TXOs in this segment.\n")

        # Amount
        amounts = pd.Series(data['amounts'])
        if len(amounts) > 0:
            f.write(f"Average Amount: {amounts.mean()} Satoshis\n")
            f.write(f"Median Amount: {amounts.median()} Satoshis\n")
            f.write(f"Min Amount: {amounts.min()} Satoshis\n")
            f.write(f"Max Amount: {amounts.max()} Satoshis\n\n")

            amount_distribution = pd.cut(amounts, bins=amount_bins).value_counts().sort_index()
            f.write("Distribution of Amount (Spent Only):\n")
            f.write(str(amount_distribution) + "\n\n")

            # Graficar distribución del segmento
            plt.figure(figsize=(12, 6))
            plt.hist(amounts, bins=100, log=True, color="skyblue", edgecolor="black")
            plt.title(f"TXO Amount Distribution (Log Scale) - Segment {segment}")
            plt.xlabel("Amount (Satoshis)")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}segment_{segment}_amount_distribution.png")
            plt.close()
        else:
            f.write("No spent TXOs in this segment.\n")

        # Locking Script Size
        locking_sizes = pd.Series(data['locking_sizes'])
        if len(locking_sizes) > 0:
            f.write(f"Average Locking Script Size: {locking_sizes.mean()} bytes\n")
            f.write(f"Median Locking Script Size: {locking_sizes.median()} bytes\n")
            f.write(f"Min Locking Script Size: {locking_sizes.min()} bytes\n")
            f.write(f"Max Locking Script Size: {locking_sizes.max()} bytes\n\n")

            locking_distribution = pd.cut(locking_sizes, bins=script_bins).value_counts().sort_index()
            f.write("Distribution of Locking Script Size (Spent Only):\n")
            f.write(str(locking_distribution) + "\n\n")

            # Graficar distribución del segmento
            plt.figure(figsize=(12, 6))
            plt.hist(locking_sizes, bins=100, log=True, color="skyblue", edgecolor="black")
            plt.title(f"TXO Locking Script Size Distribution (Log Scale) - Segment {segment}")
            plt.xlabel("Locking Script Size (bytes)")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}segment_{segment}_locking_script_size_distribution.png")
            plt.close()
        else:
            f.write("No spent TXOs in this segment.\n")
        
        # Unlocking Script Size
        unlocking_sizes = pd.Series(data['unlocking_sizes'])
        if len(unlocking_sizes) > 0:
            f.write(f"Average Unlocking Script Size: {unlocking_sizes.mean()} bytes\n")
            f.write(f"Median Unlocking Script Size: {unlocking_sizes.median()} bytes\n")
            f.write(f"Min Unlocking Script Size: {unlocking_sizes.min()} bytes\n")
            f.write(f"Max Unlocking Script Size: {unlocking_sizes.max()} bytes\n\n")

            unlocking_distribution = pd.cut(unlocking_sizes, bins=script_bins).value_counts().sort_index()
            f.write("Distribution of Unlocking Script Size (Spent Only):\n")
            f.write(str(unlocking_distribution) + "\n\n")

            # Graficar distribución del segmento
            plt.figure(figsize=(12, 6))
            plt.hist(unlocking_sizes, bins=100, log=True, color="skyblue", edgecolor="black")
            plt.title(f"TXO Unlocking Script Size Distribution (Log Scale) - Segment {segment}")
            plt.xlabel("Unlocking Script Size (bytes)")
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}segment_{segment}_unlocking_script_size_distribution.png")
            plt.close()
        else:
            f.write("No spent TXOs in this segment.\n")
    f.write("\n\n")

# # Graficar la distribución total de Lifespan
# plt.figure(figsize=(12, 6))
# plt.hist(total_lifespans, bins=100, log=True, color="skyblue", edgecolor="black")
# plt.title("Total TXO Lifespan Distribution (Log Scale)")
# plt.xlabel("Lifespan (blocks)")
# plt.ylabel("Frequency")
# plt.savefig(f"{output_dir}total_lifespan_distribution.png")
# plt.close()
# # Graficar la distribución total de Amount
# plt.figure(figsize=(12, 6))
# plt.hist(total_amounts, bins=100, log=True, color="skyblue", edgecolor="black")
# plt.title("Total TXO Amount Distribution (Log Scale)")
# plt.xlabel("Amount (Satoshis)")
# plt.ylabel("Frequency")
# plt.savefig(f"{output_dir}total_amount_distribution.png")
# plt.close()
# # Graficar la distribución total de Locking Script Size
# plt.figure(figsize=(12, 6))
# plt.hist(total_locking_sizes, bins=100, log=True, color="skyblue", edgecolor="black")
# plt.title("Total TXO Locking Script Size Distribution (Log Scale)")
# plt.xlabel("Locking Script Size (bytes)")
# plt.ylabel("Frequency")
# plt.savefig(f"{output_dir}total_locking_script_size_distribution.png")
# plt.close()
# # Graficar la distribución total de Unlocking Script Size
# plt.figure(figsize=(12, 6))
# plt.hist(total_unlocking_sizes, bins=100, log=True, color="skyblue", edgecolor="black")
# plt.title("Total TXO Unlocking Script Size Distribution (Log Scale)")
# plt.xlabel("Unlocking Script Size (bytes)")
# plt.ylabel("Frequency")
# plt.savefig(f"{output_dir}total_unlocking_script_size_distribution.png")
# plt.close()

