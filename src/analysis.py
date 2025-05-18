import pandas as pd
import re
import glob
import matplotlib.pyplot as plt
import os

# Directorio de los archivos CSV
input_dir = "/home/fernando/dev/utxo-experiments/output/"
output_dir = "/home/fernando/dev/utxo-experiments/output/graphs/"
os.makedirs(output_dir, exist_ok=True)
csv_files = glob.glob(f"{input_dir}utxo-history-*.csv")

# Ordenar archivos correctamente por número
csv_files.sort(key=lambda x: int(re.search(r'utxo-history-(\d+)\.csv', x).group(1)))

# Variables para estadísticas
segment_size = 50_000  # Tamaño de cada segmento
segment_stats = {}
total_lifespans = []
total_lifespans_zero = 0

# file_index = 0

# Procesar archivos uno por uno
for file in csv_files:
    print(f"Processing file: {file}")
    # if file_index >= 3:
    #     print("Skipping file due to index limit.")
    #     break
    # file_index += 1

    for chunk in pd.read_csv(file, names=["txid_index", "creation_block", "spent_block"], 
                             usecols=[1, 2], skiprows=1, chunksize=1_000_000):
        # Filtrar las filas con "Unspent"
        chunk = chunk[chunk["spent_block"] != "Unspent"]

        # Convertir a numérico y calcular lifespan
        chunk["spent_block"] = pd.to_numeric(chunk["spent_block"], errors="coerce")
        chunk["creation_block"] = pd.to_numeric(chunk["creation_block"], errors="coerce")
        chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]

        # Segmentar por rango de bloques
        for _, row in chunk.iterrows():
            segment = (row['creation_block'] // segment_size) * segment_size
            if segment not in segment_stats:
                segment_stats[segment] = {
                    'spent': 0, 'lifespan_0': 0, 'lifespans': []
                }

            # Contabilizar en el segmento
            if row['lifespan'] == 0:
                segment_stats[segment]['lifespan_0'] += 1
                total_lifespans_zero += 1
            else:
                segment_stats[segment]['spent'] += 1
                segment_stats[segment]['lifespans'].append(row['lifespan'])
                total_lifespans.append(row['lifespan'])

# Generar estadísticas segmentadas y totales
with open(f"{output_dir}utxo_statistics.txt", "w") as f:
    f.write("\n=== UTXO Statistics (Total, Lifespan > 0) ===\n")
    total_lifespans_series = pd.Series(total_lifespans)
    f.write(f"Total TXOs (Spent): {len(total_lifespans)}\n")
    f.write(f"Total TXOs (Lifespan = 0): {total_lifespans_zero}\n")
    f.write(f"Average Lifespan: {total_lifespans_series.mean():.2f} blocks\n")
    f.write(f"Median Lifespan: {total_lifespans_series.median()} blocks\n")
    f.write(f"Min Lifespan: {total_lifespans_series.min()} blocks\n")
    f.write(f"Max Lifespan: {total_lifespans_series.max()} blocks\n\n")

    # Guardar distribución total
    lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
    lifespan_distribution = pd.cut(total_lifespans_series, bins=lifespan_bins).value_counts().sort_index()
    f.write("Distribution of Lifespan (Spent Only, >0):\n")
    f.write(str(lifespan_distribution) + "\n\n")

    # Graficar distribución total
    plt.figure(figsize=(12, 6))
    plt.hist(total_lifespans_series, bins=100, log=True, color="skyblue", edgecolor="black")
    plt.title("TXO Lifespan Distribution (Log Scale) - Total")
    plt.xlabel("Lifespan (blocks)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}total_lifespan_distribution.png")
    plt.close()

    # Estadísticas segmentadas
    for segment, data in segment_stats.items():
        f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
        f.write(f"Total TXOs (Spent): {data['spent']}\n")
        f.write(f"Total TXOs (Lifespan = 0): {data['lifespan_0']}\n")

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
