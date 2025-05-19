import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import os
from functools import partial
from multiprocessing import Pool, cpu_count

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

def process_chunk(chunk, segment_size):
    """Procesa un chunk de datos y devuelve estadísticas parciales"""
    # Inicializar resultados locales
    local_stats = {}
    local_lifespans = []
    local_lifespans_zero = 0
    local_amounts = []
    local_locking_sizes = []
    # local_unlocking_sizes = []
    
    # Filtrar solo las filas con spent_block numérico (excluye "Unspent")
    # y realizar cálculos a nivel de dataframe en lugar de fila por fila
    spent_chunk = chunk[chunk["spent_block"] != "Unspent"].copy()
    
    if len(spent_chunk) == 0:
        return local_stats, local_lifespans, local_lifespans_zero, local_amounts, local_locking_sizes
        # , local_unlocking_sizes
    
    # Convertir columnas a numérico de una vez
    for col in ["spent_block", "creation_block", "value", "locking_script_size", "unlocking_script_size"]:
        spent_chunk[col] = pd.to_numeric(spent_chunk[col], errors="coerce")
    
    # Calcular lifespan para todos los elementos de una vez
    spent_chunk["lifespan"] = spent_chunk["spent_block"] - spent_chunk["creation_block"]
    
    # Clasificar en segmentos
    spent_chunk["segment"] = (spent_chunk["creation_block"] // segment_size) * segment_size
    
    # Separar UTXOs con lifespan = 0
    zero_lifespan = spent_chunk[spent_chunk["lifespan"] == 0]
    nonzero_lifespan = spent_chunk[spent_chunk["lifespan"] > 0]
    
    # Contar UTXOs con lifespan = 0 por segmento
    if not zero_lifespan.empty:
        zero_counts = zero_lifespan["segment"].value_counts()
        local_lifespans_zero = zero_lifespan.shape[0]
        
        for segment, count in zero_counts.items():
            if segment not in local_stats:
                local_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [], 
                    'amounts': [], 
                    'locking_sizes': [], 
                    # 'unlocking_sizes': []
                }
            local_stats[segment]['lifespan_0'] = count
    
    # Procesar UTXOs con lifespan > 0
    if not nonzero_lifespan.empty:
        # Añadir a totales
        local_lifespans = nonzero_lifespan["lifespan"].tolist()
        local_amounts = nonzero_lifespan["value"].tolist()
        local_locking_sizes = nonzero_lifespan["locking_script_size"].tolist()
        # local_unlocking_sizes = nonzero_lifespan["unlocking_script_size"].tolist()
        
        # Agrupar por segmento para estadísticas
        grouped = nonzero_lifespan.groupby("segment")
        
        for segment, group in grouped:
            if segment not in local_stats:
                local_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [], 
                    'amounts': [], 
                    'locking_sizes': [], 
                    # 'unlocking_sizes': []
                }
            
            local_stats[segment]['spent'] = group.shape[0]
            local_stats[segment]['lifespans'] = group["lifespan"].tolist()
            local_stats[segment]['amounts'] = group["value"].tolist()
            local_stats[segment]['locking_sizes'] = group["locking_script_size"].tolist()
            # local_stats[segment]['unlocking_sizes'] = group["unlocking_script_size"].tolist()
    
    return local_stats, local_lifespans, local_lifespans_zero, local_amounts, local_locking_sizes
    # , local_unlocking_sizes

def process_file(file, chunk_size=1_000_000, segment_size=50_000):
    """Procesar un archivo CSV completo"""
    print(f"Processing file: {file}")
    
    segment_stats = {}
    total_lifespans = []
    total_lifespans_zero = 0
    total_amounts = []
    total_locking_sizes = []
    # total_unlocking_sizes = []
    
    # Crear un iterador de chunks para procesar por partes
    chunks = pd.read_csv(file, names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
                        usecols=[0, 1, 2, 3, 4], skiprows=1, chunksize=chunk_size)
    
    # Procesar cada chunk
    for chunk in chunks:
        # local_stats, local_lifespans, local_lifespans_zero, local_amounts, local_locking_sizes, local_unlocking_sizes = process_chunk(chunk, segment_size)
        local_stats, local_lifespans, local_lifespans_zero, local_amounts, local_locking_sizes = process_chunk(chunk, segment_size)
        
        # Actualizar totales
        total_lifespans.extend(local_lifespans)
        total_lifespans_zero += local_lifespans_zero
        total_amounts.extend(local_amounts)
        total_locking_sizes.extend(local_locking_sizes)
        # total_unlocking_sizes.extend(local_unlocking_sizes)
        
        # Actualizar estadísticas por segmento
        for segment, stats in local_stats.items():
            if segment not in segment_stats:
                segment_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [],
                    'amounts': [], 
                    'locking_sizes': [], 
                    # 'unlocking_sizes': []
                }
            
            segment_stats[segment]['spent'] += stats['spent']
            segment_stats[segment]['lifespan_0'] += stats['lifespan_0']
            segment_stats[segment]['lifespans'].extend(stats['lifespans'])
            segment_stats[segment]['amounts'].extend(stats['amounts'])
            segment_stats[segment]['locking_sizes'].extend(stats['locking_sizes'])
            # segment_stats[segment]['unlocking_sizes'].extend(stats['unlocking_sizes'])
    
    return segment_stats, total_lifespans, total_lifespans_zero, total_amounts, total_locking_sizes
    # , total_unlocking_sizes

def generate_segment_plots(segment, data, output_dir, lifespan_bins, amount_bins, script_bins):
    """Genera gráficos para un segmento específico (puede ejecutarse en paralelo)"""
    if not data['lifespans']:  # Si no hay datos para graficar
        return
    
    # Lifespan plots
    plt.figure(figsize=(12, 6))
    plt.hist(data['lifespans'], bins=100, log=True, color="skyblue", edgecolor="black")
    plt.title(f"TXO Lifespan Distribution (Log Scale) - Segment {segment}")
    plt.xlabel("Lifespan (blocks)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}segment_{segment}_lifespan_distribution.png")
    plt.close()
    
    # Amount plots
    plt.figure(figsize=(12, 6))
    plt.hist(data['amounts'], bins=100, log=True, color="skyblue", edgecolor="black")
    plt.title(f"TXO Amount Distribution (Log Scale) - Segment {segment}")
    plt.xlabel("Amount (Satoshis)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}segment_{segment}_amount_distribution.png")
    plt.close()
    
    # Locking script size plots
    plt.figure(figsize=(12, 6))
    plt.hist(data['locking_sizes'], bins=100, log=True, color="skyblue", edgecolor="black")
    plt.title(f"TXO Locking Script Size Distribution (Log Scale) - Segment {segment}")
    plt.xlabel("Locking Script Size (bytes)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}segment_{segment}_locking_script_size_distribution.png")
    plt.close()
    
    # # Unlocking script size plots  
    # plt.figure(figsize=(12, 6))
    # plt.hist(data['unlocking_sizes'], bins=100, log=True, color="skyblue", edgecolor="black")
    # plt.title(f"TXO Unlocking Script Size Distribution (Log Scale) - Segment {segment}")
    # plt.xlabel("Unlocking Script Size (bytes)")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{output_dir}segment_{segment}_unlocking_script_size_distribution.png")
    # plt.close()

def write_statistics(output_dir, 
                    segment_stats, 
                    total_lifespans, 
                    total_lifespans_zero, 
                    total_amounts, 
                    total_locking_sizes, 
                    # total_unlocking_sizes,
                    lifespan_bins, 
                    amount_bins, 
                    script_bins, 
                    segment_size):
    """Escribir todas las estadísticas al archivo de salida"""
    # Convertir a arrays de numpy para cálculos más rápidos
    total_lifespans_np = np.array(total_lifespans)
    total_amounts_np = np.array(total_amounts)
    total_locking_sizes_np = np.array(total_locking_sizes)
    # total_unlocking_sizes_np = np.array(total_unlocking_sizes)
    
    # Series de pandas para algunas operaciones específicas
    total_lifespans_series = pd.Series(total_lifespans)
    total_amounts_series = pd.Series(total_amounts)
    total_locking_sizes_series = pd.Series(total_locking_sizes)
    # total_unlocking_sizes_series = pd.Series(total_unlocking_sizes)
    
    with open(f"{output_dir}utxo_statistics.txt", "w") as f:
        f.write("\n=== UTXO Statistics (Total, Lifespan > 0) ===\n")
        f.write(f"Total TXOs (Spent): {len(total_lifespans)}\n")
        f.write(f"Total TXOs (Lifespan = 0): {total_lifespans_zero}\n")

        # Calcular estadísticas usando numpy para mayor rapidez
        if len(total_lifespans) > 0:
            f.write(f"Average Lifespan: {np.mean(total_lifespans_np):.2f} blocks\n")
            f.write(f"Median Lifespan: {np.median(total_lifespans_np)} blocks\n")
            f.write(f"Min Lifespan: {np.min(total_lifespans_np)} blocks\n")
            f.write(f"Max Lifespan: {np.max(total_lifespans_np)} blocks\n\n")
            
            f.write(f"Average Amount: {np.mean(total_amounts_np)} Satoshis\n")
            f.write(f"Median Amount: {np.median(total_amounts_np)} Satoshis\n")
            f.write(f"Min Amount: {np.min(total_amounts_np)} Satoshis\n")
            f.write(f"Max Amount: {np.max(total_amounts_np)} Satoshis\n\n")
            
            f.write(f"Average Locking Script Size: {np.mean(total_locking_sizes_np)} bytes\n")
            f.write(f"Median Locking Script Size: {np.median(total_locking_sizes_np)} bytes\n")
            f.write(f"Min Locking Script Size: {np.min(total_locking_sizes_np)} bytes\n")
            f.write(f"Max Locking Script Size: {np.max(total_locking_sizes_np)} bytes\n\n")
            
            # f.write(f"Average Unlocking Script Size: {np.mean(total_unlocking_sizes_np)} bytes\n")
            # f.write(f"Median Unlocking Script Size: {np.median(total_unlocking_sizes_np)} bytes\n")
            # f.write(f"Min Unlocking Script Size: {np.min(total_unlocking_sizes_np)} bytes\n")
            # f.write(f"Max Unlocking Script Size: {np.max(total_unlocking_sizes_np)} bytes\n\n")

            # Distribución por lifespan
            lifespan_distribution = pd.cut(total_lifespans_series, bins=lifespan_bins).value_counts().sort_index()
            f.write("Distribution of Lifespan (Spent Only, >0):\n")
            f.write(str(lifespan_distribution) + "\n\n")

            # Distribuciones optimizadas
            for i in range(len(amount_bins) - 1):
                mask = (total_amounts_np >= amount_bins[i]) & (total_amounts_np < amount_bins[i + 1])
                lifespans_in_range = total_lifespans_np[mask]
                avg_lifespan = np.mean(lifespans_in_range) if len(lifespans_in_range) > 0 else 0
                f.write(f"Amount Range {amount_bins[i]} - {amount_bins[i+1]} Satoshis: Average Lifespan: {avg_lifespan:.2f}\n")

            for i in range(len(script_bins) - 1):
                lock_mask = (total_locking_sizes_np >= script_bins[i]) & (total_locking_sizes_np < script_bins[i + 1])
                # unlock_mask = (total_unlocking_sizes_np >= script_bins[i]) & (total_unlocking_sizes_np < script_bins[i + 1])
                
                lock_lifespans = total_lifespans_np[lock_mask]
                # unlock_lifespans = total_lifespans_np[unlock_mask]
                
                lock_avg = np.mean(lock_lifespans) if len(lock_lifespans) > 0 else 0
                # unlock_avg = np.mean(unlock_lifespans) if len(unlock_lifespans) > 0 else 0
                
                f.write(f"Locking Script Size {script_bins[i]} - {script_bins[i+1]} bytes: Average Lifespan: {lock_avg:.2f}\n")
                # f.write(f"Unlocking Script Size {script_bins[i]} - {script_bins[i+1]} bytes: Average Lifespan: {unlock_avg:.2f}\n")

        # Estadísticas segmentadas
        for segment, data in segment_stats.items():
            f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
            f.write(f"Total TXOs (Spent): {data['spent']}\n")
            f.write(f"Total TXOs (Lifespan = 0): {data['lifespan_0']}\n")

            # Lifespan
            if data['lifespans']:
                lifespans_np = np.array(data['lifespans'])
                lifespans = pd.Series(data['lifespans'])
                
                f.write(f"Average Lifespan: {np.mean(lifespans_np):.2f} blocks\n")
                f.write(f"Median Lifespan: {np.median(lifespans_np)} blocks\n")
                f.write(f"Min Lifespan: {np.min(lifespans_np)} blocks\n")
                f.write(f"Max Lifespan: {np.max(lifespans_np)} blocks\n\n")

                lifespan_distribution = pd.cut(lifespans, bins=lifespan_bins).value_counts().sort_index()
                f.write("Distribution of Lifespan (Spent Only, >0):\n")
                f.write(str(lifespan_distribution) + "\n\n")
            else:
                f.write("No spent TXOs in this segment.\n")

            # Amount
            if data['amounts']:
                amounts_np = np.array(data['amounts'])
                amounts = pd.Series(data['amounts'])
                
                f.write(f"Average Amount: {np.mean(amounts_np)} Satoshis\n")
                f.write(f"Median Amount: {np.median(amounts_np)} Satoshis\n")
                f.write(f"Min Amount: {np.min(amounts_np)} Satoshis\n")
                f.write(f"Max Amount: {np.max(amounts_np)} Satoshis\n\n")

                amount_distribution = pd.cut(amounts, bins=amount_bins).value_counts().sort_index()
                f.write("Distribution of Amount (Spent Only):\n")
                f.write(str(amount_distribution) + "\n\n")
            else:
                f.write("No spent TXOs in this segment.\n")

            # Locking Script Size
            if data['locking_sizes']:
                locking_sizes_np = np.array(data['locking_sizes'])
                locking_sizes = pd.Series(data['locking_sizes'])
                
                f.write(f"Average Locking Script Size: {np.mean(locking_sizes_np)} bytes\n")
                f.write(f"Median Locking Script Size: {np.median(locking_sizes_np)} bytes\n")
                f.write(f"Min Locking Script Size: {np.min(locking_sizes_np)} bytes\n")
                f.write(f"Max Locking Script Size: {np.max(locking_sizes_np)} bytes\n\n")

                locking_distribution = pd.cut(locking_sizes, bins=script_bins).value_counts().sort_index()
                f.write("Distribution of Locking Script Size (Spent Only):\n")
                f.write(str(locking_distribution) + "\n\n")
            else:
                f.write("No spent TXOs in this segment.\n")
            
            # # Unlocking Script Size
            # if data['unlocking_sizes']:
            #     unlocking_sizes_np = np.array(data['unlocking_sizes'])
            #     unlocking_sizes = pd.Series(data['unlocking_sizes'])
                
            #     f.write(f"Average Unlocking Script Size: {np.mean(unlocking_sizes_np)} bytes\n")
            #     f.write(f"Median Unlocking Script Size: {np.median(unlocking_sizes_np)} bytes\n")
            #     f.write(f"Min Unlocking Script Size: {np.min(unlocking_sizes_np)} bytes\n")
            #     f.write(f"Max Unlocking Script Size: {np.max(unlocking_sizes_np)} bytes\n\n")

            #     unlocking_distribution = pd.cut(unlocking_sizes, bins=script_bins).value_counts().sort_index()
            #     f.write("Distribution of Unlocking Script Size (Spent Only):\n")
            #     f.write(str(unlocking_distribution) + "\n\n")
            # else:
            #     f.write("No spent TXOs in this segment.\n")

# Procesamiento principal
def main():
    file_limit = 1  # Límite de archivos a procesar
    
    # Inicializar variables acumulativas
    all_segment_stats = {}
    all_lifespans = []
    all_lifespans_zero = 0
    all_amounts = []
    all_locking_sizes = []
    # all_unlocking_sizes = []
    
    # Procesar archivos
    for i, file in enumerate(csv_files):
        if i >= file_limit:
            print("Skipping file due to index limit.")
            break
            
        # segment_stats, lifespans, lifespans_zero, amounts, locking_sizes, unlocking_sizes = process_file(file, segment_size=segment_size)
        segment_stats, lifespans, lifespans_zero, amounts, locking_sizes = process_file(file, segment_size=segment_size)
        
        # Acumular resultados
        all_lifespans.extend(lifespans)
        all_lifespans_zero += lifespans_zero
        all_amounts.extend(amounts)
        all_locking_sizes.extend(locking_sizes)
        # all_unlocking_sizes.extend(unlocking_sizes)
        
        for segment, stats in segment_stats.items():
            if segment not in all_segment_stats:
                all_segment_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [],
                    'amounts': [], 
                    'locking_sizes': [], 
                    # 'unlocking_sizes': []
                }
                
            all_segment_stats[segment]['spent'] += stats['spent']
            all_segment_stats[segment]['lifespan_0'] += stats['lifespan_0']
            all_segment_stats[segment]['lifespans'].extend(stats['lifespans'])
            all_segment_stats[segment]['amounts'].extend(stats['amounts'])
            all_segment_stats[segment]['locking_sizes'].extend(stats['locking_sizes'])
            # all_segment_stats[segment]['unlocking_sizes'].extend(stats['unlocking_sizes'])
    
    # Escribir estadísticas al archivo de salida
    write_statistics(output_dir, 
                     all_segment_stats, 
                     all_lifespans, 
                     all_lifespans_zero, 
                     all_amounts, 
                     all_locking_sizes, 
                    #  all_unlocking_sizes,
                     lifespan_bins, 
                     amount_bins, 
                     script_bins, 
                     segment_size)
    
    # Generar gráficos en paralelo
    print("Generating plots...")
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        plot_fn = partial(generate_segment_plots, output_dir=output_dir, 
                          lifespan_bins=lifespan_bins, amount_bins=amount_bins, script_bins=script_bins)
        pool.starmap(plot_fn, [(segment, data) for segment, data in all_segment_stats.items()])
    
    print("Analysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()

