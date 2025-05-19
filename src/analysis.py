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

# Identificar el límite entre archivos spent y unspent
spent_files = [f for f in csv_files if int(re.search(r'utxo-history-(\d+)\.csv', f).group(1)) <= 248]
unspent_files = [f for f in csv_files if int(re.search(r'utxo-history-(\d+)\.csv', f).group(1)) >= 249]

# Configuración
segment_size = 50_000  # Tamaño de cada segmento
lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
amount_bins = [0, 10_000, 1_000_000, 100_000_000, 1_000_000_000, 10_000_000_000, float('inf')]
script_bins = [0, 20, 50, 100, 200, 500, float('inf')]

def process_spent_chunk(chunk, segment_size):
    """Procesa un chunk de datos de UTXOs gastados y devuelve estadísticas parciales"""
    # Inicializar resultados locales
    local_stats = {}
    local_lifespans = []
    local_lifespans_zero = 0
    local_amounts = []
    local_locking_sizes = []
    
    # Convertir columnas a numérico de una vez
    for col in ["spent_block", "creation_block", "value", "locking_script_size", "unlocking_script_size"]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    
    # Calcular lifespan para todos los elementos de una vez
    chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]
    
    # Clasificar en segmentos
    chunk["segment"] = (chunk["creation_block"] // segment_size) * segment_size
    
    # Separar UTXOs con lifespan = 0
    zero_lifespan = chunk[chunk["lifespan"] == 0]
    nonzero_lifespan = chunk[chunk["lifespan"] > 0]
    
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

def process_unspent_chunk(chunk, segment_size):
    """Procesa un chunk de datos de UTXOs no gastados y devuelve estadísticas parciales"""
    # Inicializar resultados locales
    local_stats = {}
    local_count = 0
    local_amounts = []
    local_locking_sizes = []
    
    # Convertir columnas numéricas
    chunk["creation_block"] = pd.to_numeric(chunk["creation_block"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    chunk["locking_script_size"] = pd.to_numeric(chunk["locking_script_size"], errors="coerce")
    
    # Clasificar en segmentos
    chunk["segment"] = (chunk["creation_block"] // segment_size) * segment_size
    
    # Contar UTXOs no gastados y procesar valores
    local_count = chunk.shape[0]
    local_amounts = chunk["value"].tolist()
    local_locking_sizes = chunk["locking_script_size"].tolist()
    
    # Agrupar por segmento
    grouped = chunk.groupby("segment")
    
    for segment, group in grouped:
        if segment not in local_stats:
            local_stats[segment] = {
                'count': 0,
                'amounts': [],
                'locking_sizes': []
            }
        
        local_stats[segment]['count'] = group.shape[0]
        local_stats[segment]['amounts'] = group["value"].tolist()
        local_stats[segment]['locking_sizes'] = group["locking_script_size"].tolist()
    
    return local_stats, local_count, local_amounts, local_locking_sizes

def process_spent_file(file, chunk_size=1_000_000, segment_size=50_000):
    """Procesar un archivo CSV de UTXOs gastados"""
    print(f"Processing spent UTXO file: {file}")
    
    segment_stats = {}
    total_lifespans = []
    total_lifespans_zero = 0
    total_amounts = []
    total_locking_sizes = []
    
    # Crear un iterador de chunks para procesar por partes
    chunks = pd.read_csv(
        file, 
        names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
        usecols=[0, 1, 2, 3, 4], 
        skiprows=1, 
        chunksize=chunk_size,
        low_memory=False
    )
    
    # Procesar cada chunk
    for chunk in chunks:
        local_stats, local_lifespans, local_lifespans_zero, local_amounts, local_locking_sizes = process_spent_chunk(chunk, segment_size)
        
        # Actualizar totales
        total_lifespans.extend(local_lifespans)
        total_lifespans_zero += local_lifespans_zero
        total_amounts.extend(local_amounts)
        total_locking_sizes.extend(local_locking_sizes)
        
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
            # if 'unlocking_sizes' in stats:
            #     segment_stats[segment]['unlocking_sizes'].extend(stats['unlocking_sizes'])
    
    return segment_stats, total_lifespans, total_lifespans_zero, total_amounts, total_locking_sizes

def process_unspent_file(file, chunk_size=1_000_000, segment_size=50_000):
    """Procesar un archivo CSV de UTXOs no gastados"""
    print(f"Processing unspent UTXO file: {file}")
    
    segment_stats = {}
    total_count = 0
    total_amounts = []
    total_locking_sizes = []
    
    # Crear un iterador de chunks para procesar por partes
    chunks = pd.read_csv(
        file, 
        names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
        usecols=[0, 1, 2, 3, 4], 
        skiprows=1, 
        chunksize=chunk_size,
        low_memory=False
    )
    
    # Procesar cada chunk
    for chunk in chunks:
        local_stats, local_count, local_amounts, local_locking_sizes = process_unspent_chunk(chunk, segment_size)
        
        # Actualizar totales
        total_count += local_count
        total_amounts.extend(local_amounts)
        total_locking_sizes.extend(local_locking_sizes)
        
        # Actualizar estadísticas por segmento
        for segment, stats in local_stats.items():
            if segment not in segment_stats:
                segment_stats[segment] = {
                    'count': 0,
                    'amounts': [],
                    'locking_sizes': []
                }
            
            segment_stats[segment]['count'] += stats['count']
            segment_stats[segment]['amounts'].extend(stats['amounts'])
            segment_stats[segment]['locking_sizes'].extend(stats['locking_sizes'])
    
    return segment_stats, total_count, total_amounts, total_locking_sizes

def generate_plots(data, output_dir, plot_prefix, color="skyblue", title_prefix=""):
    """Genera gráficos para un conjunto de datos"""
    # Plot para montos
    if 'amounts' in data and data['amounts']:
        plt.figure(figsize=(12, 6))
        plt.hist(data['amounts'], bins=100, log=True, color=color, edgecolor="black")
        plt.title(f"{title_prefix} Amount Distribution (Log Scale)")
        plt.xlabel("Amount (Satoshis)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}{plot_prefix}_amount_distribution.png")
        plt.close()
    
    # Plot para tamaños de locking script
    if 'locking_sizes' in data and data['locking_sizes']:
        plt.figure(figsize=(12, 6))
        plt.hist(data['locking_sizes'], bins=100, log=True, color=color, edgecolor="black")
        plt.title(f"{title_prefix} Locking Script Size Distribution (Log Scale)")
        plt.xlabel("Locking Script Size (bytes)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}{plot_prefix}_locking_script_distribution.png")
        plt.close()
    
    # Plot para lifespan (solo UTXOs gastados)
    if 'lifespans' in data and data['lifespans']:
        plt.figure(figsize=(12, 6))
        plt.hist(data['lifespans'], bins=100, log=True, color=color, edgecolor="black")
        plt.title(f"{title_prefix} Lifespan Distribution (Log Scale)")
        plt.xlabel("Lifespan (blocks)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}{plot_prefix}_lifespan_distribution.png")
        plt.close()

def generate_segment_plots(segment, data, output_dir, type_prefix="spent"):
    """Genera gráficos para un segmento específico con diferenciación por tipo"""
    if type_prefix == "spent":
        color = "skyblue"
        title_prefix = f"Spent TXO - Segment {segment}"
        # Verificar si hay datos de lifespan para gráficos
        if not data.get('lifespans'):
            return
    else:  # unspent
        color = "lightgreen"
        title_prefix = f"Unspent TXO - Segment {segment}"
        # Verificar si hay datos de montos para gráficos
        if not data.get('amounts'):
            return
    
    plot_prefix = f"segment_{segment}_{type_prefix}"
    generate_plots(data, output_dir, plot_prefix, color, title_prefix)

def calc_stats(values):
    """Calcula estadísticas básicas para un conjunto de valores"""
    if not values:
        return None
    
    values_np = np.array(values)
    return {
        'mean': np.mean(values_np),
        'median': np.median(values_np),
        'min': np.min(values_np),
        'max': np.max(values_np),
        'count': len(values_np)
    }

def write_statistics(output_dir, spent_stats, unspent_stats):
    """Escribir todas las estadísticas al archivo de salida"""
    with open(f"{output_dir}utxo_statistics.txt", "w") as f:
        # RESUMEN GENERAL
        f.write("======== UTXO SET ANALYSIS SUMMARY ========\n\n")
        
        # Estadísticas generales de UTXOs gastados
        spent_total = len(spent_stats['total_lifespans'])
        spent_zero = spent_stats['total_lifespans_zero']
        spent_all = spent_total + spent_zero
        
        f.write("=== SPENT UTXOs ===\n")
        f.write(f"Total Spent UTXOs: {spent_all}\n")
        f.write(f"- With lifespan > 0: {spent_total}\n")
        f.write(f"- With lifespan = 0: {spent_zero}\n")
        
        # Estadísticas de lifespan para UTXOs gastados
        if spent_total > 0:
            lifespan_stats = calc_stats(spent_stats['total_lifespans'])
            f.write(f"\nLifespan Statistics (blocks):\n")
            f.write(f"- Average: {lifespan_stats['mean']:.2f}\n")
            f.write(f"- Median: {lifespan_stats['median']}\n")
            f.write(f"- Min: {lifespan_stats['min']}\n")
            f.write(f"- Max: {lifespan_stats['max']}\n")
            
            # Distribución de lifespan
            lifespans_series = pd.Series(spent_stats['total_lifespans'])
            lifespan_distribution = pd.cut(lifespans_series, bins=lifespan_bins).value_counts().sort_index()
            f.write(f"\nLifespan Distribution:\n")
            f.write(str(lifespan_distribution) + "\n")
        
        # Estadísticas de montos para UTXOs gastados
        if spent_stats['total_amounts']:
            amount_stats = calc_stats(spent_stats['total_amounts'])
            f.write(f"\nAmount Statistics (Satoshis):\n")
            f.write(f"- Average: {amount_stats['mean']:.2f}\n")
            f.write(f"- Median: {amount_stats['median']}\n")
            f.write(f"- Min: {amount_stats['min']}\n")
            f.write(f"- Max: {amount_stats['max']}\n")
            
            # Distribución de montos
            amounts_series = pd.Series(spent_stats['total_amounts'])
            amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
            f.write(f"\nAmount Distribution:\n")
            f.write(str(amount_distribution) + "\n")
        
        # Estadísticas de locking script para UTXOs gastados
        if spent_stats['total_locking_sizes']:
            locking_stats = calc_stats(spent_stats['total_locking_sizes'])
            f.write(f"\nLocking Script Size Statistics (bytes):\n")
            f.write(f"- Average: {locking_stats['mean']:.2f}\n")
            f.write(f"- Median: {locking_stats['median']}\n")
            f.write(f"- Min: {locking_stats['min']}\n")
            f.write(f"- Max: {locking_stats['max']}\n")
            
            # Distribución de tamaños de locking script
            locking_series = pd.Series(spent_stats['total_locking_sizes'])
            locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
            f.write(f"\nLocking Script Size Distribution:\n")
            f.write(str(locking_distribution) + "\n")
        
        # Estadísticas generales de UTXOs no gastados
        f.write("\n\n=== UNSPENT UTXOs ===\n")
        f.write(f"Total Unspent UTXOs: {unspent_stats['total_count']}\n")
        
        # Estadísticas de montos para UTXOs no gastados
        if unspent_stats['total_amounts']:
            amount_stats = calc_stats(unspent_stats['total_amounts'])
            f.write(f"\nAmount Statistics (Satoshis):\n")
            f.write(f"- Average: {amount_stats['mean']:.2f}\n")
            f.write(f"- Median: {amount_stats['median']}\n")
            f.write(f"- Min: {amount_stats['min']}\n")
            f.write(f"- Max: {amount_stats['max']}\n")
            
            # Distribución de montos
            amounts_series = pd.Series(unspent_stats['total_amounts'])
            amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
            f.write(f"\nAmount Distribution:\n")
            f.write(str(amount_distribution) + "\n")
        
        # Estadísticas de locking script para UTXOs no gastados
        if unspent_stats['total_locking_sizes']:
            locking_stats = calc_stats(unspent_stats['total_locking_sizes'])
            f.write(f"\nLocking Script Size Statistics (bytes):\n")
            f.write(f"- Average: {locking_stats['mean']:.2f}\n")
            f.write(f"- Median: {locking_stats['median']}\n")
            f.write(f"- Min: {locking_stats['min']}\n")
            f.write(f"- Max: {locking_stats['max']}\n")
            
            # Distribución de tamaños de locking script
            locking_series = pd.Series(unspent_stats['total_locking_sizes'])
            locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
            f.write(f"\nLocking Script Size Distribution:\n")
            f.write(str(locking_distribution) + "\n")
        
        # ANÁLISIS COMPARATIVO
        f.write("\n\n======== COMPARATIVE ANALYSIS ========\n")
        
        # Montos: Comparativa entre UTXOs gastados vs. no gastados
        if spent_stats['total_amounts'] and unspent_stats['total_amounts']:
            spent_amount_avg = np.mean(np.array(spent_stats['total_amounts']))
            unspent_amount_avg = np.mean(np.array(unspent_stats['total_amounts']))
            spent_amount_median = np.median(np.array(spent_stats['total_amounts']))
            unspent_amount_median = np.median(np.array(unspent_stats['total_amounts']))
            
            f.write("\nAmount Comparison (Spent vs. Unspent):\n")
            f.write(f"- Average Amount (Spent): {spent_amount_avg:.2f} Satoshis\n")
            f.write(f"- Average Amount (Unspent): {unspent_amount_avg:.2f} Satoshis\n")
            f.write(f"- Ratio (Unspent/Spent): {unspent_amount_avg/spent_amount_avg:.2f}x\n")
            f.write(f"- Median Amount (Spent): {spent_amount_median} Satoshis\n")
            f.write(f"- Median Amount (Unspent): {unspent_amount_median} Satoshis\n")
            f.write(f"- Ratio (Unspent/Spent): {unspent_amount_median/spent_amount_median:.2f}x\n")
        
        # Locking Script: Comparativa entre UTXOs gastados vs. no gastados
        if spent_stats['total_locking_sizes'] and unspent_stats['total_locking_sizes']:
            spent_locking_avg = np.mean(np.array(spent_stats['total_locking_sizes']))
            unspent_locking_avg = np.mean(np.array(unspent_stats['total_locking_sizes']))
            spent_locking_median = np.median(np.array(spent_stats['total_locking_sizes']))
            unspent_locking_median = np.median(np.array(unspent_stats['total_locking_sizes']))
            
            f.write("\nLocking Script Size Comparison (Spent vs. Unspent):\n")
            f.write(f"- Average Size (Spent): {spent_locking_avg:.2f} bytes\n")
            f.write(f"- Average Size (Unspent): {unspent_locking_avg:.2f} bytes\n")
            f.write(f"- Ratio (Unspent/Spent): {unspent_locking_avg/spent_locking_avg:.2f}x\n")
            f.write(f"- Median Size (Spent): {spent_locking_median} bytes\n")
            f.write(f"- Median Size (Unspent): {unspent_locking_median} bytes\n")
            f.write(f"- Ratio (Unspent/Spent): {unspent_locking_median/spent_locking_median:.2f}x\n")
        
        # ESTADÍSTICAS POR SEGMENTO
        f.write("\n\n======== SEGMENTED ANALYSIS ========\n")
        
        # Unificar segmentos de ambos conjuntos para análisis
        all_segments = set(spent_stats['segment_stats'].keys()) | set(unspent_stats['segment_stats'].keys())
        
        for segment in sorted(all_segments):
            f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
            
            # Estadísticas del segmento para UTXOs gastados
            if segment in spent_stats['segment_stats']:
                seg_data = spent_stats['segment_stats'][segment]
                spent_count = seg_data['spent'] + seg_data.get('lifespan_0', 0)
                f.write(f"Spent UTXOs: {spent_count}\n")
                
                if seg_data['lifespans']:
                    lifespan_stats = calc_stats(seg_data['lifespans'])
                    f.write(f"- Average Lifespan: {lifespan_stats['mean']:.2f} blocks\n")
                    f.write(f"- Median Lifespan: {lifespan_stats['median']} blocks\n")
                
                if seg_data['amounts']:
                    amount_stats = calc_stats(seg_data['amounts'])
                    f.write(f"- Average Amount (Spent): {amount_stats['mean']:.2f} Satoshis\n")
            else:
                f.write("No Spent UTXOs in this segment.\n")
            
            # Estadísticas del segmento para UTXOs no gastados
            if segment in unspent_stats['segment_stats']:
                seg_data = unspent_stats['segment_stats'][segment]
                f.write(f"Unspent UTXOs: {seg_data['count']}\n")
                
                if seg_data['amounts']:
                    amount_stats = calc_stats(seg_data['amounts'])
                    f.write(f"- Average Amount (Unspent): {amount_stats['mean']:.2f} Satoshis\n")
            else:
                f.write("No Unspent UTXOs in this segment.\n")
            
            # Análisis comparativo del segmento (si hay datos de ambos tipos)
            if (segment in spent_stats['segment_stats'] and 
                segment in unspent_stats['segment_stats'] and
                spent_stats['segment_stats'][segment]['amounts'] and
                unspent_stats['segment_stats'][segment]['amounts']):
                
                spent_avg = np.mean(np.array(spent_stats['segment_stats'][segment]['amounts']))
                unspent_avg = np.mean(np.array(unspent_stats['segment_stats'][segment]['amounts']))
                
                f.write(f"\nSegment Comparison:\n")
                f.write(f"- Amount Ratio (Unspent/Spent): {unspent_avg/spent_avg:.2f}x\n")

def main():
    # Inicializar variables para UTXOs gastados
    spent_segment_stats = {}
    spent_lifespans = []
    spent_lifespans_zero = 0
    spent_amounts = []
    spent_locking_sizes = []
    
    # Inicializar variables para UTXOs no gastados
    unspent_segment_stats = {}
    unspent_count = 0
    unspent_amounts = []
    unspent_locking_sizes = []
    
    # Procesar archivos de UTXOs gastados (hasta 248)
    print(f"Processing {len(spent_files)} spent UTXO files...")
    for file in spent_files:
        seg_stats, lifespans, lifespans_zero, amounts, locking_sizes = process_spent_file(file, segment_size=segment_size)
        
        # Acumular resultados
        spent_lifespans.extend(lifespans)
        spent_lifespans_zero += lifespans_zero
        spent_amounts.extend(amounts)
        spent_locking_sizes.extend(locking_sizes)
        
        # Actualizar estadísticas por segmento
        for segment, stats in seg_stats.items():
            if segment not in spent_segment_stats:
                spent_segment_stats[segment] = {
                    'spent': 0, 
                    'lifespan_0': 0, 
                    'lifespans': [],
                    'amounts': [], 
                    'locking_sizes': [],
                    # 'unlocking_sizes': []
                }
            
            spent_segment_stats[segment]['spent'] += stats['spent']
            spent_segment_stats[segment]['lifespan_0'] += stats['lifespan_0']
            spent_segment_stats[segment]['lifespans'].extend(stats['lifespans'])
            spent_segment_stats[segment]['amounts'].extend(stats['amounts'])
            spent_segment_stats[segment]['locking_sizes'].extend(stats['locking_sizes'])
            # if 'unlocking_sizes' in stats:
            #     spent_segment_stats[segment]['unlocking_sizes'].extend(stats['unlocking_sizes'])
    
    # Procesar archivos de UTXOs no gastados (desde 249)
    print(f"Processing {len(unspent_files)} unspent UTXO files...")
    # print(f"File list: {unspent_files}")

    for file in unspent_files:
        seg_stats, count, amounts, locking_sizes = process_unspent_file(file, segment_size=segment_size)
        
        # Acumular resultados
        unspent_count += count
        unspent_amounts.extend(amounts)
        unspent_locking_sizes.extend(locking_sizes)
        
        # Actualizar estadísticas por segmento
        for segment, stats in seg_stats.items():
            if segment not in unspent_segment_stats:
                unspent_segment_stats[segment] = {
                    'count': 0,
                    'amounts': [],
                    'locking_sizes': []
                }
            
            unspent_segment_stats[segment]['count'] += stats['count']
            unspent_segment_stats[segment]['amounts'].extend(stats['amounts'])
            unspent_segment_stats[segment]['locking_sizes'].extend(stats['locking_sizes'])
    
    # Consolidar estadísticas para generar informes
    spent_stats = {
        'segment_stats': spent_segment_stats,
        'total_lifespans': spent_lifespans,
        'total_lifespans_zero': spent_lifespans_zero,
        'total_amounts': spent_amounts,
        'total_locking_sizes': spent_locking_sizes
    }
    
    unspent_stats = {
        'segment_stats': unspent_segment_stats,
        'total_count': unspent_count,
        'total_amounts': unspent_amounts,
        'total_locking_sizes': unspent_locking_sizes
    }
    
    # Escribir estadísticas al archivo de salida
    print("Writing statistics...")
    write_statistics(output_dir, spent_stats, unspent_stats)
    
    # Generar gráficos generales
    print("Generating overall plots...")
    # Gráficos para UTXOs gastados
    generate_plots({
        'lifespans': spent_lifespans,
        'amounts': spent_amounts,
        'locking_sizes': spent_locking_sizes
    }, output_dir, "spent_overall", "skyblue", "Spent TXOs")
    
    # Gráficos para UTXOs no gastados
    generate_plots({
        'amounts': unspent_amounts,
        'locking_sizes': unspent_locking_sizes
    }, output_dir, "unspent_overall", "lightgreen", "Unspent TXOs")
    
    # Generar gráficos comparativos
    print("Generating comparative plots...")
    
    # Comparativa de montos
    plt.figure(figsize=(14, 8))
    plt.hist(spent_amounts, bins=100, alpha=0.5, log=True, color="skyblue", edgecolor="black", label="Spent TXOs")
    plt.hist(unspent_amounts, bins=100, alpha=0.5, log=True, color="lightgreen", edgecolor="black", label="Unspent TXOs")
    plt.title(f"Amount Distribution Comparison (Log Scale)")
    plt.xlabel("Amount (Satoshis)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{output_dir}comparative_amount_distribution.png")
    plt.close()
    
    # Comparativa de locking script sizes
    plt.figure(figsize=(14, 8))
    plt.hist(spent_locking_sizes, bins=100, alpha=0.5, log=True, color="skyblue", edgecolor="black", label="Spent TXOs")
    plt.hist(unspent_locking_sizes, bins=100, alpha=0.5, log=True, color="lightgreen", edgecolor="black", label="Unspent TXOs")
    plt.title(f"Locking Script Size Distribution Comparison (Log Scale)")
    plt.xlabel("Locking Script Size (bytes)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{output_dir}comparative_locking_script_distribution.png")
    plt.close()
    
    # Generar gráficos por segmento
    print("Generating segment plots...")
    # Usar multiprocessing para generar gráficos por segmento más rápidamente
    plot_tasks = []
    
    # Tareas para UTXOs gastados
    for segment, data in spent_segment_stats.items():
        if data['lifespans']:
            plot_tasks.append((segment, data, "spent"))
    
    # Tareas para UTXOs no gastados
    for segment, data in unspent_segment_stats.items():
        if data['amounts']:
            plot_tasks.append((segment, data, "unspent"))
    
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        for segment, data, type_prefix in plot_tasks:
            generate_segment_plots(segment, data, output_dir, type_prefix)
    
    print("Analysis complete. Results saved to:", output_dir)

if __name__ == "__main__":
    main()

