import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import gc


chunk_size = 1_000_000  # Reducir el tamaño del chunk

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
min_block = 0
max_block = 800_000  # Último bloque conocido

# Pre-calcular todos los segmentos posibles
all_segments = list(range(min_block, max_block, segment_size))
print(f"Pre-calculated {len(all_segments)} segments from {min_block} to {max_block}")

lifespan_bins = [0, 10, 100, 1000, 10000, 100000, 1000000]
amount_bins = [0, 10_000, 1_000_000, 100_000_000, 1_000_000_000, 10_000_000_000, float('inf')]
script_bins = [0, 20, 50, 100, 200, 500, float('inf')]

def process_spent_chunk_for_segment(chunk, target_segment=None, collect_totals=False):
    """Procesa un chunk de datos de UTXOs gastados y devuelve estadísticas para un segmento específico o totales"""
    # Inicializar resultados
    results = {
        'lifespans': [],
        'lifespans_zero': 0,
        'amounts': [],
        'locking_sizes': []
    }
    
    # Convertir columnas a numérico de una vez
    for col in ["spent_block", "creation_block", "value", "locking_script_size", "unlocking_script_size"]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    
    # Calcular lifespan para todos los elementos de una vez
    chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]
    
    # Clasificar en segmentos
    chunk["segment"] = (chunk["creation_block"] // segment_size) * segment_size
    
    # Si estamos recolectando totales
    if collect_totals:
        # Separar UTXOs con lifespan = 0 para estadísticas totales
        zero_lifespan = chunk[chunk["lifespan"] == 0]
        nonzero_lifespan = chunk[chunk["lifespan"] > 0]
        
        results['lifespans_zero'] = zero_lifespan.shape[0]
        if not nonzero_lifespan.empty:
            results['lifespans'] = nonzero_lifespan["lifespan"].tolist()
            results['amounts'] = nonzero_lifespan["value"].tolist()
            results['locking_sizes'] = nonzero_lifespan["locking_script_size"].tolist()
    
    # Si estamos procesando un segmento específico
    elif target_segment is not None:
        # Filtrar por el segmento objetivo
        segment_data = chunk[chunk["segment"] == target_segment]
        
        if not segment_data.empty:
            # Contar UTXOs con lifespan = 0 en este segmento
            zero_lifespan = segment_data[segment_data["lifespan"] == 0]
            nonzero_lifespan = segment_data[segment_data["lifespan"] > 0]
            
            results['lifespans_zero'] = zero_lifespan.shape[0]
            if not nonzero_lifespan.empty:
                results['lifespans'] = nonzero_lifespan["lifespan"].tolist()
                results['amounts'] = nonzero_lifespan["value"].tolist()
                results['locking_sizes'] = nonzero_lifespan["locking_script_size"].tolist()
    
    return results

def process_unspent_chunk_for_segment(chunk, target_segment=None, collect_totals=False):
    """Procesa un chunk de datos de UTXOs no gastados y devuelve estadísticas para un segmento específico o totales"""
    # Inicializar resultados
    results = {
        'count': 0,
        'amounts': [],
        'locking_sizes': []
    }
    
    # Convertir columnas numéricas
    chunk["creation_block"] = pd.to_numeric(chunk["creation_block"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    chunk["locking_script_size"] = pd.to_numeric(chunk["locking_script_size"], errors="coerce")
    
    # Clasificar en segmentos
    chunk["segment"] = (chunk["creation_block"] // segment_size) * segment_size
    
    # Si estamos recolectando totales
    if collect_totals:
        results['count'] = chunk.shape[0]
        results['amounts'] = chunk["value"].tolist()
        results['locking_sizes'] = chunk["locking_script_size"].tolist()
    
    # Si estamos procesando un segmento específico
    elif target_segment is not None:
        # Filtrar por el segmento objetivo
        segment_data = chunk[chunk["segment"] == target_segment]
        
        if not segment_data.empty:
            results['count'] = segment_data.shape[0]
            results['amounts'] = segment_data["value"].tolist()
            results['locking_sizes'] = segment_data["locking_script_size"].tolist()
    
    return results

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

def process_total_spent_stats():
    """Procesa todos los archivos para obtener estadísticas totales de UTXOs gastados"""
    # Inicializar variables para UTXOs gastados
    spent_lifespans_zero = 0
    
    # Usamos listas optimizadas de NumPy en lugar de listas de Python
    # para reducir el uso de memoria y mejorar el rendimiento
    spent_lifespans = np.array([], dtype=np.int32)
    spent_amounts = np.array([], dtype=np.int64)
    spent_locking_sizes = np.array([], dtype=np.int16)
    
    # Procesar archivos de UTXOs gastados
    print(f"  Processing {len(spent_files)} spent UTXO files for totals...")
    for i, file in enumerate(spent_files):
        print(f"    Processing file: {file} ({i+1}/{len(spent_files)})")
        chunks = pd.read_csv(
            file, 
            names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
            usecols=[0, 1, 2, 3], # Skip unlocking_script_size to save memory
            dtype={
                "creation_block": np.int32,
                "spent_block": np.int32,
                "value": np.int64,
                "locking_script_size": np.int16
            },
            skiprows=1, 
            chunksize=chunk_size,
            low_memory=False
        )
        
        for chunk in chunks:
            # Calcular lifespan directamente
            chunk["lifespan"] = chunk["spent_block"] - chunk["creation_block"]
            
            # Ya no necesitamos identificar segmentos aquí porque los tenemos precalculados
            
            # Procesar UTXOs con lifespan = 0 por separado
            zero_mask = chunk["lifespan"] == 0
            zero_count = zero_mask.sum()
            spent_lifespans_zero += zero_count
            
            # Procesar UTXOs con lifespan > 0
            if zero_count < len(chunk):
                nonzero_mask = ~zero_mask
                
                # Extraer valores como arrays de NumPy
                chunk_lifespans = chunk.loc[nonzero_mask, "lifespan"].to_numpy(dtype=np.int32)
                chunk_amounts = chunk.loc[nonzero_mask, "value"].to_numpy(dtype=np.int64)
                chunk_locking = chunk.loc[nonzero_mask, "locking_script_size"].to_numpy(dtype=np.int16)
                
                # Concatenar con los arrays existentes
                spent_lifespans = np.concatenate([spent_lifespans, chunk_lifespans])
                spent_amounts = np.concatenate([spent_amounts, chunk_amounts])
                spent_locking_sizes = np.concatenate([spent_locking_sizes, chunk_locking])
                
                # Si los arrays crecen demasiado, podemos calcular estadísticas parciales
                # y mantener solo un resumen, pero para este caso conservamos todos los valores
                # para cálculos exactos de la mediana y distribuciones
            
            # Liberar memoria
            # del chunk
            # gc.collect()
    
    # Consolidar resultados - ya no necesitamos devolver all_segments
    # Convertir arrays de NumPy a listas de Python para compatibilidad con el código existente
    return {
        'total_lifespans': spent_lifespans.tolist(),
        'total_lifespans_zero': spent_lifespans_zero,
        'total_amounts': spent_amounts.tolist(),
        'total_locking_sizes': spent_locking_sizes.tolist(),
    }

def process_total_unspent_stats():
    """Procesa todos los archivos para obtener estadísticas totales de UTXOs no gastados"""
    # Inicializar variables para UTXOs no gastados
    unspent_count = 0
    unspent_amounts = []
    unspent_locking_sizes = []
    
    # Procesar archivos de UTXOs no gastados
    print(f"  Processing {len(unspent_files)} unspent UTXO files for totals...")
    for file in unspent_files:
        print(f"    Processing file: {file}")
        chunks = pd.read_csv(
            file, 
            names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
            usecols=[0, 1, 2, 3], # Skip unlocking_script_size to save memory
            dtype={
                "creation_block": np.int32,
                "spent_block": np.int32,
                "value": np.int64,
                "locking_script_size": np.int16
            },
            skiprows=1, 
            chunksize=chunk_size,
            low_memory=False
        )
        
        for chunk in chunks:
            # Procesar el chunk para estadísticas totales
            results = process_unspent_chunk_for_segment(chunk, collect_totals=True)
            
            # Actualizar totales
            unspent_count += results['count']
            unspent_amounts.extend(results['amounts'])
            unspent_locking_sizes.extend(results['locking_sizes'])
            
            # Ya no necesitamos identificar segmentos
            
            # del chunk
            # gc.collect()
    
    # Consolidar resultados - ya no necesitamos devolver all_segments
    return {
        'total_count': unspent_count,
        'total_amounts': unspent_amounts,
        'total_locking_sizes': unspent_locking_sizes,
    }

def process_segment_spent_stats(segment):
    """Procesa todos los archivos para un segmento específico de UTXOs gastados"""
    # Inicializar resultados para este segmento
    spent = 0
    lifespan_0 = 0
    lifespans = []
    amounts = []
    locking_sizes = []
    
    # Procesar archivos de UTXOs gastados para este segmento
    for file in spent_files:
        print(f"    Processing file: {file}")
        chunks = pd.read_csv(
            file, 
            names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
            usecols=[0, 1, 2, 3, 4], 
            skiprows=1, 
            chunksize=chunk_size,
            low_memory=False
        )
        
        for chunk in chunks:
            # Procesar el chunk para este segmento
            results = process_spent_chunk_for_segment(chunk, target_segment=segment)
            
            # Actualizar estadísticas del segmento
            spent += len(results['lifespans'])
            lifespan_0 += results['lifespans_zero']
            lifespans.extend(results['lifespans'])
            amounts.extend(results['amounts'])
            locking_sizes.extend(results['locking_sizes'])
            
            # del chunk
            # gc.collect()
    
    # Consolidar resultados
    return {
        'spent': spent,
        'lifespan_0': lifespan_0,
        'lifespans': lifespans,
        'amounts': amounts,
        'locking_sizes': locking_sizes
    }

def process_segment_unspent_stats(segment):
    """Procesa todos los archivos para un segmento específico de UTXOs no gastados"""
    # Inicializar resultados para este segmento
    count = 0
    amounts = []
    locking_sizes = []
    
    # Procesar archivos de UTXOs no gastados para este segmento
    for file in unspent_files:
        print(f"    Processing file: {file}")
        chunks = pd.read_csv(
            file, 
            names=["creation_block", "spent_block", "value", "locking_script_size", "unlocking_script_size"], 
            usecols=[0, 1, 2, 3, 4], 
            skiprows=1, 
            chunksize=chunk_size,
            low_memory=False
        )
        
        for chunk in chunks:
            # Procesar el chunk para este segmento
            results = process_unspent_chunk_for_segment(chunk, target_segment=segment)
            
            # Actualizar estadísticas del segmento
            count += results['count']
            amounts.extend(results['amounts'])
            locking_sizes.extend(results['locking_sizes'])
            
            # del chunk
            # gc.collect()
    
    # Consolidar resultados
    return {
        'count': count,
        'amounts': amounts,
        'locking_sizes': locking_sizes
    }

def write_spent_total_stats(output_dir, spent_stats):
    """Escribe las estadísticas totales de UTXOs gastados a un archivo"""
    filename = f"{output_dir}spent_utxo_statistics.txt"
    print(f"Writing spent UTXO statistics to file {filename}...")
    
    # Convertir listas a numpy arrays para cálculos más eficientes
    spent_total = len(spent_stats['total_lifespans'])
    spent_zero = spent_stats['total_lifespans_zero']
    spent_all = spent_total + spent_zero
    
    with open(filename, "w") as f:
        # Estadísticas generales de UTXOs gastados
        f.write("======== SPENT UTXO ANALYSIS SUMMARY ========\n\n")
        f.write(f"Total Spent UTXOs: {spent_all}\n")
        f.write(f"- With lifespan > 0: {spent_total}\n")
        f.write(f"- With lifespan = 0: {spent_zero}\n")
        
        # Estadísticas de lifespan para UTXOs gastados
        if spent_total > 0:
            # Crear array NumPy para cálculos eficientes
            lifespans_np = np.array(spent_stats['total_lifespans'], dtype=np.int32)
            
            # Calcular estadísticas básicas
            lifespan_mean = np.mean(lifespans_np)
            lifespan_median = np.median(lifespans_np)  # Mediana exacta
            lifespan_min = np.min(lifespans_np)
            lifespan_max = np.max(lifespans_np)
            
            f.write(f"\nLifespan Statistics (blocks):\n")
            f.write(f"- Average: {lifespan_mean:.2f}\n")
            f.write(f"- Median: {lifespan_median}\n")
            f.write(f"- Min: {lifespan_min}\n")
            f.write(f"- Max: {lifespan_max}\n")
            
            # Calcular distribución de manera eficiente sin pd.cut
            f.write(f"\nLifespan Distribution:\n")
            
            # Función auxiliar para contar valores en rangos específicos
            def count_in_bins(data, bins):
                counts = {}
                for i in range(len(bins)-1):
                    start = bins[i]
                    end = bins[i+1]
                    
                    if i == len(bins)-2:  # Último bin
                        mask = (data >= start)
                        label = f"[{start}, inf)"
                    else:
                        mask = (data >= start) & (data < end)
                        label = f"[{start}, {end})"
                    
                    counts[label] = np.sum(mask)
                return counts
            
            lifespan_dist = count_in_bins(lifespans_np, lifespan_bins)
            for label, count in sorted(lifespan_dist.items()):
                f.write(f"{label}: {count}\n")
            
            # Liberar memoria
            del lifespans_np
            gc.collect()
        
        # Estadísticas de montos para UTXOs gastados
        if spent_stats['total_amounts']:
            # Crear array NumPy para cálculos eficientes
            amounts_np = np.array(spent_stats['total_amounts'], dtype=np.int64)
            
            # Calcular estadísticas básicas
            amount_mean = np.mean(amounts_np)
            amount_median = np.median(amounts_np)  # Mediana exacta
            amount_min = np.min(amounts_np)
            amount_max = np.max(amounts_np)
            
            f.write(f"\nAmount Statistics (Satoshis):\n")
            f.write(f"- Average: {amount_mean:.2f}\n")
            f.write(f"- Median: {amount_median}\n")
            f.write(f"- Min: {amount_min}\n")
            f.write(f"- Max: {amount_max}\n")
            
            # Calcular distribución
            f.write(f"\nAmount Distribution:\n")
            amount_dist = count_in_bins(amounts_np, amount_bins)
            for label, count in sorted(amount_dist.items()):
                f.write(f"{label}: {count}\n")
            
            # Liberar memoria
            del amounts_np
            gc.collect()
        
        # Estadísticas de locking script para UTXOs gastados
        if spent_stats['total_locking_sizes']:
            # Crear array NumPy para cálculos eficientes
            locking_np = np.array(spent_stats['total_locking_sizes'], dtype=np.int16)
            
            # Calcular estadísticas básicas
            locking_mean = np.mean(locking_np)
            locking_median = np.median(locking_np)  # Mediana exacta
            locking_min = np.min(locking_np)
            locking_max = np.max(locking_np)
            
            f.write(f"\nLocking Script Size Statistics (bytes):\n")
            f.write(f"- Average: {locking_mean:.2f}\n")
            f.write(f"- Median: {locking_median}\n")
            f.write(f"- Min: {locking_min}\n")
            f.write(f"- Max: {locking_max}\n")
            
            # Calcular distribución
            f.write(f"\nLocking Script Size Distribution:\n")
            locking_dist = count_in_bins(locking_np, script_bins)
            for label, count in sorted(locking_dist.items()):
                f.write(f"{label}: {count}\n")
            
            # Liberar memoria
            del locking_np
            gc.collect()
    
    # Forzar limpieza adicional de memoria
    print(f"Finished writing spent UTXO statistics to {filename}")
    gc.collect()

def write_unspent_total_stats(output_dir, unspent_stats):
    """Escribe las estadísticas totales de UTXOs no gastados a un archivo"""
    with open(f"{output_dir}unspent_utxo_statistics.txt", "w") as f:
        f.write("======== UNSPENT UTXO ANALYSIS SUMMARY ========\n\n")
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

def write_total_comparative_stats(output_dir):
    """Escribe las estadísticas comparativas totales entre UTXOs gastados y no gastados"""
    # Leer las estadísticas previamente calculadas
    spent_amounts = []
    spent_locking_sizes = []
    unspent_amounts = []
    unspent_locking_sizes = []
    
    # Leer datos mínimos necesarios para la comparación
    with open(f"{output_dir}spent_utxo_statistics.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Amount Statistics (Satoshis):" in line:
                spent_avg_line = lines[i + 1]
                spent_amount_avg = float(spent_avg_line.split(": ")[1].split(" ")[0])
                spent_median_line = lines[i + 2]
                spent_amount_median = float(spent_median_line.split(": ")[1].split(" ")[0])
            elif "Locking Script Size Statistics (bytes):" in line:
                spent_lock_avg_line = lines[i + 1]
                spent_locking_avg = float(spent_lock_avg_line.split(": ")[1].split(" ")[0])
                spent_lock_median_line = lines[i + 2]
                spent_locking_median = float(spent_lock_median_line.split(": ")[1].split(" ")[0])
                
    with open(f"{output_dir}unspent_utxo_statistics.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Amount Statistics (Satoshis):" in line:
                unspent_avg_line = lines[i + 1]
                unspent_amount_avg = float(unspent_avg_line.split(": ")[1].split(" ")[0])
                unspent_median_line = lines[i + 2]
                unspent_amount_median = float(unspent_median_line.split(": ")[1].split(" ")[0])
            elif "Locking Script Size Statistics (bytes):" in line:
                unspent_lock_avg_line = lines[i + 1]
                unspent_locking_avg = float(unspent_lock_avg_line.split(": ")[1].split(" ")[0])
                unspent_lock_median_line = lines[i + 2]
                unspent_locking_median = float(unspent_lock_median_line.split(": ")[1].split(" ")[0])
    
    # Escribir estadísticas comparativas
    with open(f"{output_dir}comparative_utxo_statistics.txt", "w") as f:
        f.write("======== COMPARATIVE UTXO ANALYSIS ========\n\n")
        
        # Montos: Comparativa entre UTXOs gastados vs. no gastados
        f.write("\nAmount Comparison (Spent vs. Unspent):\n")
        f.write(f"- Average Amount (Spent): {spent_amount_avg:.2f} Satoshis\n")
        f.write(f"- Average Amount (Unspent): {unspent_amount_avg:.2f} Satoshis\n")
        f.write(f"- Ratio (Unspent/Spent): {unspent_amount_avg/spent_amount_avg:.2f}x\n")
        f.write(f"- Median Amount (Spent): {spent_amount_median} Satoshis\n")
        f.write(f"- Median Amount (Unspent): {unspent_amount_median} Satoshis\n")
        f.write(f"- Ratio (Unspent/Spent): {unspent_amount_median/spent_amount_median:.2f}x\n")
        
        # Locking Script: Comparativa entre UTXOs gastados vs. no gastados
        f.write("\nLocking Script Size Comparison (Spent vs. Unspent):\n")
        f.write(f"- Average Size (Spent): {spent_locking_avg:.2f} bytes\n")
        f.write(f"- Average Size (Unspent): {unspent_locking_avg:.2f} bytes\n")
        f.write(f"- Ratio (Unspent/Spent): {unspent_locking_avg/spent_locking_avg:.2f}x\n")
        f.write(f"- Median Size (Spent): {spent_locking_median} bytes\n")
        f.write(f"- Median Size (Unspent): {unspent_locking_median} bytes\n")
        f.write(f"- Ratio (Unspent/Spent): {unspent_locking_median/spent_locking_median:.2f}x\n")
    
    # Generar gráficos comparativos
    # Para esto necesitaríamos cargar los datos nuevamente, pero esto consume mucha memoria
    # Una alternativa es generar estos gráficos en un paso separado si realmente son necesarios

def write_segment_spent_stats(output_dir, segment, stats):
    """Escribe las estadísticas de un segmento específico de UTXOs gastados"""
    with open(f"{output_dir}segment_{segment}_spent_statistics.txt", "w") as f:
        f.write(f"======== SEGMENT {segment} - {segment + segment_size - 1} SPENT UTXO ANALYSIS ========\n\n")
        
        spent_count = stats['spent'] + stats.get('lifespan_0', 0)
        f.write(f"Total Spent UTXOs in segment: {spent_count}\n")
        f.write(f"- With lifespan > 0: {stats['spent']}\n")
        f.write(f"- With lifespan = 0: {stats['lifespan_0']}\n")
        
        # Estadísticas de lifespan
        if stats['lifespans']:
            lifespan_stats = calc_stats(stats['lifespans'])
            f.write(f"\nLifespan Statistics (blocks):\n")
            f.write(f"- Average: {lifespan_stats['mean']:.2f}\n")
            f.write(f"- Median: {lifespan_stats['median']}\n")
            f.write(f"- Min: {lifespan_stats['min']}\n")
            f.write(f"- Max: {lifespan_stats['max']}\n")
            
            if len(stats['lifespans']) > 10:  # Solo si hay suficientes datos
                lifespans_series = pd.Series(stats['lifespans'])
                lifespan_distribution = pd.cut(lifespans_series, bins=lifespan_bins).value_counts().sort_index()
                f.write(f"\nLifespan Distribution:\n")
                f.write(str(lifespan_distribution) + "\n")
        
        # Estadísticas de montos
        if stats['amounts']:
            amount_stats = calc_stats(stats['amounts'])
            f.write(f"\nAmount Statistics (Satoshis):\n")
            f.write(f"- Average: {amount_stats['mean']:.2f}\n")
            f.write(f"- Median: {amount_stats['median']}\n")
            f.write(f"- Min: {amount_stats['min']}\n")
            f.write(f"- Max: {amount_stats['max']}\n")
            
            if len(stats['amounts']) > 10:  # Solo si hay suficientes datos
                amounts_series = pd.Series(stats['amounts'])
                amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
                f.write(f"\nAmount Distribution:\n")
                f.write(str(amount_distribution) + "\n")
        
        # Estadísticas de locking script
        if stats['locking_sizes']:
            locking_stats = calc_stats(stats['locking_sizes'])
            f.write(f"\nLocking Script Size Statistics (bytes):\n")
            f.write(f"- Average: {locking_stats['mean']:.2f}\n")
            f.write(f"- Median: {locking_stats['median']}\n")
            f.write(f"- Min: {locking_stats['min']}\n")
            f.write(f"- Max: {locking_stats['max']}\n")
            
            if len(stats['locking_sizes']) > 10:  # Solo si hay suficientes datos
                locking_series = pd.Series(stats['locking_sizes'])
                locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
                f.write(f"\nLocking Script Size Distribution:\n")
                f.write(str(locking_distribution) + "\n")

def write_segment_unspent_stats(output_dir, segment, stats):
    """Escribe las estadísticas de un segmento específico de UTXOs no gastados"""
    with open(f"{output_dir}segment_{segment}_unspent_statistics.txt", "w") as f:
        f.write(f"======== SEGMENT {segment} - {segment + segment_size - 1} UNSPENT UTXO ANALYSIS ========\n\n")
        
        f.write(f"Total Unspent UTXOs in segment: {stats['count']}\n")
        
        # Estadísticas de montos
        if stats['amounts']:
            amount_stats = calc_stats(stats['amounts'])
            f.write(f"\nAmount Statistics (Satoshis):\n")
            f.write(f"- Average: {amount_stats['mean']:.2f}\n")
            f.write(f"- Median: {amount_stats['median']}\n")
            f.write(f"- Min: {amount_stats['min']}\n")
            f.write(f"- Max: {amount_stats['max']}\n")
            
            if len(stats['amounts']) > 10:  # Solo si hay suficientes datos
                amounts_series = pd.Series(stats['amounts'])
                amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
                f.write(f"\nAmount Distribution:\n")
                f.write(str(amount_distribution) + "\n")
        
        # Estadísticas de locking script
        if stats['locking_sizes']:
            locking_stats = calc_stats(stats['locking_sizes'])
            f.write(f"\nLocking Script Size Statistics (bytes):\n")
            f.write(f"- Average: {locking_stats['mean']:.2f}\n")
            f.write(f"- Median: {locking_stats['median']}\n")
            f.write(f"- Min: {locking_stats['min']}\n")
            f.write(f"- Max: {locking_stats['max']}\n")
            
            if len(stats['locking_sizes']) > 10:  # Solo si hay suficientes datos
                locking_series = pd.Series(stats['locking_sizes'])
                locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
                f.write(f"\nLocking Script Size Distribution:\n")
                f.write(str(locking_distribution) + "\n")

def write_segment_comparative_stats(output_dir, all_segments):
    """Escribe las estadísticas comparativas para cada segmento"""
    with open(f"{output_dir}segment_comparative_statistics.txt", "w") as f:
        f.write("======== SEGMENT COMPARATIVE ANALYSIS ========\n\n")
        
        for segment in sorted(all_segments):
            spent_file = f"{output_dir}segment_{segment}_spent_statistics.txt"
            unspent_file = f"{output_dir}segment_{segment}_unspent_statistics.txt"
            
            if not (os.path.exists(spent_file) and os.path.exists(unspent_file)):
                continue
                
            spent_amount_avg = None
            spent_amount_median = None
            unspent_amount_avg = None
            unspent_amount_median = None
            
            # Leer datos del segmento de UTXOs gastados
            if os.path.exists(spent_file):
                with open(spent_file, "r") as sf:
                    lines = sf.readlines()
                    for i, line in enumerate(lines):
                        if "Amount Statistics (Satoshis):" in line:
                            try:
                                spent_avg_line = lines[i + 1]
                                spent_amount_avg = float(spent_avg_line.split(": ")[1].split(" ")[0])
                                spent_median_line = lines[i + 2]
                                spent_amount_median = float(spent_median_line.split(": ")[1].split(" ")[0])
                            except (IndexError, ValueError):
                                pass
            
            # Leer datos del segmento de UTXOs no gastados
            if os.path.exists(unspent_file):
                with open(unspent_file, "r") as uf:
                    lines = uf.readlines()
                    for i, line in enumerate(lines):
                        if "Amount Statistics (Satoshis):" in line:
                            try:
                                unspent_avg_line = lines[i + 1]
                                unspent_amount_avg = float(unspent_avg_line.split(": ")[1].split(" ")[0])
                                unspent_median_line = lines[i + 2]
                                unspent_amount_median = float(unspent_median_line.split(": ")[1].split(" ")[0])
                            except (IndexError, ValueError):
                                pass
            
            # Escribir comparativa del segmento
            if spent_amount_avg and unspent_amount_avg:
                f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
                f.write(f"Amount Comparison (Spent vs. Unspent):\n")
                f.write(f"- Average Amount (Spent): {spent_amount_avg:.2f} Satoshis\n")
                f.write(f"- Average Amount (Unspent): {unspent_amount_avg:.2f} Satoshis\n")
                f.write(f"- Ratio (Unspent/Spent): {unspent_amount_avg/spent_amount_avg:.2f}x\n")
                
                if spent_amount_median and unspent_amount_median:
                    f.write(f"- Median Amount (Spent): {spent_amount_median} Satoshis\n")
                    f.write(f"- Median Amount (Unspent): {unspent_amount_median} Satoshis\n")
                    f.write(f"- Ratio (Unspent/Spent): {unspent_amount_median/spent_amount_median:.2f}x\n")

def main():
    print("Starting UTXO set analysis...")
    
    # # Paso 1: Procesar datos para estadísticas totales de UTXOs gastados
    # print("Step 1: Processing total statistics for spent UTXOs...")
    # spent_total_stats = process_total_spent_stats()
    
    # # Generar gráficos y escribir estadísticas para UTXOs gastados totales
    # # print("  Generating plots and writing statistics for spent UTXOs...")
    # # generate_plots({
    # #     'lifespans': spent_total_stats['total_lifespans'],
    # #     'amounts': spent_total_stats['total_amounts'],
    # #     'locking_sizes': spent_total_stats['total_locking_sizes']
    # # }, output_dir, "spent_overall", "skyblue", "Spent TXOs")
    
    # write_spent_total_stats(output_dir, spent_total_stats)
    
    # # Liberar memoria - ya no necesitamos obtener all_segments de aquí
    # del spent_total_stats
    # gc.collect()
    
    # Paso 2: Procesar datos para estadísticas totales de UTXOs no gastados
    print("Step 2: Processing total statistics for unspent UTXOs...")
    unspent_total_stats = process_total_unspent_stats()
    
    # # Generar gráficos y escribir estadísticas para UTXOs no gastados totales
    # print("  Generating plots and writing statistics for unspent UTXOs...")
    # generate_plots({
    #     'amounts': unspent_total_stats['total_amounts'],
    #     'locking_sizes': unspent_total_stats['total_locking_sizes']
    # }, output_dir, "unspent_overall", "lightgreen", "Unspent TXOs")
    
    write_unspent_total_stats(output_dir, unspent_total_stats)
    
    # Liberar memoria
    del unspent_total_stats
    gc.collect()
    
    # # Escribir estadísticas comparativas totales
    # print("Step 3: Writing total comparative statistics...")
    # write_total_comparative_stats(output_dir)
    
    # Paso 4: Procesar cada segmento para UTXOs gastados
    # Ahora usamos los segmentos precalculados
    print(f"Step 4: Processing {len(all_segments)} segments for spent UTXOs...")
    for segment in all_segments:
        print(f"  Processing segment {segment} for spent UTXOs...")
        segment_stats = process_segment_spent_stats(segment)
        
        if segment_stats['lifespans'] or segment_stats['amounts']:
            # Generar gráficos y escribir estadísticas para este segmento
            generate_segment_plots(segment, segment_stats, output_dir, "spent")
            write_segment_spent_stats(output_dir, segment, segment_stats)
        
        # Liberar memoria
        del segment_stats
        gc.collect()
    
    # Paso 5: Procesar cada segmento para UTXOs no gastados
    # También usamos los segmentos precalculados
    print(f"Step 5: Processing {len(all_segments)} segments for unspent UTXOs...")
    for segment in all_segments:
        print(f"  Processing segment {segment} for unspent UTXOs...")
        segment_stats = process_segment_unspent_stats(segment)
        
        if segment_stats['amounts']:
            # Generar gráficos y escribir estadísticas para este segmento
            generate_segment_plots(segment, segment_stats, output_dir, "unspent")
            write_segment_unspent_stats(output_dir, segment, segment_stats)
        
        # Liberar memoria
        del segment_stats
        gc.collect()
    
    # # Paso 6: Escribir estadísticas comparativas por segmento
    # print("Step 6: Writing comparative statistics for each segment...")
    # write_segment_comparative_stats(output_dir, all_segments)
    
    # print("Analysis complete. All results saved to:", output_dir)


if __name__ == "__main__":
    main()






# def write_statistics(output_dir, spent_stats, unspent_stats):
#     """Escribir todas las estadísticas al archivo de salida"""
#     with open(f"{output_dir}utxo_statistics.txt", "w") as f:
#         # RESUMEN GENERAL
#         f.write("======== UTXO SET ANALYSIS SUMMARY ========\n\n")
        
#         # Estadísticas generales de UTXOs gastados
#         spent_total = len(spent_stats['total_lifespans'])
#         spent_zero = spent_stats['total_lifespans_zero']
#         spent_all = spent_total + spent_zero
        
#         f.write("=== SPENT UTXOs ===\n")
#         f.write(f"Total Spent UTXOs: {spent_all}\n")
#         f.write(f"- With lifespan > 0: {spent_total}\n")
#         f.write(f"- With lifespan = 0: {spent_zero}\n")
        
#         # Estadísticas de lifespan para UTXOs gastados
#         if spent_total > 0:
#             lifespan_stats = calc_stats(spent_stats['total_lifespans'])
#             f.write(f"\nLifespan Statistics (blocks):\n")
#             f.write(f"- Average: {lifespan_stats['mean']:.2f}\n")
#             f.write(f"- Median: {lifespan_stats['median']}\n")
#             f.write(f"- Min: {lifespan_stats['min']}\n")
#             f.write(f"- Max: {lifespan_stats['max']}\n")
            
#             # Distribución de lifespan
#             lifespans_series = pd.Series(spent_stats['total_lifespans'])
#             lifespan_distribution = pd.cut(lifespans_series, bins=lifespan_bins).value_counts().sort_index()
#             f.write(f"\nLifespan Distribution:\n")
#             f.write(str(lifespan_distribution) + "\n")
        
#         # Estadísticas de montos para UTXOs gastados
#         if spent_stats['total_amounts']:
#             amount_stats = calc_stats(spent_stats['total_amounts'])
#             f.write(f"\nAmount Statistics (Satoshis):\n")
#             f.write(f"- Average: {amount_stats['mean']:.2f}\n")
#             f.write(f"- Median: {amount_stats['median']}\n")
#             f.write(f"- Min: {amount_stats['min']}\n")
#             f.write(f"- Max: {amount_stats['max']}\n")
            
#             # Distribución de montos
#             amounts_series = pd.Series(spent_stats['total_amounts'])
#             amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
#             f.write(f"\nAmount Distribution:\n")
#             f.write(str(amount_distribution) + "\n")
        
#         # Estadísticas de locking script para UTXOs gastados
#         if spent_stats['total_locking_sizes']:
#             locking_stats = calc_stats(spent_stats['total_locking_sizes'])
#             f.write(f"\nLocking Script Size Statistics (bytes):\n")
#             f.write(f"- Average: {locking_stats['mean']:.2f}\n")
#             f.write(f"- Median: {locking_stats['median']}\n")
#             f.write(f"- Min: {locking_stats['min']}\n")
#             f.write(f"- Max: {locking_stats['max']}\n")
            
#             # Distribución de tamaños de locking script
#             locking_series = pd.Series(spent_stats['total_locking_sizes'])
#             locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
#             f.write(f"\nLocking Script Size Distribution:\n")
#             f.write(str(locking_distribution) + "\n")
        
#         # Estadísticas generales de UTXOs no gastados
#         f.write("\n\n=== UNSPENT UTXOs ===\n")
#         f.write(f"Total Unspent UTXOs: {unspent_stats['total_count']}\n")
        
#         # Estadísticas de montos para UTXOs no gastados
#         if unspent_stats['total_amounts']:
#             amount_stats = calc_stats(unspent_stats['total_amounts'])
#             f.write(f"\nAmount Statistics (Satoshis):\n")
#             f.write(f"- Average: {amount_stats['mean']:.2f}\n")
#             f.write(f"- Median: {amount_stats['median']}\n")
#             f.write(f"- Min: {amount_stats['min']}\n")
#             f.write(f"- Max: {amount_stats['max']}\n")
            
#             # Distribución de montos
#             amounts_series = pd.Series(unspent_stats['total_amounts'])
#             amount_distribution = pd.cut(amounts_series, bins=amount_bins).value_counts().sort_index()
#             f.write(f"\nAmount Distribution:\n")
#             f.write(str(amount_distribution) + "\n")
        
#         # Estadísticas de locking script para UTXOs no gastados
#         if unspent_stats['total_locking_sizes']:
#             locking_stats = calc_stats(unspent_stats['total_locking_sizes'])
#             f.write(f"\nLocking Script Size Statistics (bytes):\n")
#             f.write(f"- Average: {locking_stats['mean']:.2f}\n")
#             f.write(f"- Median: {locking_stats['median']}\n")
#             f.write(f"- Min: {locking_stats['min']}\n")
#             f.write(f"- Max: {locking_stats['max']}\n")
            
#             # Distribución de tamaños de locking script
#             locking_series = pd.Series(unspent_stats['total_locking_sizes'])
#             locking_distribution = pd.cut(locking_series, bins=script_bins).value_counts().sort_index()
#             f.write(f"\nLocking Script Size Distribution:\n")
#             f.write(str(locking_distribution) + "\n")
        
#         # ANÁLISIS COMPARATIVO
#         f.write("\n\n======== COMPARATIVE ANALYSIS ========\n")
        
#         # Montos: Comparativa entre UTXOs gastados vs. no gastados
#         if spent_stats['total_amounts'] and unspent_stats['total_amounts']:
#             spent_amount_avg = np.mean(np.array(spent_stats['total_amounts']))
#             unspent_amount_avg = np.mean(np.array(unspent_stats['total_amounts']))
#             spent_amount_median = np.median(np.array(spent_stats['total_amounts']))
#             unspent_amount_median = np.median(np.array(unspent_stats['total_amounts']))
            
#             f.write("\nAmount Comparison (Spent vs. Unspent):\n")
#             f.write(f"- Average Amount (Spent): {spent_amount_avg:.2f} Satoshis\n")
#             f.write(f"- Average Amount (Unspent): {unspent_amount_avg:.2f} Satoshis\n")
#             f.write(f"- Ratio (Unspent/Spent): {unspent_amount_avg/spent_amount_avg:.2f}x\n")
#             f.write(f"- Median Amount (Spent): {spent_amount_median} Satoshis\n")
#             f.write(f"- Median Amount (Unspent): {unspent_amount_median} Satoshis\n")
#             f.write(f"- Ratio (Unspent/Spent): {unspent_amount_median/spent_amount_median:.2f}x\n")
        
#         # Locking Script: Comparativa entre UTXOs gastados vs. no gastados
#         if spent_stats['total_locking_sizes'] and unspent_stats['total_locking_sizes']:
#             spent_locking_avg = np.mean(np.array(spent_stats['total_locking_sizes']))
#             unspent_locking_avg = np.mean(np.array(unspent_stats['total_locking_sizes']))
#             spent_locking_median = np.median(np.array(spent_stats['total_locking_sizes']))
#             unspent_locking_median = np.median(np.array(unspent_stats['total_locking_sizes']))
            
#             f.write("\nLocking Script Size Comparison (Spent vs. Unspent):\n")
#             f.write(f"- Average Size (Spent): {spent_locking_avg:.2f} bytes\n")
#             f.write(f"- Average Size (Unspent): {unspent_locking_avg:.2f} bytes\n")
#             f.write(f"- Ratio (Unspent/Spent): {unspent_locking_avg/spent_locking_avg:.2f}x\n")
#             f.write(f"- Median Size (Spent): {spent_locking_median} bytes\n")
#             f.write(f"- Median Size (Unspent): {unspent_locking_median} bytes\n")
#             f.write(f"- Ratio (Unspent/Spent): {unspent_locking_median/spent_locking_median:.2f}x\n")
        
#         # ESTADÍSTICAS POR SEGMENTO
#         f.write("\n\n======== SEGMENTED ANALYSIS ========\n")
        
#         # Unificar segmentos de ambos conjuntos para análisis
#         all_segments = set(spent_stats['segment_stats'].keys()) | set(unspent_stats['segment_stats'].keys())
        
#         for segment in sorted(all_segments):
#             f.write(f"\n=== Segment {segment} - {segment + segment_size - 1} ===\n")
            
#             # Estadísticas del segmento para UTXOs gastados
#             if segment in spent_stats['segment_stats']:
#                 seg_data = spent_stats['segment_stats'][segment]
#                 spent_count = seg_data['spent'] + seg_data.get('lifespan_0', 0)
#                 f.write(f"Spent UTXOs: {spent_count}\n")
                
#                 if seg_data['lifespans']:
#                     lifespan_stats = calc_stats(seg_data['lifespans'])
#                     f.write(f"- Average Lifespan: {lifespan_stats['mean']:.2f} blocks\n")
#                     f.write(f"- Median Lifespan: {lifespan_stats['median']} blocks\n")
                
#                 if seg_data['amounts']:
#                     amount_stats = calc_stats(seg_data['amounts'])
#                     f.write(f"- Average Amount (Spent): {amount_stats['mean']:.2f} Satoshis\n")
#             else:
#                 f.write("No Spent UTXOs in this segment.\n")
            
#             # Estadísticas del segmento para UTXOs no gastados
#             if segment in unspent_stats['segment_stats']:
#                 seg_data = unspent_stats['segment_stats'][segment]
#                 f.write(f"Unspent UTXOs: {seg_data['count']}\n")
                
#                 if seg_data['amounts']:
#                     amount_stats = calc_stats(seg_data['amounts'])
#                     f.write(f"- Average Amount (Unspent): {amount_stats['mean']:.2f} Satoshis\n")
#             else:
#                 f.write("No Unspent UTXOs in this segment.\n")
            
#             # Análisis comparativo del segmento (si hay datos de ambos tipos)
#             if (segment in spent_stats['segment_stats'] and 
#                 segment in unspent_stats['segment_stats'] and
#                 spent_stats['segment_stats'][segment]['amounts'] and
#                 unspent_stats['segment_stats'][segment]['amounts']):
                
#                 spent_avg = np.mean(np.array(spent_stats['segment_stats'][segment]['amounts']))
#                 unspent_avg = np.mean(np.array(unspent_stats['segment_stats'][segment]['amounts']))
                
#                 f.write(f"\nSegment Comparison:\n")
#                 f.write(f"- Amount Ratio (Unspent/Spent): {unspent_avg/spent_avg:.2f}x\n")
