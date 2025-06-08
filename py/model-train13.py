import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# import onnx
from hummingbird.ml import convert

import warnings
warnings.filterwarnings('ignore')

import os

class UTXOStorageClassifier:
    """
    Sistema para clasificar UTXOs de Bitcoin Cash en hot/cold storage
    basado en la probabilidad de gasto en los próximos 1000 bloques.
    """
    
    def __init__(self, prediction_horizon: int = 1000, max_block_height: int = 789999):
        self.prediction_horizon = prediction_horizon
        self.max_block_height = max_block_height
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.thresholds = {'hot_threshold': 0.7, 'cold_threshold': 0.3}
        self.data_stats = {}
        
    # def load_and_prepare_data(self, parquet_files: List[str]) -> pd.DataFrame:
    #     """
    #     Carga y prepara los datos desde archivos parquet
    #     """
    #     print("📥 Cargando datos...")
        
    #     # Cargar datos en chunks para manejar el volumen
    #     df_chunks = []
    #     for file in parquet_files:
    #         print(f"  Cargando {file}...")
    #         chunk = pd.read_parquet(file)
    #         df_chunks.append(chunk)
    #         print(f"  Cargado {len(chunk):,} UTXOs")
        
    #     print("📊 Concatenando chunks...")
    #     df = pd.concat(df_chunks, ignore_index=True)
    #     print(f"📊 Datos cargados: {len(df):,} UTXOs")
        
        # return self._clean_and_filter_data(df)
        
    def load_and_prepare_data(self, parquet_files: List[str]) -> pd.DataFrame:
        """
        Carga y filtra los datos desde archivos parquet uno por uno sin consumir memoria excesiva.
        Usa _clean_and_filter_data() en cada archivo individualmente.
        """
        print("📥 Cargando y limpiando archivos parquet...")

        df_chunks = []
        total_loaded = 0

        for file in parquet_files:
            print(f"  → Procesando {file}...")
            try:
                chunk = pd.read_parquet(file)
                print(f"    Cargado {len(chunk):,} UTXOs")
                cleaned_chunk = self._clean_and_filter_data(chunk)

                if not cleaned_chunk.empty:
                    df_chunks.append(cleaned_chunk)
                    total_loaded += len(cleaned_chunk)
                    print(f"    ✔️  Retenidos: {len(cleaned_chunk):,}")
                else:
                    print("    ⚠️  Chunk vacío tras limpieza.")

            except Exception as e:
                print(f"    ❌ Error procesando {file}: {e}")

        print(f"📊 Total cargado: {total_loaded:,} UTXOs")
        df_final = pd.concat(df_chunks, ignore_index=True)
        print(f"✅ Dataset final: {len(df_final):,} filas")
        print(f"🎯 Distribución target: {df_final['target'].value_counts().to_dict()}")

        # Guardar estadísticas básicas
        self.data_stats = {
            'total_utxos': len(df_final),
            'spent_utxos': df_final['event'].sum(),
            'unspent_utxos': (~df_final['event']).sum(),
            'value_range': (df_final['value'].min(), df_final['value'].max()),
            'block_range': (df_final['creation_block'].min(), df_final['creation_block'].max())
        }

        return df_final


    def _clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y filtra los datos para el entrenamiento
        """
        print("🧹 Limpiando datos...")
        
        initial_count = len(df)
        
        # Filtros de calidad de datos
        df = df.dropna(subset=['creation_block', 'value'])
        df = df[df['value'] > 0]  # Eliminar valores <= 0
        df = df[~df['op_return']]  # Eliminar OP_RETURN (no gastables)
        
        # Para UTXOs gastados, calcular si se gastaron dentro del horizonte
        df['target'] = False
        spent_mask = df['event'] == True
        df.loc[spent_mask, 'target'] = df.loc[spent_mask, 'duration'] <= self.prediction_horizon
        
        # Para UTXOs no gastados, si han vivido más del horizonte, son negativos
        unspent_mask = df['event'] == False
        long_lived_mask = df['duration'] > self.prediction_horizon
        df.loc[unspent_mask & long_lived_mask, 'target'] = False

        # Filtrar UTXOs muy recientes (menos del horizonte) para tener labels confiables
        df = df[spent_mask | (df['duration'] >= self.prediction_horizon)]

        # 🔻 Mantener solo un 10% aleatorio
        df = df.sample(frac=0.1, random_state=42)

        print(f"📊 Después de filtros: {len(df):,} UTXOs ({len(df)/initial_count:.1%} retenido)")
        print(f"🎯 Distribución target: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características predictivas basadas en patrones conocidos de Bitcoin Cash
        """
        print("⚙️ Creando características...")
        
        df = df.copy()
        
        # === CARACTERÍSTICAS DE VALOR ===
        df['log_value'] = np.log10(df['value'] + 1)
        df['value_satoshi_class'] = pd.cut(df['value'], 
                                         bins=[0, 546, 1000, 10000, 100000, 1000000, float('inf')],
                                         labels=['dust', 'micro', 'small', 'medium', 'large', 'whale'])
        
        # === CARACTERÍSTICAS DE SCRIPTS ===
        df['total_script_size'] = df['locking_script_size'] + df['unlocking_script_size'].fillna(0)
        df['script_efficiency'] = df['value'] / (df['total_script_size'] + 1)
        
        # === CARACTERÍSTICAS TEMPORALES ===
        # Hora del día (aproximada por posición en bloque)
        df['block_time_proxy'] = df['creation_block'] % 144  # ~144 bloques por día
        df['creation_epoch'] = df['creation_block'] // 100_000

        df['is_coinbase'] = df['tx_coinbase'].astype(int)
        
        # === CARACTERÍSTICAS DE CONTEXTO ===
        # Densidad de UTXOs en bloques cercanos
        df['block_density'] = df.groupby('creation_block')['creation_block'].transform('count')
        
        # Percentiles de valor por bloque
        df['value_percentile_in_block'] = df.groupby('creation_block')['value'].rank(pct=True)
        
        # === PATRONES DE COMPORTAMIENTO ===
        # UTXOs pequeños tienden a gastarse rápido (cambio, micropagos)
        df['is_likely_change'] = (df['value'] < 10000) & (df['value'] > 546)
        
        # UTXOs muy grandes tienden a mantenerse (ahorros)
        df['is_likely_savings'] = df['value'] > 100000000  # > 1 BCH
        
        # Coinbase rewards tienden a mantenerse por maduración
        df['coinbase_maturity_factor'] = df['is_coinbase'] * np.log10(df['value'] + 1)
        
        # One-hot encoding para variables categóricas
        df = pd.get_dummies(df, columns=['value_satoshi_class'], prefix='value_class')
        
        # Seleccionar características para el modelo
        feature_cols = [
            'log_value', 'total_script_size', 'script_efficiency',
            'block_time_proxy', 'creation_epoch', 'is_coinbase', 'block_density',
            'value_percentile_in_block', 'is_likely_change', 
            'is_likely_savings', 'coinbase_maturity_factor'
        ]
        
        # Añadir columnas one-hot
        feature_cols.extend([col for col in df.columns if col.startswith('value_class_')])
        
        self.feature_columns = feature_cols
        print(self.feature_columns)  # Esta lista es la única verdad
        print(len(self.feature_columns))  # Debería dar 17        
        print(f"✅ {len(feature_cols)} características creadas")
        return df
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analiza patrones en los datos para entender el comportamiento de los UTXOs
        """
        print("🔍 Analizando patrones...")
        
        analysis = {}
        
        # # Análisis por rangos de valor
        # value_analysis = df.groupby(pd.cut(df['value'], 
        #                                  bins=[0, 546,    1000,   10000, 100000, 1000000, float('inf')],
        #                                  labels=['dust', 'micro', 'small', 'medium', 'large', 'whale'])).agg({
        #     'target': ['count', 'mean'],
        #     'duration': 'median'
        # }).round(3)
        
        # Análisis por rangos de valor (en satoshis)
        value_analysis = df.groupby(pd.cut(df['value'], 
            bins=[
                0,               # 
                1000,            # dust   (< 0.00001 BCH =      $0.004)
                100_000,         # micro  (< 0.001 BCH   =      $0.4) 
                1_000_000,       # small  (< 0.01 BCH    =      $4)
                10_000_000,      # medium (< 0.1 BCH     =     $40)
                1_000_000_000,   # large  (< 10 BCH      =  $4,000)
                10_000_000_000,  # big    (< 100 BCH     = $400,000)
                float('inf')     # whale  (≥ 100 BCH     = $40,000)
            ],
            labels=['dust', 'micro', 'small', 'medium', 'large', 'big', 'whale']
        )).agg({
            'target': ['count', 'mean'],
            'duration': 'median'
        }).round(3)

        analysis['value_patterns'] = value_analysis
        
        # Análisis coinbase vs regular
        coinbase_analysis = df.groupby('tx_coinbase').agg({
            'target': ['count', 'mean'],
            'duration': 'median',
            'value': 'median'
        }).round(3)
        
        analysis['coinbase_patterns'] = coinbase_analysis
        
        # Correlaciones importantes
        corr_with_target = df[self.feature_columns + ['target']].corr()['target'].sort_values(key=abs, ascending=False)
        analysis['feature_correlations'] = corr_with_target[1:-1]  # Excluir autocorrelación
        
        print("📈 Análisis completado")
        return analysis
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """
        Entrena el modelo de clasificación
        """
        print("🎯 Entrenando modelo...")
        
        X = df[self.feature_columns]
        y = df['target']
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalización
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Probar múltiples modelos
        models = {
            # 'RandomForest': RandomForestClassifier(
            #     n_estimators=100, 
            #     max_depth=10, 
            #     min_samples_split=100,
            #     min_samples_leaf=50,
            #     random_state=42,
            #     n_jobs=-1
            # ),
            # 'GradientBoosting': GradientBoostingClassifier(
            #     n_estimators=100,
            #     max_depth=6,
            #     learning_rate=0.1,
            #     random_state=42
            # )
            'HistGradientBoosting': HistGradientBoostingClassifier(
                max_iter=100,         # igual a n_estimators
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                early_stopping=False  # opcional: desactiva si estás haciendo tu propia validación
            )            
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  Entrenando {name}...")
            
            # Entrenar
            if name == 'RandomForest':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Evaluar
            if name == 'RandomForest':
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Métricas
            cv_scores = cross_val_score(model, X_train_scaled if name != 'RandomForest' else X_train, 
                                      y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"    Test AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Seleccionar mejor modelo
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_auc_mean'])
        self.model = results[best_model_name]['model']
        
        print(f"✅ Mejor modelo: {best_model_name}")
        
        return results, X_test, y_test
    
    def calibrate_thresholds(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Calibra los umbrales de decisión basado en costos de negocio
        """
        print("⚖️ Calibrando umbrales...")
        
        # Obtener probabilidades
        if isinstance(self.model, RandomForestClassifier):
            probas = self.model.predict_proba(X_test)[:, 1]
        else:
            probas = self.model.predict_proba(self.scaler.transform(X_test))[:, 1]
        
        # Análisis de umbrales
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for thresh in thresholds:
            predictions = probas >= thresh
            
            # Métricas básicas
            tp = np.sum((predictions == 1) & (y_test == 1))
            fp = np.sum((predictions == 1) & (y_test == 0))
            tn = np.sum((predictions == 0) & (y_test == 0))
            fn = np.sum((predictions == 0) & (y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Costos de negocio aproximados
            # Costo de poner en cold lo que debería estar en hot (fn)
            # Costo de poner en hot lo que debería estar en cold (fp)
            cost_hot_to_cold = fn * 10  # Alto costo por pérdida de liquidez
            cost_cold_to_hot = fp * 1   # Menor costo por almacenamiento ineficiente
            total_cost = cost_hot_to_cold + cost_cold_to_hot
            
            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                'total_cost': total_cost,
                'hot_storage_rate': np.mean(predictions)
            })
        
        results_df = pd.DataFrame(results)
        
        # Umbrales recomendados
        best_f1_idx = results_df['f1'].idxmax()
        best_cost_idx = results_df['total_cost'].idxmin()
        
        recommendations = {
            'best_f1_threshold': results_df.loc[best_f1_idx, 'threshold'],
            'best_cost_threshold': results_df.loc[best_cost_idx, 'threshold'],
            'analysis': results_df
        }
        
        # Actualizar umbrales por defecto
        self.thresholds['hot_threshold'] = results_df.loc[best_cost_idx, 'threshold']
        self.thresholds['cold_threshold'] = 0.3
        
        print(f"🎯 Umbral recomendado (F1): {recommendations['best_f1_threshold']:.2f}")
        print(f"💰 Umbral recomendado (costo): {recommendations['best_cost_threshold']:.2f}")
        
        return recommendations
    
    def predict_storage_decision(self, utxos: pd.DataFrame) -> pd.DataFrame:
        """
        Predice decisiones de almacenamiento para nuevos UTXOs
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecutar train_model() primero.")
        
        # Preparar características
        utxos_processed = self.engineer_features(utxos)
        X = utxos_processed[self.feature_columns]
        
        # Obtener probabilidades
        if isinstance(self.model, RandomForestClassifier):
            probas = self.model.predict_proba(X)[:, 1]
        else:
            X_scaled = self.scaler.transform(X)
            probas = self.model.predict_proba(X_scaled)[:, 1]
        
        # Decisiones
        decisions = np.where(
            probas >= self.thresholds['hot_threshold'], 'hot_storage',
            np.where(probas <= self.thresholds['cold_threshold'], 'cold_storage', 'review')
        )
        
        # Resultado
        result = utxos.copy()
        result['spend_probability'] = probas
        result['storage_decision'] = decisions
        result['confidence'] = np.where(
            (probas >= 0.8) | (probas <= 0.2), 'high',
            np.where((probas >= 0.6) | (probas <= 0.4), 'medium', 'low')
        )
        
        return result
    
    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado (para uso en Python)
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'thresholds': self.thresholds,
            'prediction_horizon': self.prediction_horizon,
            'max_block_height': self.max_block_height,
            'data_stats': self.data_stats
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Modelo guardado en: {filepath}")
    
    def export_for_cpp(self, base_filepath: str):
        """
        Exporta el modelo en múltiples formatos para uso en C++
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        print("🔄 Exportando modelo para C++...")
        
        # 1. ONNX Format (Recomendado)
        self._export_to_onnx(f"{base_filepath}.onnx")
        
        # # 2. JSON Format (Alternativa simple)
        # self._export_to_json(f"{base_filepath}.json")
        
        # # 3. Custom Binary Format (Para máximo rendimiento)
        # self._export_to_binary(f"{base_filepath}.bin")
        
        # # 4. C++ Header (Para embedding directo)
        # self._export_to_cpp_header(f"{base_filepath}_model.h")
        
        # print("✅ Exportación completa para C++")
    
    def _export_to_onnx(self, filepath: str):
        """
        Exporta a formato ONNX (Open Neural Network Exchange)
        Compatible con C++ usando ONNX Runtime
        """
        try:
            from skl2onnx import to_onnx
            import numpy as np
            
            # Crear datos dummy para inferir tipos
            X_dummy = np.random.randn(1, len(self.feature_columns)).astype(np.float32)
            
            # Convertir a ONNX
            onnx_model = to_onnx(
                self.model, 
                X_dummy,
                target_opset=12,
                options={type(self.model): {'zipmap': False}}
            )
            
            # Guardar
            with open(filepath, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"📦 ONNX exportado: {filepath}")
            print("   Para C++: usar ONNX Runtime (https://onnxruntime.ai/)")
            
        except ImportError:
            print("⚠️  skl2onnx no disponible. Instalar con: pip install skl2onnx")
        except Exception as e:
            print(f"⚠️  Error exportando ONNX: {e}")
    
    def _export_to_json(self, filepath: str):
        """
        Exporta a formato JSON con estructura del Random Forest
        """
        import json
        
        if not hasattr(self.model, 'estimators_'):
            print("⚠️  Solo soportado para Random Forest")
            return
        
        # Extraer estructura del Random Forest
        forest_data = {
            'metadata': {
                'model_type': 'RandomForest',
                'n_estimators': self.model.n_estimators,
                'n_features': len(self.feature_columns),
                'feature_names': self.feature_columns,
                'thresholds': self.thresholds,
                'prediction_horizon': self.prediction_horizon,
                'max_block_height': self.max_block_height
            },
            'trees': []
        }
        
        # Exportar cada árbol
        for i, tree in enumerate(self.model.estimators_):
            tree_data = self._extract_tree_structure(tree.tree_)
            forest_data['trees'].append(tree_data)
            
            if (i + 1) % 10 == 0:
                print(f"   Procesando árbol {i+1}/{len(self.model.estimators_)}")
        
        # Guardar JSON
        with open(filepath, 'w') as f:
            json.dump(forest_data, f, indent=2)
        
        print(f"📦 JSON exportado: {filepath}")
        print("   Para C++: usar nlohmann/json o similar")
    
    def _extract_tree_structure(self, tree):
        """
        Extrae la estructura de un árbol de decisión
        """
        def recurse(node_id):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Nodo hoja
                return {
                    'type': 'leaf',
                    'value': float(tree.value[node_id][0][1] / tree.value[node_id][0].sum())  # probabilidad clase 1
                }
            else:
                # Nodo interno
                return {
                    'type': 'split',
                    'feature': int(tree.feature[node_id]),
                    'threshold': float(tree.threshold[node_id]),
                    'left': recurse(tree.children_left[node_id]),
                    'right': recurse(tree.children_right[node_id])
                }
        
        return recurse(0)
    
    def _export_to_binary(self, filepath: str):
        """
        Exporta a formato binario compacto para C++
        """
        import struct
        
        if not hasattr(self.model, 'estimators_'):
            print("⚠️  Solo soportado para Random Forest")
            return
        
        with open(filepath, 'wb') as f:
            # Header
            f.write(b'UTXORF')  # Magic number
            f.write(struct.pack('I', 1))  # Version
            f.write(struct.pack('I', self.model.n_estimators))
            f.write(struct.pack('I', len(self.feature_columns)))
            f.write(struct.pack('f', self.thresholds['hot_threshold']))
            f.write(struct.pack('f', self.thresholds['cold_threshold']))
            
            # Feature names (para debugging)
            for feature in self.feature_columns:
                name_bytes = feature.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
            
            # Trees
            for tree in self.model.estimators_:
                self._write_tree_binary(f, tree.tree_)
        
        print(f"📦 Binario exportado: {filepath}")
        print("   Para C++: implementar parser binario personalizado")
    
    def _write_tree_binary(self, f, tree):
        """
        Escribe un árbol en formato binario
        """
        import struct
        
        n_nodes = tree.node_count
        f.write(struct.pack('I', n_nodes))
        
        for i in range(n_nodes):
            # Tipo de nodo
            is_leaf = tree.children_left[i] == tree.children_right[i]
            f.write(struct.pack('B', 1 if is_leaf else 0))
            
            if is_leaf:
                # Valor de hoja
                prob = tree.value[i][0][1] / tree.value[i][0].sum()
                f.write(struct.pack('f', prob))
            else:
                # Nodo de división
                f.write(struct.pack('I', tree.feature[i]))
                f.write(struct.pack('f', tree.threshold[i]))
                f.write(struct.pack('I', tree.children_left[i]))
                f.write(struct.pack('I', tree.children_right[i]))
    
    def _export_to_cpp_header(self, filepath: str):
        """
        Exporta como header C++ con arrays embebidos
        """
        if not hasattr(self.model, 'estimators_'):
            print("⚠️  Solo soportado para Random Forest")
            return
        
        with open(filepath, 'w') as f:
            f.write(f"""#ifndef UTXO_MODEL_H
#define UTXO_MODEL_H

#include <vector>
#include <string>
#include <array>

namespace UTXOModel {{

// Metadata
const int N_ESTIMATORS = {self.model.n_estimators};
const int N_FEATURES = {len(self.feature_columns)};
const float HOT_THRESHOLD = {self.thresholds['hot_threshold']}f;
const float COLD_THRESHOLD = {self.thresholds['cold_threshold']}f;

// Feature names
const std::array<std::string, N_FEATURES> FEATURE_NAMES = {{
""")
            
            for i, feature in enumerate(self.feature_columns):
                f.write(f'    "{feature}"')
                if i < len(self.feature_columns) - 1:
                    f.write(',')
                f.write('\n')
            
            f.write("};\n\n")
            
            # Estructuras simplificadas de árboles (solo para árboles pequeños)
            if self.model.n_estimators <= 10:  # Solo para bosques pequeños
                f.write("// Tree structures (simplified)\n")
                for i, tree in enumerate(self.model.estimators_[:5]):  # Solo primeros 5 árboles
                    f.write(f"// Tree {i} - {tree.tree_.node_count} nodes\n")
                    f.write(f"const int TREE_{i}_NODES = {tree.tree_.node_count};\n")
            
            f.write(f"""
// Prediction function declaration
float predict_spend_probability(const std::vector<float>& features);

enum class StorageDecision {{
    HOT_STORAGE,
    COLD_STORAGE,
    REVIEW
}};

StorageDecision classify_utxo(const std::vector<float>& features);

}} // namespace UTXOModel

#endif // UTXO_MODEL_H
""")
        
        print(f"📦 C++ Header exportado: {filepath}")
        print("   Para C++: incluir header e implementar lógica de predicción")
    
    def generate_cpp_example(self, filepath: str):
        """
        Genera código C++ de ejemplo para usar el modelo
        """
        cpp_code = f"""#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "utxo_model.h"

// Ejemplo de implementación simple para Random Forest
class UTXOClassifier {{
private:
    // Simplificación: usar solo las primeras características más importantes
    std::vector<int> important_features = {{0, 1, 2, 3, 4}};  // Indices de features importantes
    
public:
    float predict_probability(const std::vector<float>& features) {{
        // Implementación simplificada basada en reglas heurísticas
        // derivadas del modelo entrenado
        
        float log_value = features[0];  // Asumiendo que log_value es feature 0
        float is_likely_change = features[1];  // feature 1
        float is_likely_savings = features[2];  // feature 2
        
        // Reglas simplificadas basadas en patrones del modelo
        float score = 0.0f;
        
        // Valor pequeño -> más probable gasto rápido
        if (log_value < 3.0f) score += 0.3f;
        else if (log_value > 8.0f) score -= 0.3f;
        
        // UTXOs de cambio tienden a gastarse rápido
        if (is_likely_change > 0.5f) score += 0.4f;
        
        // UTXOs de ahorro tienden a mantenerse
        if (is_likely_savings > 0.5f) score -= 0.5f;
        
        // Convertir score a probabilidad
        return 1.0f / (1.0f + std::exp(-score));
    }}
    
    UTXOModel::StorageDecision classify(const std::vector<float>& features) {{
        float prob = predict_probability(features);
        
        if (prob >= UTXOModel::HOT_THRESHOLD) {{
            return UTXOModel::StorageDecision::HOT_STORAGE;
        }} else if (prob <= UTXOModel::COLD_THRESHOLD) {{
            return UTXOModel::StorageDecision::COLD_STORAGE;
        }} else {{
            return UTXOModel::StorageDecision::REVIEW;
        }}
    }}
}};

// Ejemplo de uso
int main() {{
    UTXOClassifier classifier;
    
    // Ejemplo de UTXO
    std::vector<float> utxo_features = {{
        4.5f,  // log_value (pequeño)
        1.0f,  // is_likely_change
        0.0f,  // is_likely_savings
        // ... más features
    }};
    
    float prob = classifier.predict_probability(utxo_features);
    auto decision = classifier.classify(utxo_features);
    
    std::cout << "Probabilidad de gasto: " << prob << std::endl;
    std::cout << "Decisión: ";
    switch(decision) {{
        case UTXOModel::StorageDecision::HOT_STORAGE:
            std::cout << "HOT STORAGE";
            break;
        case UTXOModel::StorageDecision::COLD_STORAGE:
            std::cout << "COLD STORAGE";
            break;
        case UTXOModel::StorageDecision::REVIEW:
            std::cout << "REVIEW";
            break;
    }}
    std::cout << std::endl;
    
    return 0;
}}
"""
        
        with open(filepath, 'w') as f:
            f.write(cpp_code)
        
        print(f"📦 Ejemplo C++ generado: {filepath}")
        print("   Compilar con: g++ -std=c++17 -o utxo_classifier example.cpp")
    
    def load_model(self, filepath: str):
        """
        Carga un modelo previamente entrenado
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.thresholds = model_data['thresholds']
        self.prediction_horizon = model_data['prediction_horizon']
        self.max_block_height = model_data['max_block_height']
        self.data_stats = model_data['data_stats']
        
        print(f"📥 Modelo cargado desde: {filepath}")
        print(f"   - Horizon: {self.prediction_horizon} bloques")
        print(f"   - Max block: {self.max_block_height}")
        print(f"   - Features: {len(self.feature_columns)}")
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Obtiene la importancia de las características
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    


    def generate_report(self, analysis: Dict, model_results: Dict, threshold_analysis: Dict):
        """
        Genera un reporte comprehensivo del sistema
        """
        print("\n" + "="*60)
        print("📊 REPORTE DEL SISTEMA DE CLASIFICACIÓN UTXO")
        print("="*60)
        
        print("\n🔍 PATRONES IDENTIFICADOS:")
        print("\nPor rango de valor:")
        print(analysis['value_patterns'])
        
        print("\nCoinbase vs Regular:")
        print(analysis['coinbase_patterns'])
        
        print("\n📈 CORRELACIONES MÁS IMPORTANTES:")
        top_corr = analysis['feature_correlations'].head(10)
        for feature, corr in top_corr.items():
            print(f"  {feature}: {corr:.3f}")
        

        # # ------------------------------------------------------------------------
        # # Inspección rápida del primer classification_report
        # print("\n🧪 Dump del classification_report del primer modelo:")

        # first_key = list(model_results.keys())[0]
        # report = model_results[first_key]['classification_report']

        # print(f"Modelo: {first_key}")
        # print("Claves del classification_report:", list(report.keys()))

        # for k, v in report.items():
        #     print(f"  {repr(k)} ({type(k)}): {v if isinstance(v, dict) else '<métrica global>'}")
        # # ------------------------------------------------------------------------



        print("\n🎯 RENDIMIENTO DEL MODELO:")
        for name, results in model_results.items():
            print(f"\n{name}:")
            print(f"  CV AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}")
            print(f"  Test AUC: {results['test_auc']:.3f}")
            print(f"  Precision: {results['classification_report']['True']['precision']:.3f}")
            print(f"  Recall: {results['classification_report']['True']['recall']:.3f}")
        
        print(f"\n⚖️ UMBRALES CALIBRADOS:")
        print(f"  Hot Storage: >= {self.thresholds['hot_threshold']:.2f}")
        print(f"  Cold Storage: <= {self.thresholds['cold_threshold']:.2f}")
        print(f"  Review: entre {self.thresholds['cold_threshold']:.2f} y {self.thresholds['hot_threshold']:.2f}")
        
        print("\n✅ SISTEMA LISTO PARA PRODUCCIÓN")
        print("="*60)


def test_model_on_random_utxos(classifier, n_samples=100_000):
    """
    Toma una muestra aleatoria de todos los .parquet y evalúa el modelo actual.
    No guarda ni carga nada, usa classifier en memoria.
    """
    import glob
    from pathlib import Path

    print(f"\n🧪 Probando modelo sobre muestra aleatoria de {n_samples:,} UTXOs reales...")

    # === Recolectar todos los .parquet normalizados
    parquet_dir = Path("/home/fernando/dev/utxo-experiments/parquet_normalized")
    files = sorted(glob.glob(str(parquet_dir / "utxo-history-*.parquet")))

    # === Leer una fracción aleatoria de cada archivo
    dfs = []
    for path in files:
        try:
            df = pd.read_parquet(path)

            # Filtros básicos (opcional)
            df = df.dropna(subset=['creation_block', 'value'])
            df = df[df['value'] > 0]
            df = df[~df['op_return']]
            df['duration'] = df['duration'].fillna(0)

            # Tomar muestra parcial
            frac = n_samples / (len(files) * len(df))
            frac = min(frac, 0.1)
            sample = df.sample(frac=frac, random_state=42)
            dfs.append(sample)
        except Exception as e:
            print(f"⚠️  Error en {path}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    if len(df_all) > n_samples:
        df_all = df_all.sample(n=n_samples, random_state=42)

    # if 'duration' in df_all.columns:
    #     print(f"Borrando columna 'duration' para evitar confusiones...")
    #     del df_all['duration']

    print(f"✅ Muestra cargada: {len(df_all):,} UTXOs")

    # === Predecir
    df_all = df_all.drop(columns=['duration'], errors='ignore')
    df_pred = classifier.predict_storage_decision(df_all)

    if 'creation_block' in df_pred.columns:
        df_pred['block_time_proxy'] = df_pred['creation_block'] % 144    
    else:
        print("⚠️  No se encontró 'creation_block' en el DataFrame. No se puede calcular 'block_time_proxy'.")

    # === Mostrar resumen
    print("\n🔍 Distribución de decisiones:")
    print(df_pred['storage_decision'].value_counts())

    print("\n🎯 Ejemplos:")
    # print(df_pred[['value', 'duration', 'spend_probability', 'storage_decision', 'confidence']].head(20))
    print(df_pred[['value', 'block_time_proxy', 'spend_probability', 'storage_decision', 'confidence']].head(100))
    

    return df_pred



def main():
    """
    Uso real del sistema UTXOStorageClassifier con tus archivos reales
    """
    import glob

    horizon = 10

    # Inicializar clasificador
    classifier = UTXOStorageClassifier(prediction_horizon=horizon, max_block_height=789_999)

    # === Cargar tus archivos parquet
    parquet_dir = "/home/fernando/dev/utxo-experiments/parquet_normalized"
    parquet_files = sorted(glob.glob(f"{parquet_dir}/utxo-history-*.parquet")) #[:1]

    # === Cargar y preparar datos
    df = classifier.load_and_prepare_data(parquet_files)

    # === Samplear para no explotar RAM (ajustable)
    if len(df) > 5_000_000:
        print(f"⚠️ Dataset muy grande, usando solo muestra aleatoria de 5M")
        df = df.sample(n=5_000_000, random_state=42)
        print(f"✅ Dataset final: {len(df):,} filas")
        print(f"🎯 Distribución target: {df['target'].value_counts().to_dict()}")

    # === Ingeniería de features
    df_features = classifier.engineer_features(df)

    # === Análisis exploratorio
    analysis = classifier.analyze_patterns(df_features)

    # === Entrenar modelos
    model_results, X_test, y_test = classifier.train_model(df_features)

    # === Calibrar umbrales
    threshold_analysis = classifier.calibrate_thresholds(X_test, y_test)




    # === Exportar modelo para C++
    base_path = "/home/fernando/dev/utxo-experiments/model-13"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # Exportar en múltiples formatos
    classifier.export_for_cpp(base_path)
    
    # Generar código de ejemplo
    classifier.generate_cpp_example(f"{base_path}_example.cpp")
    
    # Guardar modelo Python también
    classifier.save_model(f"{base_path}.pkl")
        









    # === Reporte final
    classifier.generate_report(analysis, model_results, threshold_analysis)
    print("\n✅ Clasificador entrenado sobre datos reales.")
    # print("ℹ️  Podés usar classifier.predict_storage_decision(...) para hacer predicciones.")





    df_resultado = test_model_on_random_utxos(classifier, n_samples=100_000)
    print(f"✅ Resultado de prueba: {len(df_resultado):,} UTXOs procesados.")
    print(f"  Hot Storage:  {len(df_resultado[df_resultado['storage_decision'] == 'hot_storage']):,}")
    print(f"  Cold Storage: {len(df_resultado[df_resultado['storage_decision'] == 'cold_storage']):,}")

if __name__ == "__main__":
    main()



