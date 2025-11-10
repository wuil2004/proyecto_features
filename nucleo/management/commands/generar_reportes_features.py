# nucleo/management/commands/generar_reportes_features.py

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

from django.conf import settings
from django.core.management.base import BaseCommand
from nucleo.models import ReportePrecalculado

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# --- CAMBIO AQUÍ: Imports para el Árbol y la Regresión ---
from sklearn.tree import DecisionTreeClassifier, plot_tree # <-- ¡Importamos plot_tree!
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Ya no necesitamos 'import graphviz'
# --- FIN CAMBIO ---


# --- Helpers de Carga (Sin cambios) ---
def load_dataset_locally():
    FILE_PATH = os.path.join(settings.BASE_DIR, 'data', 'TotalFeatures-ISCXFlowMeter.csv')
    if not os.path.exists(FILE_PATH):
        print(f"¡ERROR! No se encontró el archivo en {FILE_PATH}")
        return None
    print(f"Cargando dataset desde {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    print("Dataset cargado.")
    return df

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return(X, y)

# --- Función de ML (Sin cambios) ---
def correr_analisis_features(n, feature_list, X_train, y_train, X_val, y_val, class_names, base_f1_score):
    # (Esta función es la misma de antes)
    print(f"  Analizando con n={n}...")
    
    top_n_columns = list(feature_list.head(n).index)
    X_train_reduced = X_train[top_n_columns]
    X_val_reduced = X_val[top_n_columns]
    
    clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd_reduced.fit(X_train_reduced, y_train)
    
    y_pred_reduced = clf_rnd_reduced.predict(X_val_reduced)
    f1_reduced = f1_score(y_val, y_pred_reduced, average='weighted')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_val, 
        y_pred_reduced, 
        ax=ax, 
        display_labels=class_names,
        cmap='Blues',
        xticks_rotation='vertical'
    )
    ax.set_title(f"Matriz de Confusión (Top {n} Características)")
    plt.tight_layout()
    
    media_dir = os.path.join(settings.MEDIA_ROOT)
    os.makedirs(media_dir, exist_ok=True)
    
    image_name = f"matriz_confusion_n{n}.png"
    image_path = os.path.join(media_dir, image_name)
    plt.savefig(image_path)
    plt.close(fig)
    
    results = {
        "n_features": n,
        "f1_score_base": base_f1_score,
        "f1_score_reduced": f1_reduced,
        "f1_score_difference": f1_reduced - base_f1_score,
        "confusion_matrix_image_url": os.path.join(settings.MEDIA_URL, image_name),
        "features_used": top_n_columns
    }
    return results

# --- El comando que correremos ---

class Command(BaseCommand):
    help = 'Pre-calcula reportes de N-features. (ej: 1 5 10 20)'

    def add_arguments(self, parser):
        parser.add_argument(
            'n_features_list',
            nargs='+',
            type=int,
            help='La(s) N-features a calcular (ej: 1 5 10 20 30)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('--- Iniciando pre-cálculo de N-Features... ---'))
        start_time = time.time()
        
        # --- 1. PREPARACIÓN (Sin cambios) ---
        self.stdout.write("Cargando y preparando datos maestros...")
        df = load_dataset_locally()
        if df is None:
            self.stderr.write(self.style.ERROR("Fallo al cargar el dataset. Abortando."))
            return

        X = df.copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True) 

        class_codes, class_labels = X['calss'].factorize()
        X['calss'] = class_codes
        CLASS_NAMES = list(class_labels) 
        
        train_set, val_set, test_set = train_val_test_split(X, stratify='calss')
        X_train, y_train = remove_labels(train_set, 'calss')
        X_val, y_val = remove_labels(val_set, 'calss')
        
        # --- 2. MODELO MAESTRO (Sin cambios) ---
        self.stdout.write("Datos listos. Entrenando modelo maestro (full)...")
        clf_rnd_full = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf_rnd_full.fit(X_train, y_train)

        y_pred_full = clf_rnd_full.predict(X_val)
        F1_SCORE_BASE = f1_score(y_val, y_pred_full, average='weighted')
        self.stdout.write(self.style.SUCCESS(f"Modelo Maestro listo. F1-Base: {F1_SCORE_BASE:.4f}"))

        FEATURE_IMPORTANCES_SORTED = pd.Series(
            {name: score for name, score in zip(list(df.drop('calss', axis=1)), clf_rnd_full.feature_importances_)}
        ).sort_values(ascending=False)

        # --- 3. CAMBIO AQUÍ: GRÁFICOS ADICIONALES (Se hacen 1 vez) ---
        self.stdout.write(self.style.WARNING("Generando gráficos adicionales (Árbol y Regresión)..."))
        media_dir = os.path.join(settings.MEDIA_ROOT)
        os.makedirs(media_dir, exist_ok=True)
        
        # --- Lógica para el Árbol de Decisión (con plot_tree) ---
        try:
            tree_features = ['min_flowpktl', 'flow_fin']
            # Asegurarnos de que las features existan
            if all(f in X_train.columns for f in tree_features):
                X_train_reduced_tree = X_train[tree_features]
                
                clf_tree_simple = DecisionTreeClassifier(max_depth=4, random_state=42)
                clf_tree_simple.fit(X_train_reduced_tree, y_train)
                
                plt.figure(figsize=(20, 10)) # Un tamaño grande para que sea legible
                plot_tree(
                    clf_tree_simple,
                    feature_names=tree_features,
                    class_names=CLASS_NAMES,
                    filled=True,
                    rounded=True,
                    fontsize=10
                )
                
                # Guardar el árbol como PNG
                arbol_image_name = "arbol_decision_visual.png"
                arbol_path = os.path.join(media_dir, arbol_image_name)
                plt.savefig(arbol_path)
                plt.close()
                
                ARBOL_PNG_URL = os.path.join(settings.MEDIA_URL, arbol_image_name)
                self.stdout.write(self.style.SUCCESS("  -> Gráfico de Árbol (PNG) generado."))
            else:
                self.stdout.write(self.style.ERROR("  -> ERROR: 'min_flowpktl' o 'flow_fin' no encontradas."))
                ARBOL_PNG_URL = None
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  -> ERROR al generar el Árbol con plot_tree: {e}"))
            ARBOL_PNG_URL = None

        # --- Lógica para la Regresión (Sin cambios) ---
        try:
            reg_target = 'min_flowpktl'
            Xr_train = X_train.drop(columns=[reg_target], errors='ignore')
            yr_train = X_train[reg_target]
            Xr_val = X_val.drop(columns=[reg_target], errors='ignore')
            yr_val_true = X_val[reg_target]

            rf_reg = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            rf_reg.fit(Xr_train, yr_train)
            yr_val_pred = rf_reg.predict(Xr_val)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(yr_val_true, yr_val_pred, alpha=0.5)
            plt.plot([yr_val_true.min(), yr_val_true.max()], [yr_val_true.min(), yr_val_true.max()], 'r--', lw=2)
            plt.xlabel('Valor Real (min_flowpktl)')
            plt.ylabel('Predicción (min_flowpktl)')
            plt.title('Regresión RF - Predicción vs Real (Validación)')
            plt.tight_layout()
            
            regresion_image_name = "plot_regresion.png"
            regresion_path = os.path.join(media_dir, regresion_image_name)
            plt.savefig(regresion_path)
            plt.close()
            
            REGRESION_PNG_URL = os.path.join(settings.MEDIA_URL, regresion_image_name)
            self.stdout.write(self.style.SUCCESS("  -> Gráfico de Regresión generado."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  -> ERROR al generar la Regresión: {e}"))
            REGRESION_PNG_URL = None

        # --- FIN CAMBIO ---


        # --- 4. BUCLE DE CÁLCULO (El "trabajo pesado") ---
        n_features_a_correr = options['n_features_list']
        self.stdout.write(f"Iniciando bucle para: {n_features_a_correr}")

        for n in n_features_a_correr:
            if n > len(FEATURE_IMPORTANCES_SORTED):
                self.stdout.write(self.style.ERROR(f"Omitiendo n={n}. ..."))
                continue

            self.stdout.write(f"\nGenerando reporte para {n} features...")
            
            resultados_dict = correr_analisis_features(
                n=n,
                feature_list=FEATURE_IMPORTANCES_SORTED,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                class_names=CLASS_NAMES,
                base_f1_score=F1_SCORE_BASE
            )

            if resultados_dict:
                # --- CAMBIO AQUÍ: Añadir las URLs de los gráficos extra al dict ---
                resultados_dict["arbol_png_url"] = ARBOL_PNG_URL # <-- Cambiado a PNG
                resultados_dict["regresion_png_url"] = REGRESION_PNG_URL
                # --- FIN CAMBIO ---
            
                # 2. Guardamos en la DB
                job, created = ReportePrecalculado.objects.update_or_create(
                    n_features=n, 
                    defaults={
                        'resultados_json': json.dumps(resultados_dict)
                    }
                )

                if created: self.stdout.write(self.style.SUCCESS(f'¡Éxito! Reporte para {n} features CREADO.'))
                else: self.stdout.write(self.style.WARNING(f'¡Éxito! Reporte para {n} features ACTUALIZADO.'))
        
        total_time = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(f'\n--- ¡Proceso completado en {total_time:.2f} segundos! ---'))