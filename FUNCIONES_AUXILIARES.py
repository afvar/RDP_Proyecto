# ----------------------------------------------------------------------------------------
# MODULO: Funciones Auxiliares para dataset
# ----------------------------------------------------------------------------------------
# Descripción: Este módulo contiene funciones para transformar datasets
# ----------------------------------------------------------------------------------------
# Autor: Andrés Felipe Vargas
# Version: 1.0
# [22.04.2024]
# ----------------------------------------------------------------------------------------
# IMPORTAR MODULOS
# Se necesita polars == ****

import polars as pl
import CLASES_AUXILIARES as clases
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------------------------------------------------------------------
# FUNCIÓN: read_file
# ----------------------------------------------------------------------------------------
# Convierte las columnas de tipo string a variables categoricas que pueda leer las dife-
# rentes técnicas de reconocimiento de patrones.
# ----------------------------------------------------------------------------------------
# Solo recibe objetos de la clase texto
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------

def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(clases.Pipeline.set_table_dtypes)
    
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(clases.Aggregator.get_exprs(df))
    
    return df

# ----------------------------------------------------------------------------------------
# FUNCIÓN: read_files
# ----------------------------------------------------------------------------------------
# Convierte las columnas de tipo string a variables categoricas que pueda leer las dife-
# rentes técnicas de reconocimiento de patrones.
# ----------------------------------------------------------------------------------------
# Solo recibe objetos de la clase texto
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------

def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(clases.Pipeline.set_table_dtypes)
        
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(clases.Aggregator.get_exprs(df))
        
        chunks.append(df)
        
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    
    return df
 
# ----------------------------------------------------------------------------------------
# FUNCIÓN: feature_eng
# ----------------------------------------------------------------------------------------
# Ingeniería de funciones en un proceso de preprocesamiento de aprendizaje automático. Mo-
# difica y enriquece el marco de datos base (df_base) agregando nuevas características de-
# rivadas de la columna date_decision, específicamente el mes y el día de la semana de las
# decisiones. Luego integra datos adicionales uniendo df_base con otros marcos de datos
# (profundidad_0, profundidad_1 y profundidad_2) en la columna "case_id".
# ----------------------------------------------------------------------------------------
# Solo recibe objetos de la clase polars dataframe y enteros para depth.
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )
        
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        
    df_base = df_base.pipe(clases.Pipeline.handle_dates)
    
    return df_base

# ----------------------------------------------------------------------------------------
# FUNCIÓN: to_pandas
# ----------------------------------------------------------------------------------------
# Convierte el objeto de clase polars dataframe a un objeto pandas dataframe.
# ----------------------------------------------------------------------------------------
# Solo recibe objetos de la clase polars dataframe.
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols

def evaluate_model(model, model_name, X_val, y_val, params):
    y_pred = model.predict(X_val)
    return {
        'modelo': model_name
        ,'parametros': params
        ,'accuracy_score': accuracy_score(y_val, y_pred)
        ,'precision_score': precision_score(y_val, y_pred)
        ,'recall_score': recall_score(y_val, y_pred)
        ,'f1_score': f1_score(y_val, y_pred)
        ,'roc_auc_score': roc_auc_score(y_val, y_pred)
    }