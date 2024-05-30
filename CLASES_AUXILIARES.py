# ------------------------------------------------------
# Definicion de las clases Figuras
# Código
# Andres Vargas y Camilo Diaz Granados
# v1 - 22.04.2024
# ------------------------------------------------------
# IMPORTAR MODULOS
# Se necesita polars == ****

from sklearn.base import BaseEstimator, ClassifierMixin
import polars as pl

# ------------------------------------------------------
# CLASE Pipeline:
# Preprocesar datos del challenge.
# METODOS:
# set_table_dtypes: Este método estático convierte co-
# lumnas de un DataFrame en tipos de datos específicos
# según los nombres de las columnas. Por ejemplo, deter-
# minadas columnas se convierten en números enteros, fe-
# chas, flotantes o cadenas según sus nombres o sufijos.
# handle_dates: Procesa columnas de fechas. Calcula la
# diferencia entre las fechas en el DataFrame y la co-
# lumna date_decision, convierte esta diferencia en días
# totales y luego la convierte en un tipo flotante. Tam-
# bién elimina la fecha_decisión y cualquier columna
# 'MONTH'.
# filter_cols: Filtra columnas según condiciones especí-
# ficas. Elimina columnas con más del 95% de valores fa-
# ltantes y también elimina columnas de cadena con un
#solo valor único o más de 200 valores únicos, excepto
# las columnas esenciales como "target", "case_id" y
# "WEEK_NUM".
# ------------------------------------------------------

class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                df = df.with_columns(pl.col(col).cast(pl.Float32))
                
        df = df.drop("date_decision", "MONTH")

        return df
    
    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df

# ------------------------------------------------------
# CLASE Aggregator:
# Automatizar la agregación de columnas en función de
# sus sufijos en un conjunto de datos, probablemente pa-
# ra la ingeniería de funciones antes del entrenamiento
# del modelo.
# METODOS:
# num_expr: agrega columnas que terminan en "P" o "A",
# normalmente valores numéricos, calculando el máximo.
# date_expr: apunta a columnas que terminan en "D", que
# son columnas de fecha, y calcula la fecha máxima.
# str_expr: procesa columnas de cadena que terminan en
# "M", encontrando el valor máximo, que podría ser útil
# para codificar variables categóricas.
# other_expr: Maneja columnas que terminan en "T" o "L",
# con tipos de datos especí ficos, calculando sus valor-
# es máximos.
# count_expr: Se centra en las columnas con "num_group"
# en su nombre, calculando también el valor máximo.
# vget_exprs: Combina todas las expresiones de los otros
# métodos en una sola lista para aplicarla al marco de
# datos, agilizando el proceso de generación de caracte-
# rísticas agregadas.
# ------------------------------------------------------

class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max
    
    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs

# ------------------------------------------------------
# CLASE VotingModel:
# Gestiona múltiples estimadores y agrega sus predi-
# cciones.
# METODOS:
# fit: devuelve la instancia de clase sin cambios, ya
# que no entrena ningún modelo sino que se basa en mode-
# los previamente entrenados.
# predict: Calcula el promedio de predicciones de cada 
# estimador para las características de entrada X, lo
# convierte en una forma simple de conjunto de votación.
# predict_proba: similar a predecir, pero promedia las
# probabilidades predichas por cada estimador.
# ------------------------------------------------------

class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
