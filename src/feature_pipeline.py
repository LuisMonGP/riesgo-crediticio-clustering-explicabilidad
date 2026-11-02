"""Pipelines de Preprocesamiento y Feature Engineering Libres de Data Leakage.

Construye ColumnTransformers de Scikit-Learn ajustados exclusivamente sobre X_train,
incorporando imputación, corrección de asimetría log1p, escalado estándar y codificación one-hot.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def construir_preprocesador_shoppers() -> ColumnTransformer:
    """Construye un ColumnTransformer robusto libre de fuga de datos (Data Leakage).

    Returns:
        ColumnTransformer: Preprocesador listo para ajustar estrictamente en X_train.
    """
    # 1. Variables numéricas de alta asimetría
    sesgadas_num = ["PageValues", "ProductRelated_Duration", "BounceRates", "ExitRates"]
    pipe_sesgadas = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Variables numéricas de conteo y duración general
    estandar_num = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "SpecialDay",
    ]
    pipe_estandar = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 3. Variables categóricas
    categoricas = ["Month", "VisitorType"]
    pipe_cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num_sesgadas", pipe_sesgadas, sesgadas_num),
            ("num_estandar", pipe_estandar, estandar_num),
            ("cat", pipe_cat, categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocesador
