"""Suite de pruebas unitarias para riesgo-crediticio-clustering-explicabilidad."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.feature_pipeline import construir_preprocesador_shoppers
from src.clustering import proyectar_pca_2d, evaluar_kmeans
from src.supervised_xai import entrenar_evaluar_logistica


@pytest.fixture
def df_shoppers() -> pd.DataFrame:
    """Fixture que provee la muestra de datos de Shoppers Intention."""
    return load_dataset(force_sample=True)


def test_pipeline_cero_nan_en_datos_nuevos(df_shoppers: pd.DataFrame) -> None:
    """Valida que el preprocesador transforme datos con nulos produciendo 0 NaNs."""
    preprocesador = construir_preprocesador_shoppers()

    mitad = len(df_shoppers) // 2
    df_train = df_shoppers.iloc[:mitad]
    df_test = df_shoppers.iloc[mitad:].copy()

    # Inyectar nulos controlados en conjunto de prueba
    df_test.loc[df_test.index[0], "PageValues"] = np.nan
    df_test.loc[df_test.index[1], "Month"] = np.nan

    preprocesador.fit(df_train)
    X_test_proc = preprocesador.transform(df_test)

    assert not np.isnan(X_test_proc).any(), "La salida transformada no debe contener NaNs"
    assert X_test_proc.shape[0] == len(df_test)


def test_probabilidades_predichas_acotadas(df_shoppers: pd.DataFrame) -> None:
    """Valida que las probabilidades del modelo supervisado estén estrictamente en [0.0, 1.0]."""
    preprocesador = construir_preprocesador_shoppers()
    X = preprocesador.fit_transform(df_shoppers)
    y = df_shoppers["Revenue"].values

    clf, metricas = entrenar_evaluar_logistica(X[:250], y[:250], X[250:], y[250:])
    probas = clf.predict_proba(X[250:])[:, 1]

    assert (probas >= 0.0).all() and (probas <= 1.0).all()
    assert "roc_auc" in metricas and 0.0 <= metricas["roc_auc"] <= 1.0


def test_dimensionalidad_pca(df_shoppers: pd.DataFrame) -> None:
    """Verifica que PCA proyecte exactamente a 2 dimensiones."""
    preprocesador = construir_preprocesador_shoppers()
    X = preprocesador.fit_transform(df_shoppers)

    coords, pca_obj = proyectar_pca_2d(X, n_componentes=2)
    assert coords.shape == (len(df_shoppers), 2)
    assert len(pca_obj.explained_variance_ratio_) == 2
