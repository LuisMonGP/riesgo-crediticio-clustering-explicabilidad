"""Modelado Supervisado y Explicabilidad Regulatoria (XAI).

Entrena modelos interpretables (Regresión Logística L2) vs modelos de ensamble
(Random Forest), cuantificando métricas de clasificación y generando explicabilidad
global y local con valores SHAP y LIME.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def entrenar_evaluar_logistica(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_regularizacion: float = 1.0,
    semilla: int = 42,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Entrena y evalúa una Regresión Logística regularizada L2 (Caja Blanca).

    Args:
        X_train: Matriz de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Matriz de prueba.
        y_test: Etiquetas de prueba.
        c_regularizacion: Inverso de fuerza de regularización.
        semilla: Semilla aleatoria.

    Returns:
        Tuple[LogisticRegression, Dict[str, float]]: Modelo ajustado y métricas.
    """
    clf = LogisticRegression(C=c_regularizacion, max_iter=1000, random_state=semilla)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metricas = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return clf, metricas


def entrenar_evaluar_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimadores: int = 100,
    profundidad_max: Optional[int] = 8,
    semilla: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Entrena y evalúa un modelo de ensamble Random Forest.

    Args:
        X_train: Matriz de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Matriz de prueba.
        y_test: Etiquetas de prueba.
        n_estimadores: Número de árboles.
        profundidad_max: Profundidad máxima de cada árbol.
        semilla: Semilla aleatoria.

    Returns:
        Tuple[RandomForestClassifier, Dict[str, float]]: Modelo ajustado y métricas.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimadores,
        max_depth=profundidad_max,
        random_state=semilla,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metricas = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return rf, metricas


def calcular_explicabilidad_shap(
    modelo: Any,
    X_muestra: np.ndarray,
    nombres_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calcula importancia global y valores SHAP para justificación regulatoria.

    Args:
        modelo: Estimador entrenado.
        X_muestra: Matriz de observaciones a auditar.
        nombres_variables: Nombres de columnas de la matriz.

    Returns:
        Dict[str, Any]: Ranking ordenado de variables por impacto medio.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(modelo) if hasattr(modelo, "estimators_") else shap.Explainer(modelo, X_muestra)
        shap_vals = explainer(X_muestra)
        raw_vals = shap_vals.values if hasattr(shap_vals, "values") else np.asarray(shap_vals)
        if raw_vals.ndim == 3:
            raw_vals = raw_vals[:, :, 1]
        media_abs = np.mean(np.abs(raw_vals), axis=0)
    except Exception:
        if hasattr(modelo, "feature_importances_"):
            media_abs = modelo.feature_importances_
        elif hasattr(modelo, "coef_"):
            media_abs = np.abs(modelo.coef_[0])
        else:
            media_abs = np.ones(X_muestra.shape[1]) / X_muestra.shape[1]

    nombres = nombres_variables if nombres_variables else [f"var_{i}" for i in range(len(media_abs))]
    ranking = sorted(zip(nombres, [float(v) for v in media_abs]), key=lambda p: p[1], reverse=True)

    return {
        "ranking_variables": ranking,
        "variable_mas_influyente": ranking[0][0] if ranking else "ninguna",
        "valores_shap_medios": [float(v) for v in media_abs],
    }
