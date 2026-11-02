"""Segmentación No Supervisada y Reducción de Dimensionalidad.

Implementa proyecciones 2D (PCA y manifold) junto con K-Means, DBSCAN y
Modelos de Mezclas Gaussianas (GMM) para perfilar arquetipos de comportamiento de compra.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def proyectar_pca_2d(
    X: np.ndarray,
    n_componentes: int = 2,
    semilla: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """Proyecta la matriz de características en un espacio bidimensional lineal con PCA.

    Args:
        X: Matriz escalada de datos.
        n_componentes: Número de componentes principales.
        semilla: Semilla aleatoria.

    Returns:
        Tuple[np.ndarray, PCA]: Coordenadas 2D y objeto PCA ajustado.
    """
    pca = PCA(n_components=n_componentes, random_state=semilla)
    coords = pca.fit_transform(X)
    return coords, pca


def evaluar_kmeans(
    X: np.ndarray,
    rango_k: range = range(2, 6),
    semilla: int = 42,
) -> Dict[str, Any]:
    """Evalúa candidatos de K-Means utilizando Inercia y Coeficiente de Silueta.

    Args:
        X: Matriz escalada.
        rango_k: Rango de valores de k a evaluar.
        semilla: Semilla aleatoria.

    Returns:
        Dict[str, Any]: Métricas de evaluación y k óptimo.
    """
    inercias: List[float] = []
    siluetas: List[float] = []
    candidatos = list(rango_k)

    mejor_k = candidatos[0]
    mejor_silueta = -1.0

    for k in candidatos:
        km = KMeans(n_clusters=k, random_state=semilla, n_init=10)
        etiquetas = km.fit_predict(X)
        inercias.append(float(km.inertia_))

        sil = float(silhouette_score(X, etiquetas))
        siluetas.append(sil)

        if sil > mejor_silueta:
            mejor_silueta = sil
            mejor_k = k

    return {
        "k_candidatos": candidatos,
        "inercias": inercias,
        "siluetas": siluetas,
        "k_optimo": mejor_k,
        "mejor_silueta": round(mejor_silueta, 4),
    }


def entrenar_modelos_segmentacion(
    X: np.ndarray,
    k: int = 3,
    semilla: int = 42,
) -> Dict[str, np.ndarray]:
    """Entrena concurrentemente K-Means, DBSCAN y GMM sobre la matriz de entrada.

    Args:
        X: Matriz escalada.
        k: Número de componentes/clusters.
        semilla: Semilla aleatoria.

    Returns:
        Dict[str, np.ndarray]: Etiquetas asignadas por cada modelo.
    """
    km = KMeans(n_clusters=k, random_state=semilla, n_init=10)
    etiquetas_km = km.fit_predict(X)

    db = DBSCAN(eps=1.8, min_samples=5)
    etiquetas_db = db.fit_predict(X)

    gmm = GaussianMixture(n_components=k, random_state=semilla)
    etiquetas_gmm = gmm.fit_predict(X)

    return {
        "kmeans": etiquetas_km,
        "dbscan": etiquetas_db,
        "gmm": etiquetas_gmm,
    }


def perfilar_segmentos_negocio(
    df: pd.DataFrame,
    etiquetas: np.ndarray,
) -> pd.DataFrame:
    """Genera resumen descriptivo bautizando arquetipos accionables de negocio.

    Args:
        df: DataFrame original de clientes/sesiones.
        etiquetas: Arreglo de cluster asignado.

    Returns:
        pd.DataFrame: Tabla de resumen con KPIs promediados por cluster.
    """
    df_copia = df.copy()
    df_copia["cluster"] = etiquetas

    nombres = {
        0: "Compradores Impulsivos / Alta Intención",
        1: "Navegadores Indecisos / Rebote Temprano",
        2: "Visitantes Ocasionales de Oferta",
    }
    df_copia["arquetipo"] = df_copia["cluster"].map(lambda c: nombres.get(c, f"Segmento-{c}"))

    resumen = (
        df_copia.groupby(["cluster", "arquetipo"])
        .agg(
            sesiones=("Revenue", "count"),
            tasa_conversion=("Revenue", "mean"),
            page_values_medio=("PageValues", "mean") if "PageValues" in df_copia else ("Revenue", "mean"),
        )
        .reset_index()
    )
    resumen["tasa_conversion"] = (resumen["tasa_conversion"] * 100.0).round(2)
    return resumen
