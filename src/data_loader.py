"""Cargador de datos e ingesta adaptativa para E-Commerce y Riesgo Crediticio.

Implementa el patrón de adaptador profesional: intenta cargar el dataset oficial
completo desde data/raw/online_shoppers_intention.csv; si no existe, utiliza la muestra
reproducible de data/sample/sample_data.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "online_shoppers_intention.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample" / "sample_data.csv"


def load_dataset(force_sample: bool = False, nrows: Optional[int] = None) -> Any:
    """Carga el dataset orgánico o la muestra representativa para CI/CD y tests.

    Args:
        force_sample: Si es True, fuerza la carga de data/sample/sample_data.csv.
        nrows: Límite opcional de filas.

    Returns:
        pd.DataFrame | List[dict]: Dataset cargado.
    """
    target_path = None
    if not force_sample and RAW_PATH.exists():
        target_path = RAW_PATH
        print(f"[INFO] Cargando dataset orgánico completo desde: {RAW_PATH}")
    elif SAMPLE_PATH.exists():
        target_path = SAMPLE_PATH
        print(f"[INFO] Cargando muestra representativa reproducible desde: {SAMPLE_PATH}")
    else:
        raise FileNotFoundError(
            f"No se encontró dataset crudo en '{RAW_PATH}' ni muestra en '{SAMPLE_PATH}'."
        )

    if HAS_PANDAS:
        df = pd.read_csv(target_path, nrows=nrows)
        if "Revenue" in df.columns and df["Revenue"].dtype == bool:
            df["Revenue"] = df["Revenue"].astype(int)
        if "Weekend" in df.columns and df["Weekend"].dtype == bool:
            df["Weekend"] = df["Weekend"].astype(int)
        return df

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)[:nrows] if nrows else list(reader)


if __name__ == "__main__":
    df = load_dataset()
    if HAS_PANDAS:
        print(f"Dimensiones cargadas: {df.shape}")
        print(f"Distribución del Target (Revenue):\n{df['Revenue'].value_counts(normalize=True)}")
    else:
        print(f"Total filas: {len(df)}")
