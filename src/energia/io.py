"""Lectura y escritura de tablas (parquet con respaldo a CSV)."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def _motor_parquet_disponible() -> bool:
    for motor in ("pyarrow", "fastparquet"):
        try:
            __import__(motor)
            return True
        except ImportError:
            continue
    return False


def guardar_tabla(df: pd.DataFrame, ruta: Path) -> Path:
    """Guarda en parquet. Si no hay motor parquet instalado, guarda CSV al lado."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if _motor_parquet_disponible():
        df.to_parquet(ruta)
        return ruta
    alterna = ruta.with_suffix(".csv")
    warnings.warn(f"No hay pyarrow/fastparquet: se guarda {alterna.name} en lugar de parquet.")
    df.to_csv(alterna)
    return alterna


def cargar_tabla(ruta: Path) -> pd.DataFrame:
    """Lee parquet (o el CSV equivalente) y devuelve un DataFrame con índice de fechas."""
    ruta = Path(ruta)
    if ruta.exists() and _motor_parquet_disponible():
        return pd.read_parquet(ruta)
    alterna = ruta.with_suffix(".csv")
    if alterna.exists():
        return pd.read_csv(alterna, index_col=0, parse_dates=True)
    raise FileNotFoundError(
        f"No existe {ruta.name}. Ejecutar `python scripts/construir_dataset.py`."
    )
