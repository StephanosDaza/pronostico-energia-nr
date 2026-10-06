"""Descarga de métricas desde la API de XM (`pydataxm`)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config


def descargar_metrica(codigo: str, inicio: dt.date, fin: dt.date, entidad: str = "Sistema") -> pd.DataFrame:
    """Descarga una métrica por bloques anuales (la API limita el rango por consulta)."""
    from pydataxm import pydataxm

    api = pydataxm.ReadDB()
    bloques = []
    for anio in range(inicio.year, fin.year + 1):
        a = max(inicio, dt.date(anio, 1, 1))
        b = min(fin, dt.date(anio, 12, 31))
        df = api.request_data(codigo, entidad, a, b)
        if df is not None and len(df):
            bloques.append(df)
    if not bloques:
        raise RuntimeError(f"La API no devolvió datos para {codigo}")
    return pd.concat(bloques, ignore_index=True)


def descargar_todo(inicio: str = config.FECHA_INICIO, fin: str = config.FECHA_FIN) -> None:
    """Descarga todas las métricas de `config.VARIABLES_XM` a `data/raw/`."""
    ini = pd.Timestamp(inicio).date()
    fn = pd.Timestamp(fin).date()
    config.RAW.mkdir(parents=True, exist_ok=True)
    for codigo, (archivo, _, _) in config.VARIABLES_XM.items():
        print(f"Descargando {codigo} ...")
        df = descargar_metrica(codigo, ini, fn)
        df.to_csv(config.RAW / archivo, index=False)
        print(f"  -> {archivo}: {df.shape}")
