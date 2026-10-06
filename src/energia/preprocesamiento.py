"""Estandarización de los CSV crudos de XM a una sola serie horaria.

Entrada:  data/raw/*.csv  (formato XM: una fila por día, columnas Values_Hour01..24)
Salida:   data/interim/dataset_horario.parquet
          índice `fecha_hora` horario continuo, una columna por variable.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .io import guardar_tabla

COLUMNAS_HORA = [f"Values_Hour{h:02d}" for h in range(1, 25)]


def leer_horaria(archivo: str, nombre: str) -> pd.Series:
    """Convierte el formato ancho de XM (día x 24 horas) a una serie horaria.

    `Values_Hour01` corresponde al periodo 00:00-01:00, que se indexa como 00:00.
    """
    df = pd.read_csv(config.RAW / archivo, parse_dates=["Date"])
    faltan = set(COLUMNAS_HORA) - set(df.columns)
    if faltan:
        raise ValueError(f"{archivo}: faltan columnas {sorted(faltan)}")
    largo = df.melt(id_vars="Date", value_vars=COLUMNAS_HORA, var_name="hora", value_name=nombre)
    largo["hora"] = largo["hora"].str[-2:].astype(int) - 1
    largo["fecha_hora"] = largo["Date"] + pd.to_timedelta(largo["hora"], unit="h")
    serie = pd.to_numeric(largo.set_index("fecha_hora")[nombre], errors="coerce")
    return serie.sort_index()


def leer_diaria(archivo: str, nombre: str) -> pd.Series:
    """Lee una métrica diaria de XM (columnas Date, Value)."""
    df = pd.read_csv(config.RAW / archivo, parse_dates=["Date"])
    serie = pd.to_numeric(df.set_index("Date")["Value"], errors="coerce").rename(nombre)
    serie.index.name = "fecha"
    return serie.sort_index()


def construir_dataset_horario() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee todos los crudos y devuelve (horario, diario).

    - horario: variables horarias + exógenas diarias repetidas en las 24 h de su
      día (SIN rezago; el rezago anti-fuga se aplica en `features.py`).
    - diario:  exógenas diarias en su resolución nativa (útil para el EDA).
    """
    horarias, diarias = {}, {}
    for _, (archivo, nombre, resolucion) in config.VARIABLES_XM.items():
        if resolucion == "horaria":
            horarias[nombre] = leer_horaria(archivo, nombre)
        else:
            diarias[nombre] = leer_diaria(archivo, nombre)

    horario = pd.concat(horarias, axis=1, join="outer")
    indice = pd.date_range(config.FECHA_INICIO, pd.Timestamp(config.FECHA_FIN) + pd.Timedelta(hours=23), freq="h")
    horario = horario.reindex(indice)
    horario.index.name = "fecha_hora"

    diario = pd.concat(diarias, axis=1).reindex(pd.date_range(config.FECHA_INICIO, config.FECHA_FIN, freq="D"))
    diario.index.name = "fecha"

    dia = horario.index.normalize()
    for col in diario.columns:
        horario[col] = diario[col].reindex(dia).to_numpy()
    return horario, diario


def reporte_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de calidad por columna: nulos, ceros, negativos, rango."""
    return pd.DataFrame({
        "nulos": df.isna().sum(),
        "ceros": (df == 0).sum(),
        "negativos": (df < 0).sum(),
        "min": df.min(),
        "max": df.max(),
        "media": df.mean(),
    })


def validar(df: pd.DataFrame) -> None:
    """Falla ruidosamente si la serie no es apta para modelar."""
    esperado = pd.date_range(df.index.min(), df.index.max(), freq="h")
    assert df.index.equals(esperado), "El índice horario tiene huecos o duplicados"
    nulos = df.isna().sum()
    assert nulos.sum() == 0, f"Hay valores nulos:\n{nulos[nulos > 0]}"
    assert (df[config.OBJETIVOS] > 0).all().all(), "Hay objetivos <= 0"


def ejecutar() -> pd.DataFrame:
    """Paso completo raw -> interim. Devuelve el DataFrame horario."""
    horario, _ = construir_dataset_horario()
    validar(horario)
    ruta = guardar_tabla(horario, config.ARCHIVO_HORARIO)
    print(f"Dataset horario: {horario.shape} -> {ruta.relative_to(config.RAIZ)}")
    return horario
