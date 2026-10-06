"""Feature engineering: interim -> processed.

Grupos de variables (los usa `dataset.py` para armar las ventanas):

- OBJETIVOS            demanda y precio de bolsa (también entran como historia)
- COVARIABLES_PASADAS  solo se conocen hasta el instante de emisión del pronóstico
- COVARIABLES_FUTURAS  se conocen de antemano para las 24 h a pronosticar (calendario)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .festivos import es_festivo
from .io import cargar_tabla, guardar_tabla


def agregar_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Codificación cíclica (seno/coseno) + indicadores de tipo de día."""
    idx = df.index
    out = pd.DataFrame(index=idx)
    hora, dow, doy = idx.hour, idx.dayofweek, idx.dayofyear
    out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    out["dia_semana_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dia_semana_cos"] = np.cos(2 * np.pi * dow / 7)
    out["dia_anio_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["dia_anio_cos"] = np.cos(2 * np.pi * doy / 365.25)
    festivo = es_festivo(idx).to_numpy()
    out["es_festivo"] = festivo.astype(float)
    out["es_sabado"] = (dow == 5).astype(float)
    out["es_domingo_o_festivo"] = ((dow == 6) | festivo).astype(float)
    return out


def agregar_exogenas_diarias(df: pd.DataFrame, rezago_dias: int = config.REZAGO_EXOGENAS_DIARIAS) -> pd.DataFrame:
    """Exógenas diarias rezagadas `rezago_dias` días (anti-fuga) + derivadas.

    Se trabaja a nivel diario y luego se repite el valor en las 24 h del día,
    así cada hora del día d solo ve información del día d-rezago o anterior.
    """
    diario = df[config.EXOGENAS_DIARIAS].resample("D").first()
    deriv = pd.DataFrame(index=diario.index)
    deriv["aportes_kwh"] = diario["aportes_kwh"]
    deriv["aportes_media7_kwh"] = diario["aportes_kwh"].rolling(7, min_periods=1).mean()
    deriv["volumen_util_kwh"] = diario["volumen_util_kwh"]
    deriv["volumen_util_delta7_kwh"] = diario["volumen_util_kwh"].diff(7)
    deriv = deriv.shift(rezago_dias)
    horario = deriv.reindex(df.index.normalize())
    horario.index = df.index
    return horario.add_suffix(f"_d{rezago_dias}") if rezago_dias else horario


def construir_features(horario: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el dataset final (sin escalar). Elimina las filas iniciales
    que quedan con NaN por los rezagos."""
    partes = [
        horario[config.OBJETIVOS],
        horario[config.EXOGENAS_HORARIAS],
        agregar_exogenas_diarias(horario),
        agregar_calendario(horario),
    ]
    df = pd.concat(partes, axis=1).dropna()
    df.index.name = "fecha_hora"
    return df.astype("float32")


# Nombres de grupos tal como quedan después de `construir_features`
_SUF = f"_d{config.REZAGO_EXOGENAS_DIARIAS}" if config.REZAGO_EXOGENAS_DIARIAS else ""
COVARIABLES_FUTURAS = [
    "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos", "dia_anio_sin", "dia_anio_cos",
    "es_festivo", "es_sabado", "es_domingo_o_festivo",
]
COVARIABLES_PASADAS = config.EXOGENAS_HORARIAS + [
    f"aportes_kwh{_SUF}", f"aportes_media7_kwh{_SUF}", f"volumen_util_kwh{_SUF}", f"volumen_util_delta7_kwh{_SUF}",
]
COLUMNAS_SIN_ESCALAR = ["es_festivo", "es_sabado", "es_domingo_o_festivo",
                        "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
                        "dia_anio_sin", "dia_anio_cos"]


def ejecutar() -> pd.DataFrame:
    """Paso completo interim -> processed."""
    horario = cargar_tabla(config.ARCHIVO_HORARIO)
    df = construir_features(horario)
    ruta = guardar_tabla(df, config.ARCHIVO_FEATURES)
    print(f"Dataset con features: {df.shape} -> {ruta.relative_to(config.RAIZ)}")
    return df
