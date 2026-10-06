"""Métricas de evaluación (en unidades originales) y líneas base naive.

Todas las funciones reciben arrays (N, HORIZONTE, 2) en unidades originales y
calculan la métrica por objetivo sobre todos los puntos (N x HORIZONTE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NOMBRES_CORTOS = {"demanda_kwh": "demanda", "precio_bolsa_cop_kwh": "precio"}


def _por_objetivo(y: np.ndarray, yhat: np.ndarray, fn) -> list[float]:
    return [float(fn(y[..., j].ravel(), yhat[..., j].ravel())) for j in range(y.shape[-1])]


def mae(y, p):
    return np.mean(np.abs(y - p))


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def mape(y, p):
    return 100 * np.mean(np.abs((y - p) / y))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


METRICAS = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


def evaluar(y: np.ndarray, yhat: np.ndarray, objetivos: list[str] = config.OBJETIVOS) -> dict[str, float]:
    """Devuelve un diccionario plano: {'demanda_MAE': ..., 'precio_RMSE': ..., ...}."""
    out = {}
    for nombre_m, fn in METRICAS.items():
        for obj, v in zip(objetivos, _por_objetivo(y, yhat, fn)):
            out[f"{NOMBRES_CORTOS.get(obj, obj)}_{nombre_m}"] = v
    return out


def tabla(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    """{'modelo': evaluar(...)} -> DataFrame legible (filas=modelo)."""
    return pd.DataFrame(resultados).T


def error_por_hora(y: np.ndarray, yhat: np.ndarray, metrica=mae) -> pd.DataFrame:
    """Error por paso del horizonte (h+1 ... h+24) y objetivo."""
    H = y.shape[1]
    datos = {
        NOMBRES_CORTOS.get(o, o): [float(metrica(y[:, h, j], yhat[:, h, j])) for h in range(H)]
        for j, o in enumerate(config.OBJETIVOS)
    }
    return pd.DataFrame(datos, index=pd.RangeIndex(1, H + 1, name="hora_adelante"))


# Líneas base: modelo naive estacional


def naive_estacional(datos, split: str, m: int = 24) -> np.ndarray:
    """Pronóstico naive estacional: ŷ(t) = y(t - m), con m en horas.

    m=24: la hora h de mañana es la hora h de hoy (naive diario).
    m=168: la hora h de mañana es la hora h del mismo día de la semana anterior (naive semanal).
    """
    H = datos.horizonte
    if m < H or m > datos.ventana:
        raise ValueError(f"m debe estar entre {H} y {datos.ventana} horas")
    ini = datos.ventana - m
    n_obj = len(datos.columnas_objetivo)
    y_esc = datos.splits[split].pasado[:, ini : ini + H, :n_obj]
    return datos.invertir(y_esc)
