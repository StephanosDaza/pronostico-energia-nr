"""Estilo y guardado de figuras."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

COLORES = {"demanda": "#0f766e", "precio": "#b45309"}
ETIQUETAS = {"demanda_kwh": "Demanda no regulada [kWh]", "precio_bolsa_cop_kwh": "Precio de bolsa [COP/kWh]"}


def estilo() -> None:
    plt.rcParams.update({
        "figure.figsize": (12, 4), "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURAS / f"{nombre}.png", bbox_inches="tight", dpi=120)


def curvas_aprendizaje(historial, titulo: str = "") -> plt.Figure:
    """Pérdida de entrenamiento y validación por época (DataFrame con columnas train y val)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(historial.index, historial["train"], marker=".", label="entrenamiento")
    ax.plot(historial.index, historial["val"], marker=".", label="validación")
    mejor = historial["val"].idxmin()
    ax.axvline(mejor, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="época", ylabel="MSE (escala estandarizada)", yscale="log", title=titulo)
    ax.legend()
    fig.tight_layout()
    return fig


def pronostico_vs_real(datos, prediccion, split: str = "val", dias: int = 14, desde: int = 0,
                       titulo: str = "") -> plt.Figure:
    """Concatena `dias` pronósticos day-ahead consecutivos y los compara con lo observado."""
    real = datos.y_original(split)[desde : desde + dias]
    pred = np.asarray(prediccion)[desde : desde + dias]
    t0 = datos.splits[split].t0[desde : desde + dias]
    t = pd.DatetimeIndex(np.concatenate([pd.date_range(s, periods=datos.horizonte, freq="h") for s in t0]))
    fig, axes = plt.subplots(len(datos.columnas_objetivo), 1, figsize=(13, 6), sharex=True)
    for j, (ax, col) in enumerate(zip(axes, datos.columnas_objetivo)):
        ax.plot(t, real[..., j].ravel(), color="#1f2937", lw=1.2, label="real")
        ax.plot(t, pred[..., j].ravel(), color="#2563eb", lw=1.2, label="pronóstico")
        ax.set_ylabel(ETIQUETAS.get(col, col))
        ax.legend(loc="upper left")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig
