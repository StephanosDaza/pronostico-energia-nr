"""Estilo y guardado de figuras."""
from __future__ import annotations

import matplotlib.pyplot as plt

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
