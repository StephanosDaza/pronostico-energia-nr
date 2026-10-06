"""Festivos de Colombia"""
from __future__ import annotations

import datetime as dt

import pandas as pd


def _domingo_pascua(anio: int) -> dt.date:
    """Algoritmo de Meeus/Jones/Butcher (calendario gregoriano)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return dt.date(anio, mes, dia + 1)


def _siguiente_lunes(fecha: dt.date) -> dt.date:
    return fecha + dt.timedelta(days=(7 - fecha.weekday()) % 7)


def festivos_colombia(anio: int) -> dict[dt.date, str]:
    pascua = _domingo_pascua(anio)
    d = dt.timedelta
    fijos = {
        dt.date(anio, 1, 1): "Año Nuevo",
        dt.date(anio, 5, 1): "Día del Trabajo",
        dt.date(anio, 7, 20): "Independencia",
        dt.date(anio, 8, 7): "Batalla de Boyacá",
        dt.date(anio, 12, 8): "Inmaculada Concepción",
        dt.date(anio, 12, 25): "Navidad",
        pascua - d(3): "Jueves Santo",
        pascua - d(2): "Viernes Santo",
    }
    trasladables = {
        dt.date(anio, 1, 6): "Reyes Magos",
        dt.date(anio, 3, 19): "San José",
        dt.date(anio, 6, 29): "San Pedro y San Pablo",
        dt.date(anio, 8, 15): "Asunción",
        dt.date(anio, 10, 12): "Día de la Raza",
        dt.date(anio, 11, 1): "Todos los Santos",
        dt.date(anio, 11, 11): "Independencia de Cartagena",
        pascua + d(39): "Ascensión",
        pascua + d(60): "Corpus Christi",
        pascua + d(68): "Sagrado Corazón",
    }
    out = dict(fijos)
    out.update({_siguiente_lunes(f): n for f, n in trasladables.items()})
    return out


def es_festivo(fechas: pd.DatetimeIndex) -> pd.Series:
    """Devuelve una serie booleana alineada con `fechas` (horarias o diarias)."""
    anios = range(fechas.min().year, fechas.max().year + 1)
    todos = set()
    for a in anios:
        todos.update(festivos_colombia(a))
    dias = pd.Series(fechas.normalize().date, index=fechas)
    return dias.isin(todos)
