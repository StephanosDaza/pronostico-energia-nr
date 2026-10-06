"""Herramientas estadísticas para series de tiempo (numpy/scipy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def acf(x: pd.Series | np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrelación muestral para rezagos 0..nlags."""
    x = np.asarray(x, dtype="float64")
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    return np.array([np.dot(x[: n - k], x[k:]) / n / var for k in range(nlags + 1)])


def pacf(x: pd.Series | np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrelación parcial para rezagos 0..nlags (recursión de Durbin-Levinson)."""
    r = acf(x, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros(nlags + 1)
    var = 1.0
    for k in range(1, nlags + 1):
        a = (r[k] - np.dot(phi[1:k], r[k - 1:0:-1])) / var
        nuevo = phi.copy()
        nuevo[k] = a
        nuevo[1:k] = phi[1:k] - a * phi[k - 1:0:-1]
        phi = nuevo
        var *= 1 - a * a
        out[k] = a
    return out


def banda_confianza(n: int, nivel: float = 0.95) -> float:
    """Banda aproximada ±z/sqrt(n) para ACF y PACF bajo ruido blanco."""
    return float(stats.norm.ppf(0.5 + nivel / 2) / np.sqrt(n))


def correlacion_cruzada(x: pd.Series, y: pd.Series, rezagos: range) -> pd.Series:
    """corr(x_{t-k}, y_t) para cada k en `rezagos` (k>0: x adelanta a y)."""
    return pd.Series({k: x.shift(k).corr(y) for k in rezagos}, name=f"{x.name}->{y.name}")


# Valores críticos de MacKinnon (2010) para la prueba ADF con constante, n grande
_CRITICOS_ADF = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def adf(x: pd.Series | np.ndarray, rezagos: int = 24) -> dict:
    """Prueba Dickey-Fuller aumentada (con constante).

    Δx_t = α + γ x_{t-1} + Σ_{i=1..p} β_i Δx_{t-i} + ε_t      H0: γ = 0 (raíz unitaria)
    """
    x = np.asarray(x, dtype="float64")
    dx = np.diff(x)
    p = rezagos
    Y = dx[p:]
    cols = [np.ones_like(Y), x[p:-1]] + [dx[p - i : -i] for i in range(1, p + 1)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    s2 = resid @ resid / (len(Y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    t_stat = beta[1] / np.sqrt(cov[1, 1])
    return {
        "estadistico": float(t_stat),
        **{f"critico_{k}": v for k, v in _CRITICOS_ADF.items()},
        "estacionaria_5%": bool(t_stat < _CRITICOS_ADF["5%"]),
    }


def descomposicion_varianza_intradia(serie: pd.Series) -> dict:
    """Parte la varianza total de una serie horaria en:

      entre días  = varianza de la media diaria   (lo que ve un modelo diario)
      intra-día   = varianza de (valor horario - media de su día)

    Si la parte intra-día es grande, agregar a resolución diaria descarta
    información que el problema necesita.
    """
    media_dia = serie.groupby(serie.index.normalize()).transform("mean")
    total = serie.var(ddof=0)
    entre = media_dia.var(ddof=0)
    intra = (serie - media_dia).var(ddof=0)
    return {
        "var_total": total,
        "pct_entre_dias": 100 * entre / total,
        "pct_intra_dia": 100 * intra / total,
    }


def resumen_distribucion(df: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación, asimetría y curtosis (exceso) por columna."""
    return pd.DataFrame({
        "media": df.mean(),
        "desv": df.std(),
        "cv_%": 100 * df.std() / df.mean(),
        "asimetria": df.apply(lambda s: stats.skew(s.dropna())),
        "curtosis": df.apply(lambda s: stats.kurtosis(s.dropna())),
    })


def atipicos_iqr(serie: pd.Series, k: float = 3.0) -> pd.Series:
    """Marca valores fuera de [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (serie < q1 - k * iqr) | (serie > q3 + k * iqr)
