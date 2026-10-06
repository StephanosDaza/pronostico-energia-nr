"""Escalado y construcción de ventanas para los modelos.

Formulación común:
    pasado : X[t0-VENTANA : t0]   -> (VENTANA, n_pasado)
    futuro : C[t0 : t0+HORIZONTE] -> (HORIZONTE, n_futuro)
    y      : Y[t0 : t0+HORIZONTE] -> (HORIZONTE, 2)

Train usa t0 en todas las horas (PASO_ENTRENAMIENTO). Val y test usan t0 a las
00:00 (un pronóstico day-ahead por día).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import COLUMNAS_SIN_ESCALAR, COVARIABLES_FUTURAS, COVARIABLES_PASADAS
from .io import cargar_tabla

@dataclass
class Escalador:
    """log1p opcional + estandarización z-score, por columna."""
    media: dict[str, float] = field(default_factory=dict)
    desv: dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    @staticmethod
    def _usa_log(col: str) -> bool:
        return any(col.startswith(p) for p in config.COLUMNAS_LOG)

    def ajustar(self, df: pd.DataFrame) -> "Escalador":
        for col in df.columns:
            if col in COLUMNAS_SIN_ESCALAR:
                continue
            x = df[col].astype("float64")
            if self._usa_log(col):
                x = np.log1p(x)
                self.log.append(col)
            self.media[col] = float(x.mean())
            self.desv[col] = float(x.std()) or 1.0
        return self

    def transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.media:
            x = out[col].astype("float64")
            if col in self.log:
                x = np.log1p(x)
            out[col] = ((x - self.media[col]) / self.desv[col]).astype("float32")
        return out

    def invertir(self, arr: np.ndarray, columnas: list[str]) -> np.ndarray:
        """Invierte la transformación en la última dimensión de `arr` (p. ej. y (N, H, 2))."""
        out = np.asarray(arr, dtype="float64").copy()
        for j, col in enumerate(columnas):
            v = out[..., j] * self.desv[col] + self.media[col]
            out[..., j] = np.expm1(v) if col in self.log else v
        return out

    def guardar(self, ruta: Path = config.ARCHIVO_ESCALADOR) -> None:
        Path(ruta).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def cargar(cls, ruta: Path = config.ARCHIVO_ESCALADOR) -> "Escalador":
        return cls(**json.loads(Path(ruta).read_text()))


@dataclass
class Split:
    pasado: np.ndarray   # (N, VENTANA, n_pasado)
    futuro: np.ndarray   # (N, HORIZONTE, n_futuro)
    y: np.ndarray        # (N, HORIZONTE, 2)   escalado
    t0: pd.DatetimeIndex  # inicio del horizonte de cada muestra

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class DatosModelo:
    splits: dict[str, Split]
    escalador: Escalador
    columnas_pasado: list[str]
    columnas_futuro: list[str]
    columnas_objetivo: list[str]
    ventana: int
    horizonte: int

    def entradas(self, split: str) -> dict[str, np.ndarray]:
        """Entradas en formato {"pasado": ..., "futuro": ...}."""
        s = self.splits[split]
        return {"pasado": s.pasado, "futuro": s.futuro}

    def y_original(self, split: str) -> np.ndarray:
        """Objetivos de un split en unidades originales (kWh, COP/kWh)."""
        return self.escalador.invertir(self.splits[split].y, self.columnas_objetivo)

    def invertir(self, y_escalado: np.ndarray) -> np.ndarray:
        return self.escalador.invertir(y_escalado, self.columnas_objetivo)

    def resumen(self) -> pd.DataFrame:
        filas = []
        for n, s in self.splits.items():
            filas.append({
                "split": n, "muestras": len(s),
                "pasado": s.pasado.shape, "futuro": s.futuro.shape, "y": s.y.shape,
                "primer_t0": s.t0.min(), "ultimo_t0": s.t0.max(),
            })
        return pd.DataFrame(filas).set_index("split")


def _indices_t0(idx: pd.DatetimeIndex, ventana: int, horizonte: int) -> dict[str, np.ndarray]:
    """Posiciones enteras válidas de t0 para cada split."""
    n = len(idx)
    pos = np.arange(ventana, n - horizonte + 1)
    t_ini = idx[pos]
    t_fin = idx[pos + horizonte - 1]
    fin_train, fin_val = pd.Timestamp(config.FIN_TRAIN), pd.Timestamp(config.FIN_VAL)
    medianoche = t_ini.hour == 0
    sel = {
        "train": (t_fin <= fin_train),
        "val": (t_ini > fin_train) & (t_fin <= fin_val) & medianoche,
        "test": (t_ini > fin_val) & medianoche,
    }
    out = {k: pos[np.asarray(v)] for k, v in sel.items()}
    out["train"] = out["train"][:: config.PASO_ENTRENAMIENTO]
    return out


def _ventanas(arr: np.ndarray, posiciones: np.ndarray, desde: int, largo: int) -> np.ndarray:
    """Extrae arr[p+desde : p+desde+largo] para cada p (vectorizado)."""
    vistas = np.lib.stride_tricks.sliding_window_view(arr, largo, axis=0)  # (n-largo+1, F, largo)
    return np.ascontiguousarray(vistas[posiciones + desde].transpose(0, 2, 1))


def cargar_datos_modelo(ventana: int = config.VENTANA, horizonte: int = config.HORIZONTE) -> DatosModelo:
    """Carga `data/processed`, escala con estadísticos de train y arma las ventanas."""
    df = cargar_tabla(config.ARCHIVO_FEATURES).sort_index().astype("float32")
    idx = pd.DatetimeIndex(df.index)

    escalador = Escalador().ajustar(df.loc[: config.FIN_TRAIN])
    dfz = escalador.transformar(df)

    col_obj = list(config.OBJETIVOS)
    col_fut = list(COVARIABLES_FUTURAS)
    col_pas = col_obj + list(COVARIABLES_PASADAS) + col_fut

    A_pas = dfz[col_pas].to_numpy("float32")
    A_fut = dfz[col_fut].to_numpy("float32")
    A_y = dfz[col_obj].to_numpy("float32")

    splits = {}
    for nombre, pos in _indices_t0(idx, ventana, horizonte).items():
        splits[nombre] = Split(
            pasado=_ventanas(A_pas, pos, -ventana, ventana),
            futuro=_ventanas(A_fut, pos, 0, horizonte),
            y=_ventanas(A_y, pos, 0, horizonte),
            t0=idx[pos],
        )
    return DatosModelo(splits, escalador, col_pas, col_fut, col_obj, ventana, horizonte)
