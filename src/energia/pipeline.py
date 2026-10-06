"""Orquestador del pipeline de datos de la Fase 1.

    raw (CSV XM) --preprocesamiento--> interim/dataset_horario
                 --features---------> processed/dataset_features  (+ escalador.json)

Uso:  python scripts/construir_dataset.py      o      from energia.pipeline import construir_dataset
"""
from __future__ import annotations

from . import config, features, preprocesamiento
from .dataset import Escalador


def construir_dataset() -> None:
    preprocesamiento.ejecutar()
    df = features.ejecutar()
    Escalador().ajustar(df.loc[: config.FIN_TRAIN]).guardar()
    print(f"Escalador (ajustado solo con train) -> {config.ARCHIVO_ESCALADOR.relative_to(config.RAIZ)}")
