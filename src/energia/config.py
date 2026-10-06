"""Parámetros comunes del proyecto."""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

DATA = RAIZ / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FIGURAS = RAIZ / "reports" / "figuras"

ARCHIVO_HORARIO = INTERIM / "dataset_horario.parquet"
ARCHIVO_FEATURES = PROCESSED / "dataset_features.parquet"
ARCHIVO_ESCALADOR = PROCESSED / "escalador.json"

FECHA_INICIO = "2022-01-01"
FECHA_FIN = "2026-08-31"

VARIABLES_XM = {
    "DemaComeNoReg":      ("demanda_no_regulada_2022_2026.csv", "demanda_kwh",             "horaria"),
    "PrecBolsNaci":       ("precio_bolsa_2022_2026.csv",        "precio_bolsa_cop_kwh",    "horaria"),
    "PrecPromContNoRegu": ("precios_contratos_2022_2026.csv",   "precio_contrato_cop_kwh", "horaria"),
    "AporEner":           ("aportes_hidricos_2022_2026.csv",    "aportes_kwh",             "diaria"),
    "VoluUtilDiarEner":   ("volumen_util_2022_2026.csv",        "volumen_util_kwh",        "diaria"),
}

OBJETIVOS = ["demanda_kwh", "precio_bolsa_cop_kwh"]
EXOGENAS_HORARIAS = ["precio_contrato_cop_kwh"]
EXOGENAS_DIARIAS = ["aportes_kwh", "volumen_util_kwh"]
REZAGO_EXOGENAS_DIARIAS = 1

COLUMNAS_LOG = ["precio_bolsa_cop_kwh", "aportes_kwh", "aportes_media7_kwh"]

HORIZONTE = 24
VENTANA = 168
PASO_ENTRENAMIENTO = 1
PASO_EVALUACION = 24

FIN_TRAIN = "2025-06-30 23:00"
FIN_VAL = "2025-12-31 23:00"

SEMILLA = 42
