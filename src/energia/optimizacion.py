"""Optimización de la fase 3.

Entrena cada configuración, la evalúa en validación y guarda sus resultados
en results/fase3/<red>/. 
"""

from __future__ import annotations

import json
import time

import pandas as pd
import torch

from . import config
from .entrenamiento import contar_parametros, entrenar, fijar_semilla, predecir
from .metricas import evaluar

METRICA = "val_precio_MAPE_%"
REQUISITO = ("val_demanda_MAPE_%", 3.83)


def carpeta(red: str):
    ruta = config.RESULTADOS / "fase3" / red
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def registro(red: str) -> pd.DataFrame:
    ruta = carpeta(red) / "experimentos.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def _ya_existe(red: str, etapa: str, nombre: str, semilla: int) -> pd.Series | None:
    reg = registro(red)
    if reg.empty:
        return None
    filas = reg[(reg["etapa"] == etapa) & (reg["nombre"] == nombre) & (reg["semilla"] == semilla)]
    return None if filas.empty else filas.iloc[-1]


def _ruta_modelo(red: str, etapa: str, nombre: str, semilla: int):
    return config.MODELOS / f"{red}_{etapa}_{nombre}_s{semilla}.pt"


def experimento(red: str, etapa: str, nombre: str, modulo, datos, hp_modelo: dict, hp_entrenamiento: dict,
                semilla: int = config.SEMILLA, prueba: bool = False, guardar: bool = False,
                verbose: bool = False) -> pd.Series:
    """Entrena y registra una configuración. Devuelve su fila del registro."""
    previa = _ya_existe(red, etapa, nombre, semilla)
    if previa is not None:
        print(f"[{etapa}] {nombre} (semilla {semilla}): ya registrado, {METRICA} = {previa[METRICA]:.2f}")
        return previa

    fijar_semilla(semilla)
    modelo = modulo.construir(datos, **hp_modelo)
    inicio = time.time()
    historial = entrenar(modelo, datos, verbose=verbose, **hp_entrenamiento)
    minutos = (time.time() - inicio) / 60

    fila = {
        "etapa": etapa, "nombre": nombre, "semilla": semilla,
        "parametros": contar_parametros(modelo),
        "mejor_epoca": int(historial["val"].idxmin()), "epocas": len(historial),
        "val_loss": float(historial["val"].min()), "minutos": round(minutos, 1),
        "hp_modelo": json.dumps(hp_modelo), "hp_entrenamiento": json.dumps(hp_entrenamiento),
    }
    fila.update({f"val_{k}": v for k, v in evaluar(datos.y_original("val"), predecir(modelo, datos, "val")).items()})
    if prueba:
        fila.update({f"test_{k}": v for k, v in evaluar(datos.y_original("test"), predecir(modelo, datos, "test")).items()})

    historiales = carpeta(red) / "historiales"
    historiales.mkdir(exist_ok=True)
    historial.to_csv(historiales / f"{etapa}__{nombre}__s{semilla}.csv")
    if guardar:
        config.MODELOS.mkdir(parents=True, exist_ok=True)
        torch.save({"estado": modelo.state_dict(), "hp_modelo": hp_modelo},
                   _ruta_modelo(red, etapa, nombre, semilla))

    ruta = carpeta(red) / "experimentos.csv"
    reg = pd.concat([registro(red), pd.DataFrame([fila])], ignore_index=True)
    reg.to_csv(ruta, index=False)
    print(f"[{etapa}] {nombre} (semilla {semilla}): {METRICA} = {fila[METRICA]:.2f} | "
          f"demanda {fila['val_demanda_MAPE_%']:.2f} | época {fila['mejor_epoca']}/{fila['epocas']} | {minutos:.0f} min")
    return pd.Series(fila)


def etapa(red: str, nombre_etapa: str) -> pd.DataFrame:
    """Resultados de una etapa, ordenados por la métrica de selección."""
    reg = registro(red)
    tabla = reg[reg["etapa"] == nombre_etapa].copy()
    columnas = ["nombre", "semilla", METRICA, "val_demanda_MAPE_%", "val_precio_RMSE", "val_precio_R2",
                "val_loss", "mejor_epoca", "epocas", "minutos"]
    columnas += [c for c in tabla.columns if c.startswith("test_") and tabla[c].notna().any()]
    return tabla[columnas].sort_values(METRICA).reset_index(drop=True)


def mejor(red: str, etapas: str | list[str]) -> pd.Series:
    """Mejor configuración de una o varias etapas: cumple el requisito de demanda y tiene el menor MAPE de precio."""
    etapas = [etapas] if isinstance(etapas, str) else etapas
    reg = registro(red)
    tabla = reg[reg["etapa"].isin(etapas)]
    cumplen = tabla[tabla[REQUISITO[0]] < REQUISITO[1]]
    if cumplen.empty:
        print(f"Ninguna configuración cumple {REQUISITO[0]} < {REQUISITO[1]}; se escoge entre todas.")
    else:
        tabla = cumplen
    fila = tabla.sort_values(METRICA).iloc[0].copy()
    fila["hp_modelo"] = json.loads(fila["hp_modelo"])
    fila["hp_entrenamiento"] = json.loads(fila["hp_entrenamiento"])
    return fila


def historial(red: str, nombre_etapa: str, nombre: str, semilla: int = config.SEMILLA) -> pd.DataFrame:
    ruta = carpeta(red) / "historiales" / f"{nombre_etapa}__{nombre}__s{semilla}.csv"
    return pd.read_csv(ruta, index_col="epoca")


def cargar_modelo(red: str, nombre_etapa: str, nombre: str, modulo, datos, semilla: int = config.SEMILLA):
    paquete = torch.load(_ruta_modelo(red, nombre_etapa, nombre, semilla), map_location="cpu", weights_only=False)
    modelo = modulo.construir(datos, **paquete["hp_modelo"])
    modelo.load_state_dict(paquete["estado"])
    return modelo
