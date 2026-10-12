"""Entrenamiento y evaluación comunes para las redes de la fase 2 (PyTorch).

Todas las redes reciben `forward(pasado, futuro)` y devuelven un tensor
(batch, HORIZONTE, 2) en escala estandarizada. Se entrenan con MSE y early
stopping sobre validación, y se evalúan en unidades originales.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .metricas import evaluar


def fijar_semilla(semilla: int = config.SEMILLA) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def cargador(datos, split: str, batch: int = 256, mezclar: bool = False) -> DataLoader:
    s = datos.splits[split]
    ds = TensorDataset(torch.from_numpy(s.pasado), torch.from_numpy(s.futuro), torch.from_numpy(s.y))
    return DataLoader(ds, batch_size=batch, shuffle=mezclar)


OPTIMIZADORES = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "nadam", "adamax"]


def crear_optimizador(nombre: str, parametros, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    nombre = nombre.lower()
    o = torch.optim
    if nombre == "sgd":
        return o.SGD(parametros, lr=lr, weight_decay=weight_decay)
    if nombre == "momentum":
        return o.SGD(parametros, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if nombre == "nesterov":
        return o.SGD(parametros, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    clases = {"adagrad": o.Adagrad, "rmsprop": o.RMSprop, "adam": o.Adam, "nadam": o.NAdam, "adamax": o.Adamax}
    if nombre not in clases:
        raise ValueError(f"Optimizador desconocido '{nombre}'. Opciones: {OPTIMIZADORES}")
    return clases[nombre](parametros, lr=lr, weight_decay=weight_decay)


def _pasada(modelo, carga, perdida, disp, optimizador=None, clip=None) -> float:
    entrenando = optimizador is not None
    modelo.train(entrenando)
    total, n = 0.0, 0
    with torch.set_grad_enabled(entrenando):
        for pasado, futuro, y in carga:
            pasado, futuro, y = pasado.to(disp), futuro.to(disp), y.to(disp)
            pred = modelo(pasado, futuro)
            valor = perdida(pred, y)
            if entrenando:
                optimizador.zero_grad()
                valor.backward()
                if clip:
                    nn.utils.clip_grad_norm_(modelo.parameters(), clip)
                optimizador.step()
            total += valor.item() * len(y)
            n += len(y)
    return total / n


def entrenar(
    modelo: nn.Module,
    datos,
    optimizador: str = "adam",
    lr: float = 1e-3,
    epocas: int = 50,
    batch: int = 256,
    paciencia: int = 8,
    clip: float | None = 1.0,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entrena con MSE y early stopping sobre validación.

    Al terminar, el modelo queda con los pesos de la época con menor pérdida
    de validación. Devuelve el historial por época.
    """
    disp = dispositivo()
    modelo.to(disp)
    opt = crear_optimizador(optimizador, modelo.parameters(), lr, weight_decay)
    perdida = nn.MSELoss()
    carga_train = cargador(datos, "train", batch, mezclar=True)
    carga_val = cargador(datos, "val", batch)

    mejor, mejor_estado, sin_mejora, historial = math.inf, None, 0, []
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        l_train = _pasada(modelo, carga_train, perdida, disp, opt, clip)
        l_val = _pasada(modelo, carga_val, perdida, disp)
        segundos = time.time() - inicio
        historial.append({"epoca": epoca, "train": l_train, "val": l_val, "segundos": segundos})
        mejora = l_val < mejor
        if mejora:
            mejor = l_val
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
        if verbose:
            marca = " *" if mejora else ""
            print(f"época {epoca:3d} | train {l_train:.4f} | val {l_val:.4f} | {segundos:5.1f} s{marca}")
        if sin_mejora >= paciencia:
            if verbose:
                print(f"Early stopping: {paciencia} épocas sin mejorar en validación.")
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return pd.DataFrame(historial).set_index("epoca")


def predecir(modelo: nn.Module, datos, split: str, batch: int = 1024) -> np.ndarray:
    """Predicción en unidades originales (kWh, COP/kWh), forma (N, HORIZONTE, 2)."""
    disp = dispositivo()
    modelo.to(disp).eval()
    salidas = []
    with torch.no_grad():
        for pasado, futuro, _ in cargador(datos, split, batch):
            salidas.append(modelo(pasado.to(disp), futuro.to(disp)).cpu().numpy())
    return datos.invertir(np.concatenate(salidas))


def evaluar_modelo(modelo: nn.Module, datos, split: str) -> dict[str, float]:
    return evaluar(datos.y_original(split), predecir(modelo, datos, split))


def guardar_modelo(modelo: nn.Module, nombre: str, hiperparametros: dict) -> str:
    config.MODELOS.mkdir(parents=True, exist_ok=True)
    ruta = config.MODELOS / f"{nombre}.pt"
    torch.save({"estado": modelo.state_dict(), "hiperparametros": hiperparametros}, ruta)
    return str(ruta)


def guardar_resultados(nombre: str, metricas: pd.DataFrame, hiperparametros: dict, historial: pd.DataFrame) -> str:
    """Guarda métricas, hiperparámetros e historial en results/fase2/<nombre>/."""
    carpeta = config.RESULTADOS / "fase2" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(carpeta / "metricas.csv")
    historial.to_csv(carpeta / "historial.csv")
    (carpeta / "hiperparametros.json").write_text(json.dumps(hiperparametros, indent=2, ensure_ascii=False, default=str))
    return str(carpeta)
