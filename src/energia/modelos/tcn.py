"""TCN (Temporal Convolutional Network) para el pronóstico day-ahead de demanda y precio.

Codificador: bloques residuales de convoluciones causales y dilatadas
(1, 2, 4, ..., 2^(niveles-1)) sobre las 168 h de historia (`pasado`).
Cabeza: cada hora h del día a pronosticar usa la salida de la TCN y los valores
observados en la misma hora del día anterior, más su calendario (`futuro`).
Una red densa compartida por las 24 horas produce las 2 salidas de cada hora.
Con `corregir_naive=True` la salida se suma a la línea base naive de cada serie
(demanda de hace 168 h y precio de hace 24 h); por defecto la red predice el valor completo.
"""
from __future__ import annotations

import torch
from torch import nn

HIPERPARAMETROS = {"canales": 32, "kernel": 3, "niveles": 6, "dropout": 0.1, "cabeza": 64,
                   "corregir_naive": False}


class ConvCausal(nn.Module):
    def __init__(self, entrada: int, salida: int, kernel: int, dilatacion: int):
        super().__init__()
        self.recorte = (kernel - 1) * dilatacion
        self.conv = nn.Conv1d(entrada, salida, kernel, dilation=dilatacion, padding=self.recorte)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x[:, :, :-self.recorte] if self.recorte else x


class BloqueResidual(nn.Module):
    def __init__(self, entrada: int, salida: int, kernel: int, dilatacion: int, dropout: float):
        super().__init__()
        self.bloque = nn.Sequential(
            ConvCausal(entrada, salida, kernel, dilatacion),
            nn.ReLU(),
            nn.Dropout(dropout),
            ConvCausal(salida, salida, kernel, dilatacion),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.atajo = nn.Conv1d(entrada, salida, 1) if entrada != salida else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bloque(x) + self.atajo(x))


class TCNPronostico(nn.Module):
    def __init__(self, n_pasado: int, n_futuro: int, horizonte: int, n_objetivos: int,
                 canales: int = 32, kernel: int = 3, niveles: int = 6, dropout: float = 0.1,
                 cabeza: int = 64, corregir_naive: bool = False):
        super().__init__()
        self.horizonte = horizonte
        self.corregir_naive = corregir_naive
        self.tcn = nn.Sequential(*[
            BloqueResidual(n_pasado if i == 0 else canales, canales, kernel, 2 ** i, dropout)
            for i in range(niveles)
        ])
        self.cabeza = nn.Sequential(
            nn.Linear(canales + n_pasado + n_futuro, cabeza),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cabeza, n_objetivos),
        )
        if corregir_naive:
            nn.init.zeros_(self.cabeza[-1].weight)
            nn.init.zeros_(self.cabeza[-1].bias)

    def forward(self, pasado: torch.Tensor, futuro: torch.Tensor) -> torch.Tensor:
        h = self.horizonte
        contexto = self.tcn(pasado.transpose(1, 2))[:, :, -h:].transpose(1, 2)
        ayer = pasado[:, -h:, :]
        y = self.cabeza(torch.cat([contexto, ayer, futuro], dim=-1))
        if self.corregir_naive:
            y = y + torch.stack([pasado[:, -168:-168 + h, 0], pasado[:, -h:, 1]], dim=-1)
        return y


def construir(datos, **hp) -> TCNPronostico:
    parametros = {**HIPERPARAMETROS, **hp}
    if parametros["corregir_naive"]:
        assert datos.columnas_pasado[:2] == datos.columnas_objetivo and datos.ventana >= 168
    return TCNPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        horizonte=datos.horizonte,
        n_objetivos=len(datos.columnas_objetivo),
        **parametros,
    )
