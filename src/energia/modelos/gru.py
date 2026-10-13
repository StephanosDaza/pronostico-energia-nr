"""GRU con atención temporal para el pronóstico day-ahead de demanda y precio.

Codificador: GRU apilada sobre las 168 h de historia (`pasado`).
Atención: cada una de las 24 horas a pronosticar, descrita por su calendario
(`futuro`) y el último estado oculto, arma su propio contexto a partir de las
168 salidas de la GRU.
Cabeza: una red densa compartida por las 24 horas combina el contexto, el último
estado, el calendario y los valores de demanda y precio de la misma hora de
ayer y de la semana pasada.
"""
from __future__ import annotations

import math

import torch
from torch import nn

HIPERPARAMETROS = {"oculto": 48, "capas": 2, "dropout": 0.15, "atencion": 32, "cabeza": 64,
                   "corregir_naive": False}


class GRUPronostico(nn.Module):
    def __init__(self, n_pasado: int, n_futuro: int, horizonte: int, n_objetivos: int,
                 oculto: int = 48, capas: int = 2, dropout: float = 0.15, atencion: int = 32,
                 cabeza: int = 64, corregir_naive: bool = False):
        super().__init__()
        self.horizonte = horizonte
        self.n_objetivos = n_objetivos
        self.corregir_naive = corregir_naive
        self.gru = nn.GRU(input_size=n_pasado, hidden_size=oculto, num_layers=capas, batch_first=True,
                          dropout=dropout if capas > 1 else 0.0)
        self.consulta = nn.Linear(oculto + n_futuro, atencion)
        self.clave = nn.Linear(oculto, atencion, bias=False)
        self.cabeza = nn.Sequential(
            nn.Linear(2 * oculto + n_futuro + 2 * n_objetivos, cabeza),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cabeza, n_objetivos),
        )
        if corregir_naive:
            nn.init.zeros_(self.cabeza[-1].weight)
            nn.init.zeros_(self.cabeza[-1].bias)

    def forward(self, pasado: torch.Tensor, futuro: torch.Tensor) -> torch.Tensor:
        h = self.horizonte
        secuencia, estado = self.gru(pasado)
        memoria = estado[-1].unsqueeze(1).expand(-1, h, -1)
        q = torch.tanh(self.consulta(torch.cat([memoria, futuro], dim=-1)))
        k = self.clave(secuencia)
        pesos = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
        contexto = torch.bmm(pesos, secuencia)
        ayer = pasado[:, -h:, :self.n_objetivos]
        semana = pasado[:, -168:-168 + h, :self.n_objetivos]
        y = self.cabeza(torch.cat([contexto, memoria, futuro, ayer, semana], dim=-1))
        if self.corregir_naive:
            y = y + torch.stack([semana[..., 0], ayer[..., 1]], dim=-1)
        return y


def construir(datos, **hp) -> GRUPronostico:
    parametros = {**HIPERPARAMETROS, **hp}
    assert datos.columnas_pasado[:2] == datos.columnas_objetivo and datos.ventana >= 168
    return GRUPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        horizonte=datos.horizonte,
        n_objetivos=len(datos.columnas_objetivo),
        **parametros,
    )
