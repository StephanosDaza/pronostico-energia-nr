"""RNN simple (Elman) para el pronóstico day-ahead de demanda y precio.

Codificador: RNN apilada sobre las 168 h de historia (`pasado`).
Cabeza: el último estado oculto se concatena con el calendario de las 24 h a
pronosticar (`futuro`) y una red densa produce las 24 x 2 salidas.
"""
from __future__ import annotations

import torch
from torch import nn

HIPERPARAMETROS = {"oculto": 64, "capas": 2, "dropout": 0.1, "cabeza": 128}


class RNNPronostico(nn.Module):
    def __init__(self, n_pasado: int, n_futuro: int, horizonte: int, n_objetivos: int,
                 oculto: int = 64, capas: int = 2, dropout: float = 0.1, cabeza: int = 128):
        super().__init__()
        self.horizonte = horizonte
        self.n_objetivos = n_objetivos
        self.rnn = nn.RNN(
            input_size=n_pasado,
            hidden_size=oculto,
            num_layers=capas,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if capas > 1 else 0.0,
        )
        self.cabeza = nn.Sequential(
            nn.Linear(oculto + horizonte * n_futuro, cabeza),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cabeza, horizonte * n_objetivos),
        )

    def forward(self, pasado: torch.Tensor, futuro: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(pasado)
        z = torch.cat([h[-1], futuro.flatten(start_dim=1)], dim=1)
        return self.cabeza(z).view(-1, self.horizonte, self.n_objetivos)


def construir(datos, **hp) -> RNNPronostico:
    parametros = {**HIPERPARAMETROS, **hp}
    return RNNPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        horizonte=datos.horizonte,
        n_objetivos=len(datos.columnas_objetivo),
        **parametros,
    )
