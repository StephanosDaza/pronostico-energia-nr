"""LSTM para el pronóstico day-ahead de demanda y precio.

Codificador: LSTM apilada (Long Short-Term Memory) sobre las 168 h de historia (`pasado`).
Cabeza: el último estado oculto se concatena con el calendario de las 24 h a
pronosticar (`futuro`) y una red densa produce las 24 x 2 salidas.
"""
from __future__ import annotations

import torch
from torch import nn

# Hiperparámetros por defecto para la arquitectura LSTM
HIPERPARAMETROS = {"oculto": 64, "capas": 2, "dropout": 0.1, "cabeza": 128}


class LSTMPronostico(nn.Module):
    """Modelo de pronóstico multivariado usando un codificador LSTM y cabeza MLP."""
    def __init__(self, n_pasado: int, n_futuro: int, horizonte: int, n_objetivos: int,
                 oculto: int = 64, capas: int = 2, dropout: float = 0.1, cabeza: int = 128):
        super().__init__()
        self.horizonte = horizonte
        self.n_objetivos = n_objetivos
        
        # Codificador recurrente LSTM
        self.lstm = nn.LSTM(
            input_size=n_pasado,
            hidden_size=oculto,
            num_layers=capas,
            batch_first=True,
            dropout=dropout if capas > 1 else 0.0,
        )
        
        # Cabeza de regresión densa para proyectar al horizonte x objetivos (24 x 2)
        self.cabeza = nn.Sequential(
            nn.Linear(oculto + horizonte * n_futuro, cabeza),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cabeza, horizonte * n_objetivos),
        )

    def forward(self, pasado: torch.Tensor, futuro: torch.Tensor) -> torch.Tensor:
        # pasado: (batch, 168, n_pasado)
        # futuro: (batch, 24, n_futuro)
        # nn.LSTM retorna: salida_secuencia, (h_n, c_n)
        # h_n tiene dimensiones (capas, batch, oculto). Tomamos la última capa h_n[-1]
        _, (h, _) = self.lstm(pasado)
        
        # Concatenar el estado oculto final con las variables exógenas futuras aplanadas
        z = torch.cat([h[-1], futuro.flatten(start_dim=1)], dim=1)
        
        # Proyectar y redimensionar a (batch, horizonte, n_objetivos) -> (batch, 24, 2)
        return self.cabeza(z).view(-1, self.horizonte, self.n_objetivos)


def construir(datos, **hp) -> LSTMPronostico:
    """Construye e inicializa el modelo LSTMPronostico según los metadatos de datos."""
    parametros = {**HIPERPARAMETROS, **hp}
    return LSTMPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        horizonte=datos.horizonte,
        n_objetivos=len(datos.columnas_objetivo),
        **parametros,
    )
