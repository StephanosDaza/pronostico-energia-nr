"""Transformer codificador-decodificador para el pronóstico day-ahead de demanda y precio.

Codificador: autoatención sobre las 168 h de historia (`pasado`).
Decodificador: las 24 h a pronosticar, descritas por su calendario (`futuro`),
consultan la historia codificada con atención cruzada y salen de una sola vez.
La salida es una corrección que se suma a la línea base naive de cada serie
(demanda de hace 168 h y precio de hace 24 h, ambas presentes en `pasado`).
"""
from __future__ import annotations

import torch
from torch import nn

HIPERPARAMETROS = {"d_model": 64, "nhead": 4, "capas_codificador": 2, "capas_decodificador": 1,
                   "dim_ff": 128, "dropout": 0.2, "corregir_naive": True}


class TransformerPronostico(nn.Module):
    def __init__(self, n_pasado: int, n_futuro: int, n_salidas: int, ventana: int, horizonte: int,
                 d_model: int = 64, nhead: int = 4, capas_codificador: int = 2, capas_decodificador: int = 1,
                 dim_ff: int = 128, dropout: float = 0.2, corregir_naive: bool = True,
                 rezagos_naive: tuple[int, ...] = (168, 24)):
        super().__init__()
        self.horizonte = horizonte
        self.entrada_pasado = nn.Linear(n_pasado, d_model)
        self.entrada_futuro = nn.Linear(n_futuro, d_model)
        self.pos_pasado = nn.Parameter(torch.randn(1, ventana, d_model) * 0.02)
        self.pos_futuro = nn.Parameter(torch.randn(1, horizonte, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        capa_cod = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                              batch_first=True, norm_first=True)
        self.codificador = nn.TransformerEncoder(capa_cod, capas_codificador, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)
        capa_dec = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, activation="gelu",
                                              batch_first=True, norm_first=True)
        self.decodificador = nn.TransformerDecoder(capa_dec, capas_decodificador, norm=nn.LayerNorm(d_model))
        self.salida = nn.Linear(d_model, n_salidas)

        # posición, dentro de la ventana, donde empieza la línea base de cada objetivo
        self.inicios_naive = [ventana - m for m in rezagos_naive] if corregir_naive else None
        if corregir_naive:
            # con la salida en cero, la red sin entrenar pronostica exactamente el naive
            nn.init.zeros_(self.salida.weight)
            nn.init.zeros_(self.salida.bias)

    def linea_base(self, pasado: torch.Tensor) -> torch.Tensor:
        return torch.stack([pasado[:, ini:ini + self.horizonte, j] for j, ini in enumerate(self.inicios_naive)], dim=-1)

    def forward(self, pasado: torch.Tensor, futuro: torch.Tensor) -> torch.Tensor:
        historia = self.dropout(self.entrada_pasado(pasado) + self.pos_pasado)
        memoria = self.codificador(historia)
        consultas = self.dropout(self.entrada_futuro(futuro) + self.pos_futuro)
        y = self.salida(self.decodificador(consultas, memoria))
        if self.inicios_naive is not None:
            y = y + self.linea_base(pasado)
        return y


def construir(datos, **hp) -> TransformerPronostico:
    parametros = {**HIPERPARAMETROS, **hp}
    n_salidas = len(datos.columnas_objetivo)
    # la línea base se lee de las primeras columnas de `pasado`, que son los objetivos
    assert datos.columnas_pasado[:n_salidas] == datos.columnas_objetivo
    return TransformerPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        n_salidas=n_salidas,
        ventana=datos.ventana,
        horizonte=datos.horizonte,
        **parametros,
    )
