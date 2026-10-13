
"""GRU con atención temporal para el pronóstico energético day-ahead.

Codificador: GRU apilada sobre las 168 horas históricas.

Atención: utiliza el calendario futuro para obtener un contexto
temporal específico para cada una de las 24 horas a pronosticar.

Predicción: una cabeza neuronal aprende correcciones sobre
referencias estacionales históricas de demanda y precio.

Salida conjunta: (batch, 24, 2), en escala transformada.
"""

from __future__ import annotations
import math
import torch
from torch import nn

HIPERPARAMETROS = {"oculto": 48,"capas": 2,"dropout": 0.15,"atencion": 32,"cabeza": 64,}

class GRUPronostico(nn.Module):
    """GRU con atención y corrección estacional multihorizonte."""
    def __init__(self,n_pasado: int,n_futuro: int,horizonte: int,n_objetivos: int,ventana: int,oculto: int = 48,
                capas: int = 2,dropout: float = 0.15,atencion: int = 32,cabeza: int = 64,):
        super().__init__()
        if n_objetivos != 2:
            raise ValueError(
                "Se requieren dos objetivos: demanda y precio.")
        if horizonte != 24 or ventana < 168:
            raise ValueError(
                "Se requiere horizonte=24 y ventana >= 168.")
        if min(oculto, capas, atencion, cabeza) < 1:
            raise ValueError(
                "Las dimensiones deben ser positivas.")

        self.horizonte = horizonte
        self.n_objetivos = n_objetivos
        # Codificador recurrente
        self.gru = nn.GRU(input_size=n_pasado,hidden_size=oculto,num_layers=capas,batch_first=True,dropout=dropout if capas > 1 else 0.0,)
        
        #  Proyecciones para atención temporal
        self.consulta = nn.Linear(oculto + n_futuro,atencion,)
        self.clave = nn.Linear(oculto, atencion,bias=False,)

        #Cabeza de predicción compartida por hora
        n_entrada = (2 * oculto+ n_futuro+ 2 * n_objetivos)

        self.cabeza = nn.Sequential(nn.Linear(n_entrada, cabeza),nn.ReLU(),nn.Dropout(dropout),nn.Linear(cabeza, n_objetivos),)
        # Inicialización residual:
        # al comenzar, la predicción coincide con la
        # referencia estacional; luego aprende correcciones.
        nn.init.zeros_(self.cabeza[-1].weight)
        nn.init.zeros_(self.cabeza[-1].bias)

    def forward(self,pasado: torch.Tensor,futuro: torch.Tensor,) -> torch.Tensor:
        secuencia, estado_final = self.gru(pasado)
        h = futuro.shape[1]
        memoria = (estado_final[-1].unsqueeze(1).expand(-1, h, -1))

        # Atención condicionada por cada hora futura
        entrada_consulta = torch.cat([memoria, futuro],dim=-1,)

        q = torch.tanh(self.consulta(entrada_consulta))
        k = self.clave(secuencia)

        puntajes = torch.bmm(q,k.transpose(1, 2),) / math.sqrt(q.shape[-1])
        pesos = torch.softmax(puntajes, dim=-1)
        contexto = torch.bmm(pesos,secuencia,)

        # Referencias históricas disponibles
        ayer = pasado[:, -24:, :self.n_objetivos]

        semana = pasado[:, -168:-168 + h, :self.n_objetivos]

        # Demanda: referencia semanal
        # Precio: referencia diaria
        referencia = torch.stack(
            [semana[..., 0],ayer[..., 1],],dim=-1,)

        # Integración de las representaciones
        entrada = torch.cat([contexto,memoria,futuro,ayer,semana,],dim=-1,)
        # Predicción residual conjunta
        correccion = self.cabeza(entrada)
        return referencia + correccion

def construir(datos, **hp) -> GRUPronostico:
    """Construye la GRU usando los metadatos del dataset."""

    objetivos = ["demanda_kwh","precio_bolsa_cop_kwh",]

    if datos.columnas_pasado[:2] != objetivos:
        raise ValueError(
            "Las primeras columnas deben ser demanda y precio.")
    parametros = {**HIPERPARAMETROS, **hp}
    
    return GRUPronostico(
        n_pasado=len(datos.columnas_pasado),
        n_futuro=len(datos.columnas_futuro),
        horizonte=datos.horizonte,
        n_objetivos=len(datos.columnas_objetivo),
        ventana=datos.ventana,
        **parametros,)
