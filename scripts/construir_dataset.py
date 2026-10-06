"""Ejecutar el pipeline completo de la Fase 1 desde la línea de comandos.

    python scripts/construir_dataset.py            # raw -> interim -> processed
    python scripts/construir_dataset.py --descargar  # además refresca data/raw desde XM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energia import pipeline, xm  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--descargar", action="store_true", help="descargar de nuevo desde la API de XM")
    args = p.parse_args()
    if args.descargar:
        xm.descargar_todo()
    pipeline.construir_dataset()
