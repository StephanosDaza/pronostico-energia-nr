# Pronóstico de Demanda No Regulada y Precio de bolsa en Colombia

Proyecto del curso Optimización e Inteligencia Artificial (UNAL, 2026).

Pronóstico horario, para las 24 horas del día siguiente, de la demanda de energía de usuarios no regulados y del precio de bolsa nacional, con datos de XM (2022-01-01 a 2026-08-31).

La documentación de la fase 1 está en **`guia_proyecto.html`** (fuente: `guia_proyecto.qmd`), la de la fase 2 en **`guia_fase2.html`** (fuente: `guia_fase2.qmd`) y la de la fase 3 en **`guia_fase3.html`** (fuente: `guia_fase3.qmd`). El documento final del proyecto y presenta el modelo ganador es **`entregable_final_proyecto.html`** (fuente: `entregable_final_proyecto.qmd`). Las instrucciones del curso están en `docs/guia_proyecto_final_curso.pdf`.

## Ejecución (VS Code)

Todos trabajamos con **Python 3.11**, **VS Code** y las versiones fijadas en `requirements.txt`. Cada uno crea su propio entorno virtual (`.venv/`), que no se sube a git.

### 1. Requisitos

- **Python 3.11**
  - Windows: <https://www.python.org/downloads/>; marcar *Add python.exe to PATH*. Verificar con `py -3.11 --version`.
  - macOS: `brew install python@3.11`. Verificar con `python3.11 --version`.
  - Linux: `sudo apt install python3.11 python3.11-venv`. Verificar con `python3.11 --version`.
- **VS Code** con las extensiones *Python* y *Jupyter*. Al abrir el proyecto, VS Code las sugiere solo (`.vscode/extensions.json`).

### 2. Clonar y abrir el proyecto

```bash
git clone <url-del-repositorio>
cd pronostico-energia-nr
code .
```

### 3. Crear el entorno (una sola vez)

Si ya existe `requirements-lock.txt` en el repositorio, se usa ese archivo en lugar de `requirements.txt` en el comando `pip install -r`. La guía (sección "Primeros pasos para cada integrante") tiene el paso a paso completo para el equipo.

En VS Code se abre la terminal (*Terminal → New Terminal*) y se ejecuta lo siguiente.

Windows (cmd). VS Code abre PowerShell por defecto; para usar cmd: `Ctrl+Shift+P` → *Terminal: Select Default Profile* → *Command Prompt*.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Seleccionar el entorno en VS Code

- **Para la terminal y los scripts:** `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir el que dice `.venv`. Desde ahí, cada terminal nueva de VS Code abre con el entorno activo y muestra `(.venv)` al inicio de la línea.
- **Para los cuadernos:** al abrir un `.ipynb`, en la esquina superior derecha *Select Kernel* → *Python Environments* → `.venv`.

### 5. Construir los datos

```bash
python scripts/construir_dataset.py
```

Después se pueden abrir y ejecutar los cuadernos de `notebooks/fase1_datos/`.

### GPU (opcional)

Con `requirements.txt`, PyTorch se instala para CPU en Windows y macOS, y con soporte CUDA en Linux. Quien tenga una GPU NVIDIA en Windows y quiera usarla ejecuta, con el entorno activo:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Para verificarlo: `python -c "import torch; print(torch.cuda.is_available())"` debe imprimir `True`.

### Fijar todas las dependencias

Esteban crea el entorno y ejecuta:

```bash
pip freeze --exclude-editable > requirements-lock.txt
```

y sube ese archivo. Desde ahí, todos instalamos con `pip install -r requirements-lock.txt` en lugar de `requirements.txt`, y quedamos con versiones idénticas, incluidas las dependencias indirectas.

### Agregar una librería

Se instala con versión fija (`pip install <libreria>==<version>`), se agrega a `requirements.txt` y se regenera el lock con el comando anterior. Los demás, con su entorno activo, ejecutan `pip install -r requirements-lock.txt`.

## Estructura

```
.python-version               versión de Python del proyecto (3.11)
.vscode/extensions.json       extensiones recomendadas de VS Code
requirements.txt              librerías con versión fijada
data/raw/                     datos originales de XM
data/interim/                 serie horaria estandarizada
data/processed/               dataset con variables para los modelos
notebooks/fase1_datos/        descarga, estandarización, análisis estadístico, feature engineering
notebooks/fase2_redes_neuronales/  un cuaderno por arquitectura (01_rnn.ipynb, ...)
notebooks/fase3_optimizacion/
src/energia/                  código del proyecto
src/energia/modelos/          arquitecturas de la fase 2
results/fase2/                métricas e historial de cada red
scripts/construir_dataset.py  pipeline de datos completo
reports/figuras/              figuras
docs/                         guía del curso, propuesta inicial y cuadernos originales
```
