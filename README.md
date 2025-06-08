# Multimodal Recommender System

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg" alt="PyTorch Version">
</div>
<br>

Un framework basado en PyTorch para construir sistemas de recomendación multimodales que integran características visuales, textuales y numéricas para generar recomendaciones personalizadas.

<p align="center">
  <img src="docs/assets/architecture_diagram.png" alt="Architecture Diagram" width="700"/>
  <br>
  <em>Diagrama de la arquitectura del recomendador multimodal.</em>
</p>

---

## Tabla de Contenidos
- [Visión General](#visión-general)
- [Características Clave](#características-clave)
- [Inicio Rápido en 5 Minutos](#inicio-rápido-en-5-minutos)
- [Instalación](#instalación)
- [Flujo de Trabajo Detallado](#flujo-de-trabajo-detallado)
  - [1. Preparación de Datos](#1-preparación-de-datos)
  - [2. Configuración](#2-configuración)
  - [3. Preprocesamiento de Datos](#3-preprocesamiento-de-datos)
  - [4. División de Datos (Splits)](#4-división-de-datos-splits)
  - [5. Pre-cómputo de Caché de Features](#5-pre-cómputo-de-caché-de-features)
  - [6. Entrenamiento](#6-entrenamiento)
  - [7. Evaluación](#7-evaluación)
  - [8. Generación de Recomendaciones](#8-generación-de-recomendaciones)
- [Uso como Librería](#uso-como-librería)
- [Gestión Avanzada](#gestión-avanzada)
  - [Gestión de Caché](#gestión-de-caché)
  - [Gestión de Checkpoints](#gestión-de-checkpoints)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Cómo Contribuir](#cómo-contribuir)
- [Licencia](#licencia)
- [Citar](#citar)

## Visión General

Este sistema implementa un recomendador neuronal que combina múltiples modalidades de datos para superar las limitaciones de los métodos tradicionales de filtrado colaborativo:

-   **Features Visuales**: Extraídas de imágenes de ítems usando modelos pre-entrenados (e.g., ResNet, CLIP, DINO).
-   **Features Textuales**: Procesadas a partir de descripciones de ítems usando modelos de lenguaje (e.g., Sentence-BERT, MPNet).
-   **Features Numéricas**: Metadatos de ítems y estadísticas de interacción.

La arquitectura utiliza mecanismos de atención para fusionar representaciones multimodales y es compatible con el aprendizaje contrastivo para una mejor alineación visión-texto.

## Características Clave

-   **Arquitectura Flexible**: Fusión configurable de embeddings de usuario/ítem y features multimodales.
-   **Modelos Pre-entrenados**: Soporte para una variedad de backbones de visión y lenguaje de Hugging Face.
-   **Procesamiento de Datos Modular**: Pipeline de preprocesamiento robusto con validación, limpieza y compresión automáticas.
-   **Estrategias de División de Datos**: Soporte para splits estratificados, temporales y por usuario/ítem.
-   **Entrenamiento Eficiente**: Optimizadores y schedulers configurables, early stopping, y seguimiento con Weights & Biases.
-   **Evaluación Exhaustiva**: Métricas estándar (Precision, Recall, NDCG, MRR) y comparación con baselines (Popularity, ItemKNN).
-   **Rendimiento Optimizado**: Sistema de caché de features con gestión de memoria LRU para acelerar el entrenamiento y la inferencia.

## Inicio Rápido en 5 Minutos

Sigue estos pasos para tener el sistema funcionando con datos de ejemplo.

```bash
# 1. Clona el repositorio
git clone [https://github.com/tu_usuario/tu_repo.git](https://github.com/tu_usuario/tu_repo.git)
cd tu_repo

# 2. Crea un entorno virtual e instala las dependencias
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Preprocesa los datos de ejemplo
# (Esto limpiará, validará y preparará los datos en data/raw/)
python scripts/preprocess_data.py --config configs/simple_config.yaml

# 4. Crea los splits de entrenamiento/validación/test
python scripts/create_splits.py --config configs/simple_config.yaml

# 5. Entrena el modelo
# (Usa --device cpu si no tienes una GPU compatible con CUDA)
python scripts/train.py --config configs/simple_config.yaml --device cuda

# 6. Evalúa el modelo entrenado
python scripts/evaluate.py --config configs/simple_config.yaml --device cuda
````

## Instalación

### Prerrequisitos

  - Python 3.7+
  - PyTorch 2.2.1+
  - Transformers 4.47.1+
  - Una GPU compatible con CUDA es recomendada para un entrenamiento rápido.

### Pasos de Instalación

1.  **Clona el repositorio:**

    ```bash
    git clone [https://github.com/tu_usuario/tu_repo.git](https://github.com/tu_usuario/tu_repo.git)
    cd tu_repo
    ```

2.  **Crea un entorno virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

## Flujo de Trabajo Detallado

### 1\. Preparación de Datos

Organiza tus datos en la carpeta `data/raw/` con la siguiente estructura:

  - `item_info.csv`: Metadatos de los ítems. Debe contener `item_id` y columnas con features textuales y numéricas.
  - `interactions.csv`: Interacciones usuario-ítem. Requiere columnas `user_id` y `item_id`.
  - `images/`: Un directorio que contenga las imágenes de los ítems, nombradas como `{item_id}.jpg`.

### 2\. Configuración

Edita los archivos de configuración en `configs/` para ajustar los parámetros.

  - **`simple_config.yaml`**: Contiene los parámetros esenciales para empezar. Ideal para experimentos iniciales.
  - **`advanced_config.yaml`**: Ofrece un control granular sobre todos los aspectos del modelo, entrenamiento y datos.

Para más detalles, consulta la [Guía de Configuración](https://www.google.com/search?q=docs/configuration.md).

### 3\. Preprocesamiento de Datos

Este script valida, limpia y procesa los datos crudos.

```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

### 4\. División de Datos (Splits)

Crea conjuntos de datos estandarizados para entrenamiento, validación y prueba.

```bash
python scripts/create_splits.py --config configs/simple_config.yaml
```

### 5\. Pre-cómputo de Caché de Features

(Opcional pero muy recomendado) Pre-calcula las features multimodales para acelerar drásticamente el entrenamiento.

```bash
python scripts/precompute_cache.py --config configs/simple_config.yaml
```

### 6\. Entrenamiento

Entrena el recomendador multimodal.

```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```

Puedes habilitar el seguimiento con Weights & Biases añadiendo los flags `--use_wandb` y `--wandb_project "MiProyecto"`.

### 7\. Evaluación

Evalúa el modelo entrenado sobre el conjunto de test.

```bash
python scripts/evaluate.py --config configs/simple_config.yaml --recommender_type multimodal --eval_task retrieval
```

El script también permite evaluar baselines:

```bash
# Evaluar baseline de popularidad
python scripts/evaluate.py --config configs/simple_config.yaml --recommender_type popularity
```

**Ejemplo de Salida de Evaluación:**

| Métrica               | Valor   |
| --------------------- | ------- |
| avg\_precision\_at\_k    | 0.1234  |
| avg\_recall\_at\_k       | 0.2345  |
| avg\_f1\_at\_k           | 0.1618  |
| avg\_hit\_rate\_at\_k     | 0.6789  |
| avg\_ndcg\_at\_k         | 0.4567  |
| avg\_mrr               | 0.3890  |

### 8\. Generación de Recomendaciones

Genera una lista de recomendaciones para usuarios específicos.

```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml --users user_123 user_456
```

## Uso como Librería

Además de usar los scripts, puedes importar las clases principales del recomendador en tu propio código.

```python
import torch
from src.config import Config
from src.data.dataset import MultimodalDataset
from src.inference.recommender import Recommender
from src.models.multimodal import MultimodalRecommender as Model

# 1. Cargar configuración
config = Config.from_yaml('configs/simple_config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Cargar el modelo entrenado
# (Asegúrate de haber entrenado un modelo primero)
model = Model(
    n_users=1000, # Reemplazar con el número real de tu dataset
    n_items=5000, # Reemplazar con el número real de tu dataset
    num_numerical_features=len(config.data.numerical_features_cols),
    embedding_dim=config.model.embedding_dim,
    vision_model_name=config.model.vision_model,
    language_model_name=config.model.language_model
).to(device)

checkpoint_path = config.get_model_checkpoint_path('best_model.pth')
model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])

# 3. Cargar el dataset (necesario para los encoders y el preprocesamiento de ítems)
# Asume que los datos ya han sido preprocesados
dataset = ... # Cargar el dataset de la misma forma que en los scripts

# 4. Inicializar el recomendador de inferencia
recommender = Recommender(model, dataset, device)

# 5. Generar recomendaciones
user_id = 'un_id_de_usuario'
top_k = 10
recommendations = recommender.get_recommendations(user_id, top_k)

print(f"Recomendaciones para {user_id}: {recommendations}")
```

## Gestión Avanzada

### Gestión de Caché

El script `scripts/cache.py` te permite inspeccionar y limpiar las cachés de features.

```bash
# Listar todas las cachés de features disponibles
python scripts/cache.py list

# Ver estadísticas de una caché específica
python scripts/cache.py stats resnet_sentence-bert

# Limpiar la caché de una combinación de modelos
python scripts/cache.py clear resnet_sentence-bert
```

### Gestión de Checkpoints

El script `scripts/checkpoint_manager.py` ayuda a organizar los checkpoints guardados.

```bash
# Listar todos los checkpoints y su estado de organización
python scripts/checkpoint_manager.py list

# Organizar automáticamente los checkpoints en directorios por modelo
python scripts/checkpoint_manager.py organize
```

## Estructura del Proyecto

```
multimodal-recommender/
├── configs/              # Archivos de configuración YAML
├── data/                 # Datos crudos, procesados y splits
├── docs/                 # Documentación adicional
├── models/               # Checkpoints de modelos guardados
├── results/              # Resultados de evaluación y logs
├── scripts/              # Scripts ejecutables (train, evaluate, etc.)
├── src/                  # Código fuente del framework
│   ├── config.py         # Gestión de configuración
│   ├── data/             # Módulos de datos y preprocesamiento
│   ├── evaluation/       # Módulos de métricas y tareas de evaluación
│   ├── inference/        # Lógica para generar recomendaciones
│   ├── models/           # Arquitecturas de los modelos
│   └── training/         # Lógica de entrenamiento
├── tests/                # Pruebas unitarias e de integración
└── requirements.txt      # Dependencias de Python
```


