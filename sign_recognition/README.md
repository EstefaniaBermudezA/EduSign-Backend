# EduSign Backend — Módulo de reconocimiento de señas (LSC)

Sistema de reconocimiento de **Lengua de Señas Colombiana (LSC)** en tiempo real, basado en visión por computadora y aprendizaje profundo. Detecta landmarks de manos y pose con MediaPipe, los normaliza y clasifica la seña con una red neuronal convolucional 1D (CNN 1D). Se integra con el cliente de realidad virtual (Unreal Engine 5) a través de un servidor WebSocket.

## Stack

| Tecnología | Propósito |
|------------|-----------|
| Python 3.10+ | Lenguaje base |
| OpenCV | Captura y procesamiento de video |
| MediaPipe Tasks API | Detección de landmarks (manos y pose) |
| PyTorch | Red neuronal convolucional 1D |
| NumPy | Procesamiento numérico |
| websockets | Servidor de integración con UE5 |
| matplotlib | Figuras de la evaluación |

## Requisitos previos

- Python 3.10 o superior
- Webcam (para detección en tiempo real)
- **Modelos preentrenados de MediaPipe** (no incluidos en el repositorio). Descargar y colocar en `models/`:
  - [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
  - [pose_landmarker.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)

El modelo entrenado `models/signs_cnn.pth` **sí** se incluye en el repositorio y está listo para inferencia.

## Instalación

```bash
cd sign_recognition
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

### Reconocimiento en tiempo real (standalone)

```bash
python src/main.py
# Opcionales:
python src/main.py --threshold 0.0004 --camera 0
```

Controles: `q` o `ESC` para salir.

### Servidor WebSocket (integración con Unreal Engine 5)

```bash
python src/sign_server.py
# Opcionales:
python src/sign_server.py --host 0.0.0.0 --port 8765 --camera 0
```

Expone las detecciones en `ws://127.0.0.1:8765` en formato JSON:

```json
{"label": "quien", "confidence": 0.87, "timestamp": 1712345678.9}
```

El cliente UE5 se conecta a esa URL para recibir las señas reconocidas en tiempo real.

### Reentrenamiento del modelo

```bash
# 1. Extraer features (landmarks) de los videos de entrenamiento
python src/extract_features.py --videos_dir videos/MiSeña --output_dir data/features/MiSeña

# 2. Entrenar la CNN (descubre automáticamente las clases en data/features/)
python src/train_model.py
```

## Pipeline técnico

1. Captura de frames desde webcam con OpenCV.
2. Detección de landmarks de manos (21 puntos por mano) y pose (hombros) con MediaPipe Tasks API.
3. Normalización de coordenadas respecto al punto medio entre hombros.
4. Acumulación de la secuencia de frames en un buffer circular.
5. Detección de finalización de seña a partir del análisis de varianza de los landmarks.
6. Normalización z-score de la secuencia activa.
7. Clasificación con la CNN 1D entrenada con data augmentation.
8. Visualización (modo standalone) o emisión por WebSocket (modo servidor).

## Evaluación

La carpeta [`evaluation/`](evaluation/) contiene la suite de evaluación del modelo. Todos los scripts se ejecutan desde esta carpeta (`sign_recognition/`) y escriben sus salidas en `evaluation/results/`.

| Script | Qué hace |
|--------|----------|
| `kfold_evaluation.py` | Validación cruzada estratificada 5-fold × 3 semillas, sin data leakage |
| `lopo_evaluation.py` | Leave-One-Participant-Out: mide la generalización a una persona nueva |
| `ablation_study.py` | Ablación del aporte de augmentation, z-score y class weights |
| `baseline_mlp.py` | Compara la CNN 1D contra dos líneas base MLP |
| `latency_benchmark.py` | Benchmark de latencia de inferencia en CPU |
| `evaluate_model.py` | Evaluación holdout del modelo desplegado sobre un único split |

```bash
python evaluation/kfold_evaluation.py
```

## Datos de entrenamiento

Los videos de entrenamiento no se distribuyen en el repositorio por su tamaño y por consideraciones sobre los derechos de imagen de los participantes. El vocabulario actual contempla señas asociadas a contenidos históricos (Ariete, Cazar, Cueva, Feudo, Fuego, Guerra, Nilo, Pirámides, Quién).

## Estructura

```
sign_recognition/
├── src/
│   ├── main.py             # Detección en tiempo real desde webcam
│   ├── sign_server.py      # Servidor WebSocket para integración con UE5
│   ├── train_model.py      # Entrenamiento de la CNN 1D (+ utilidades de señales)
│   └── extract_features.py # Extracción de landmarks desde videos
├── evaluation/             # Suite de evaluación (ver sección Evaluación)
│   └── results/            # CSV / JSON / PNG generados por los scripts
├── models/
│   ├── signs_cnn.pth       # Modelo CNN entrenado (versionado)
│   └── *.task              # Modelos de MediaPipe (se descargan aparte)
├── data/features/          # Features .npy (regenerables; no versionados)
├── videos/                 # Videos de entrenamiento (no versionados)
├── requirements.txt
└── README.md
```
