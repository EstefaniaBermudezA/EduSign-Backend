# EduSign-Backend

> Componente backend del sistema EduSign: integra módulos de reconocimiento de Lengua de Señas Colombiana (LSC) y un modelo de lenguaje (LLM) para habilitar la interacción educativa en entornos de realidad virtual.

## Descripción

Este repositorio contiene los servicios backend del sistema **EduSign**, una plataforma educativa de realidad virtual orientada a la enseñanza de historia para niños sordos que cursan entre sexto y séptimo grado, mediante el uso de la Lengua de Señas Colombiana (LSC).

El backend se compone de dos módulos independientes que se integran con el cliente de realidad virtual desarrollado en Unreal Engine 5:

1. **`sign_recognition/`**: sistema de reconocimiento de señas en tiempo real basado en visión por computadora y aprendizaje profundo.
2. **`llm/`**: servicio de procesamiento de lenguaje natural basado en modelos de lenguaje (LLM) para enriquecer la interacción educativa.


## Estructura del proyecto

```
EduSign-Backend/
├── sign_recognition/              # Módulo de reconocimiento de LSC
│   ├── src/
│   │   ├── main.py                # Detección en tiempo real desde webcam
│   │   ├── sign_server.py         # Servidor WebSocket para integración con UE5
│   │   ├── train_model.py         # Entrenamiento de la CNN 1D
│   │   └── extract_features.py    # Extracción de landmarks desde videos
│   ├── models/
│   │   └── signs_cnn.pth          # Modelo CNN entrenado
│   └── requirements.txt
│
├── llm/                           # Módulo de procesamiento de lenguaje
│   ├── app.py                     # API REST (FastAPI)
│   ├── requirements.txt
│   └── .env.example               # Plantilla de variables de entorno
│
├── .gitignore
├── LICENSE
└── README.md
```

## Tecnologías

### Módulo `sign_recognition`

| Tecnología | Propósito |
|------------|-----------|
| Python 3.10+ | Lenguaje base |
| OpenCV | Captura y procesamiento de video |
| MediaPipe Tasks API | Detección de landmarks (manos y pose) |
| PyTorch | Red neuronal convolucional 1D |
| NumPy | Procesamiento numérico |

### Módulo `llm`

| Tecnología | Propósito |
|------------|-----------|
| Python 3.10+ | Lenguaje base |
| FastAPI | Framework web asíncrono |
| Uvicorn | Servidor ASGI |
| HuggingFace Inference API | Acceso al modelo de lenguaje |
| Qwen2.5-7B-Instruct | Modelo de lenguaje seleccionado |

## Requisitos previos

- Python 3.10 o superior
- Webcam (para el módulo de reconocimiento)
- Cuenta y token de acceso de HuggingFace (para el módulo LLM)
- Aproximadamente 1 GB de espacio en disco para dependencias

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/EstefaniaBermudezA/EduSign-Backend.git
cd EduSign-Backend
```

### 2. Configurar el módulo `sign_recognition`

```bash
cd sign_recognition
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

**Modelos preentrenados de MediaPipe** (no incluidos en el repositorio):

Descargar y colocar en `sign_recognition/models/`:

- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [pose_landmarker.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)

### 3. Configurar el módulo `llm`

```bash
cd ../llm
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Crear el archivo de variables de entorno a partir de la plantilla:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

Editar `.env` y completar el campo `HF_TOKEN` con un token válido de HuggingFace.

> El token se obtiene en: https://huggingface.co/settings/tokens

## Uso

### Reconocimiento de señas en tiempo real (modo standalone)

Desde la carpeta `sign_recognition/`, con el entorno virtual activo:

```bash
python src/main.py
```

Parámetros opcionales:

```bash
python src/main.py --threshold 0.0004 --camera 0
```

Controles:
- `q` o `ESC`: salir de la aplicación.

### Servidor WebSocket (integración con Unreal Engine 5)

Para conectar el reconocimiento de señas con el cliente VR, levantar el servidor WebSocket:

```bash
python src/sign_server.py
```

Parámetros opcionales:

```bash
python src/sign_server.py --host 0.0.0.0 --port 8765 --camera 0
```

El servidor expone las detecciones en `ws://127.0.0.1:8765` en formato JSON:

```json
{"label": "quien", "confidence": 0.87, "timestamp": 1712345678.9}
```

Desde Unreal Engine 5 el cliente se conecta a esta URL para recibir las señas reconocidas en tiempo real.

### Servicio LLM

Desde la carpeta `llm/`, con el entorno virtual activo:

```bash
uvicorn app:app --reload
```

El servicio quedará disponible por defecto en `http://localhost:8000`.

### Reentrenamiento del modelo de señas

Para reentrenar la CNN con nuevos datos:

```bash
cd sign_recognition

# 1. Extraer features de los videos de entrenamiento
python src/extract_features.py --videos_dir videos/MiSeña --output_dir data/features/MiSeña

# 2. Entrenar el modelo (descubre automáticamente las clases)
python src/train_model.py
```

## Datos de entrenamiento

Los videos utilizados para el entrenamiento del modelo de reconocimiento (organizados por palabras del vocabulario histórico) no se distribuyen en este repositorio debido a su tamaño y a consideraciones sobre los derechos de imagen de los participantes. El modelo entrenado (`signs_cnn.pth`) sí se incluye en el repositorio y está listo para ser utilizado en inferencia.

El vocabulario actual contempla, entre otras, señas asociadas a contenidos históricos como: Ariete, Cazar, Cueva, Feudo, Fuego, Guerra y Nilo.

## Pipeline técnico del reconocimiento

1. Captura de frames desde webcam mediante OpenCV.
2. Detección de landmarks de manos (21 puntos por mano) y pose (hombros) con MediaPipe Tasks API.
3. Normalización de coordenadas respecto al punto medio entre hombros.
4. Acumulación de la secuencia de frames en un buffer circular.
5. Detección de finalización de seña a partir del análisis de varianza de los landmarks.
6. Aplicación de normalización z-score a la secuencia activa.
7. Clasificación mediante una red neuronal convolucional 1D entrenada con augmentation.
8. Visualización del resultado en pantalla.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consultar el archivo [LICENSE](LICENSE) para más detalles.
