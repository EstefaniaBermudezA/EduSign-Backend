# EduSign Backend — Módulo LLM

Microservicio FastAPI que expone un modelo de lenguaje (LLM) como narrador o personaje histórico para EduSign. Recibe la pregunta de un niño y devuelve una respuesta corta, en español y con vocabulario simple, adecuada para estudiantes sordos en la experiencia de realidad virtual.

Hace parte del backend monolítico modular de EduSign; corre de forma independiente con su propio `uvicorn`.

## Stack

| Tecnología | Propósito |
|------------|-----------|
| FastAPI | Framework web asíncrono |
| Uvicorn | Servidor ASGI |
| Hugging Face Inference Router | Acceso al modelo |
| Qwen2.5-7B-Instruct | Modelo de lenguaje seleccionado |
| requests | Cliente HTTP hacia el router de HF |
| python-dotenv | Carga de variables de entorno |

## Puertos del backend

| Servicio | Puerto |
| --- | --- |
| **`llm`** | **8000** |
| `notes` | 8001 |
| `students` | 8002 |

## Variables de entorno

Copiar la plantilla y completar el token:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `HF_TOKEN` | Sí | Token de acceso de Hugging Face. Se obtiene en https://huggingface.co/settings/tokens |

Si `HF_TOKEN` no está definido, el servicio no arranca (falla de forma explícita).

## Instalación

```bash
cd llm
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

```bash
uvicorn app:app --reload
```

El servicio queda disponible en `http://localhost:8000`. La documentación interactiva (OpenAPI/Swagger) se genera automáticamente en `http://localhost:8000/docs`.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio y modelo configurado |
| POST | `/ask` | Genera la respuesta del personaje/narrador a una pregunta |

### `POST /ask`

**Petición:**

```json
{
  "prompt": "¿Por qué era importante el río Nilo?",
  "character": "Anubis",
  "max_tokens": 150,
  "temperature": 0.3
}
```

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `prompt` | string | — | Pregunta del niño (obligatorio) |
| `character` | string \| null | `null` | Personaje histórico que responde. Si se omite, responde un narrador neutro |
| `max_tokens` | int (1–512) | 150 | Longitud máxima de la respuesta |
| `temperature` | float (0–2) | 0.3 | Aleatoriedad de la generación |

**Respuesta:**

```json
{
  "answer": "El Nilo nos daba agua y comida. Sus crecidas dejaban tierra buena para sembrar.",
  "latency_ms": 842
}
```

**Errores:** `504` si el proveedor excede el timeout (60 s), el código del proveedor ante un error de la API, o `500` ante cualquier otro fallo.

## Evaluación

La carpeta [`evaluation/`](evaluation/) contiene el banco de pruebas del LLM: un *gold standard* de preguntas y respuestas esperadas (`gold_standard.json`) y el script `run_eval.py`, que ejecuta el modelo contra ese conjunto y guarda los resultados en `evaluation/results/`.

```bash
python evaluation/run_eval.py
```

## Estructura

```
llm/
├── app.py                 # API FastAPI (endpoints /health y /ask)
├── evaluation/
│   ├── gold_standard.json # Preguntas/respuestas de referencia
│   ├── run_eval.py        # Ejecuta la evaluación
│   └── results/           # Salidas de la evaluación
├── requirements.txt
├── .env.example           # Plantilla de variables de entorno
└── README.md
```
