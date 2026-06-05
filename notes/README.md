# EduSign Backend — Módulo de notas

Microservicio FastAPI que guarda y consulta las notas que los niños generan cuando guardan respuestas del LLM durante su sesión de aprendizaje en realidad virtual.

Hace parte del backend monolítico modular de EduSign; corre de forma independiente con su propio `uvicorn`.

## Stack

| Tecnología | Propósito |
|------------|-----------|
| FastAPI | Framework web asíncrono |
| Uvicorn | Servidor ASGI |
| MongoDB Atlas | Base de datos (cloud free tier) |
| PyMongo (Stable API v1) | Cliente de MongoDB |
| python-dotenv | Carga de variables de entorno |

## Puertos del backend

| Servicio | Puerto |
| --- | --- |
| `llm` | 8000 |
| **`notes`** | **8001** |
| `students` | 8002 |

## Variables de entorno

Copiar la plantilla y completar con la cadena de conexión de Mongo Atlas:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

## Instalación

```bash
cd notes
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

```bash
uvicorn app:app --reload --port 8001
```

El servicio queda disponible en `http://localhost:8001` (puerto 8001 para no chocar con el LLM en 8000). Documentación interactiva en `http://localhost:8001/docs`.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio + Mongo |
| POST | `/notes` | Guarda una nota nueva |
| GET | `/notes` | Lista las notas (filtrable por `user_id`) |
| DELETE | `/notes/{id}` | Borra una nota por id |

### Ejemplo `POST /notes`

```json
{
  "sign": "Nilo",
  "question": "¿Por qué el río Nilo era tan importante para ustedes?",
  "answer": "El Nilo era nuestra vida...",
  "user_id": "default_user",
  "character": "Anubis"
}
```

### Ejemplo `GET /notes`

Filtrar por usuario y limitar a 20:

```
GET /notes?user_id=default_user&limit=20
```

## Esquema en MongoDB

Colección `edusign.notes`:

```json
{
  "_id": "ObjectId(...)",
  "sign": "Nilo",
  "question": "...",
  "answer": "...",
  "user_id": "default_user",
  "character": "Anubis",
  "created_at": "ISODate(2026-05-17T...)"
}
```

## Estructura

```
notes/
├── app.py            # API FastAPI (endpoints de notas)
├── requirements.txt
├── .env.example      # Plantilla de variables de entorno
└── README.md
```
