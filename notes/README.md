# EduSign Backend — Notes module

Módulo del repo `EduSign-Backend` que guarda y consulta las notas que los niños generan cuando guardan respuestas del LLM en su sesión de aprendizaje en VR.

Hace parte del backend monolítico modular:
- `llm/` — proxy al LLM (puerto 8000)
- `sign_recognition/` — reconocimiento de señas
- `notes/` — este módulo (puerto 8001)

Cada submódulo corre independientemente con su propio `uvicorn`.

## Stack

- FastAPI
- MongoDB Atlas (cloud free tier)
- PyMongo con Stable API v1

## Setup

1. Crear y activar virtualenv:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Instalar dependencias:

   ```
   pip install -r requirements.txt
   ```

3. Crear archivo `.env` copiando `.env.example` y completando con la password real de Mongo Atlas:

   ```
   copy .env.example .env
   ```

4. Correr el servicio (puerto 8001 para no chocar con el LLM en 8000):

   ```
   uvicorn app:app --reload --port 8001
   ```

## Endpoints

| Método | Ruta              | Descripción |
|--------|-------------------|-------------|
| GET    | `/health`         | Status del servicio + Mongo |
| POST   | `/notes`          | Guarda una nota nueva |
| GET    | `/notes`          | Lista todas las notas (filtrable por `user_id`) |
| DELETE | `/notes/{id}`     | Borra una nota por id |

### Ejemplo POST /notes

```json
{
  "sign": "Nilo",
  "question": "¿Por qué el río Nilo era tan importante para ustedes?",
  "answer": "El Nilo era nuestra vida...",
  "user_id": "default_user",
  "character": "Anubis"
}
```

### Ejemplo GET /notes

Filtrar por usuario y limitar a 20:

```
GET /notes?user_id=default_user&limit=20
```

## Esquema en MongoDB

Colección: `edusign.notes`

```json
{
  "_id": ObjectId("..."),
  "sign": "Nilo",
  "question": "...",
  "answer": "...",
  "user_id": "default_user",
  "character": "Anubis",
  "created_at": ISODate("2026-05-17T...")
}
```
