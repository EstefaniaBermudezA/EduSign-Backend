"""Microservicio REST de notas de EduSign.

Expone una API FastAPI para crear, listar y borrar las notas que los niños
guardan a partir de las respuestas del LLM dentro de la experiencia VR. Las
notas se persisten en MongoDB. Se ejecuta con un servidor ASGI, p. ej.
``uvicorn notes.app:app --reload``, y requiere la variable de entorno
``MONGO_URI`` (más las opcionales ``MONGO_DB`` y ``NOTES_COLLECTION``).
"""

import os
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "edusign")
NOTES_COLLECTION = os.getenv("NOTES_COLLECTION", "notes")

if not MONGO_URI:
    raise RuntimeError("Falta MONGO_URI en variables de entorno.")

mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = mongo_client[MONGO_DB]
notes_collection = db[NOTES_COLLECTION]

app = FastAPI(title="EduSign Notes Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteCreate(BaseModel):
    """Cuerpo de la petición para crear una nota."""

    sign: str = Field(..., description="Seña que disparó la nota (Nilo, Quien, Piramides...)")
    question: str = Field(..., description="Pregunta hecha al LLM")
    answer: str = Field(..., description="Respuesta del LLM que el niño quiere guardar")
    user_id: str = Field("default_user", description="Identificador del niño (futuro)")
    character: str = Field("Anubis", description="Personaje guía de la época")


class NoteResponse(BaseModel):
    """Representación de una nota tal como se devuelve al cliente."""

    id: str
    sign: str
    question: str
    answer: str
    user_id: str
    character: str
    created_at: str


class NotesListResponse(BaseModel):
    """Envoltura con la lista de notas devueltas por el listado."""

    notes: List[NoteResponse]

@app.get("/health")
def health():
    """Comprueba el estado del servicio y la conectividad con MongoDB."""
    try:
        mongo_client.admin.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
    return {
        "ok": True,
        "service": "notes",
        "mongo_ok": mongo_ok,
        "db": MONGO_DB,
        "collection": NOTES_COLLECTION,
    }


@app.post("/notes", response_model=NoteResponse, status_code=201)
def create_note(note: NoteCreate):
    """Guarda una nueva nota y la devuelve con su id y fecha de creación.

    Raises:
        HTTPException: 500 si falla la inserción en MongoDB.
    """
    try:
        doc = note.model_dump()
        doc["created_at"] = datetime.now(timezone.utc)
        result = notes_collection.insert_one(doc)
        return NoteResponse(
            id=str(result.inserted_id),
            sign=doc["sign"],
            question=doc["question"],
            answer=doc["answer"],
            user_id=doc["user_id"],
            character=doc["character"],
            created_at=doc["created_at"].isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando nota: {e}")


@app.get("/notes", response_model=NotesListResponse)
def list_notes(user_id: Optional[str] = None, limit: int = 50):
    """Lista las notas más recientes, opcionalmente filtradas por usuario.

    Args:
        user_id: Si se indica, solo devuelve las notas de ese niño.
        limit: Número máximo de notas a devolver (ordenadas de la más reciente).

    Raises:
        HTTPException: 500 si falla la consulta a MongoDB.
    """
    try:
        query = {}
        if user_id:
            query["user_id"] = user_id
        cursor = notes_collection.find(query).sort("created_at", -1).limit(limit)
        notes = []
        for doc in cursor:
            notes.append(
                NoteResponse(
                    id=str(doc["_id"]),
                    sign=doc.get("sign", ""),
                    question=doc.get("question", ""),
                    answer=doc.get("answer", ""),
                    user_id=doc.get("user_id", "default_user"),
                    character=doc.get("character", "Anubis"),
                    created_at=doc["created_at"].isoformat() if doc.get("created_at") else "",
                )
            )
        return NotesListResponse(notes=notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando notas: {e}")


@app.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: str):
    """Borra una nota por su id.

    Raises:
        HTTPException: 404 si la nota no existe; 500 ante otros errores.
    """
    try:
        result = notes_collection.delete_one({"_id": ObjectId(note_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Nota no encontrada")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando nota: {e}")
