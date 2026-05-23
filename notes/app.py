"""
EduSign Notes Backend
---------------------
Servicio independiente para guardar y consultar notas que los niños
generan al usar el módulo de aprendizaje en VR.

Endpoints:
- GET    /health          → status del servicio + Mongo
- POST   /notes           → guarda una nota nueva
- GET    /notes           → lista las notas (filtrable por user_id)
- DELETE /notes/{note_id} → borra una nota por id

Corre en puerto 8001 por defecto (el LLM corre en 8000).
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

# =============================================
# Config
# =============================================
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "edusign")
NOTES_COLLECTION = os.getenv("NOTES_COLLECTION", "notes")

if not MONGO_URI:
    raise RuntimeError("Falta MONGO_URI en variables de entorno.")

# Mongo client (Stable API v1)
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = mongo_client[MONGO_DB]
notes_collection = db[NOTES_COLLECTION]

# =============================================
# App
# =============================================
app = FastAPI(title="EduSign Notes Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================
# Models
# =============================================
class NoteCreate(BaseModel):
    sign: str = Field(..., description="Seña que disparó la nota (Nilo, Quien, Piramides...)")
    question: str = Field(..., description="Pregunta hecha al LLM")
    answer: str = Field(..., description="Respuesta del LLM que el niño quiere guardar")
    user_id: str = Field("default_user", description="Identificador del niño (futuro)")
    character: str = Field("Anubis", description="Personaje guía de la época")


class NoteResponse(BaseModel):
    id: str
    sign: str
    question: str
    answer: str
    user_id: str
    character: str
    created_at: str


class NotesListResponse(BaseModel):
    """
    Envuelve la lista en un objeto para que clientes con parsers JSON
    que solo aceptan object-root (como UE5 JsonBlueprintUtilities) puedan leerla.
    """
    notes: List[NoteResponse]


# =============================================
# Endpoints
# =============================================
@app.get("/health")
def health():
    """Health check del servicio y conectividad con Mongo."""
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
    """Guarda una nota en MongoDB."""
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
    """
    Lista las notas. Si se pasa user_id, filtra por ese usuario.
    Ordenadas de la más nueva a la más vieja.

    Devuelve un objeto {"notes": [...]} (envuelto en object) en lugar del
    array crudo, para compatibilidad con parsers JSON que requieren
    object-root (UE5 JsonBlueprintUtilities).
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
    """Borra una nota por su id."""
    try:
        result = notes_collection.delete_one({"_id": ObjectId(note_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Nota no encontrada")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando nota: {e}")
