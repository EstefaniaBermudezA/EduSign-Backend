"""
EduSign Students Backend
------------------------
Servicio independiente para gestionar el padrón de estudiantes que
pueden acceder a la experiencia VR de EduSign.

Endpoints:
- GET    /health                 → status del servicio + Mongo
- GET    /students/{codigo}      → busca un estudiante por código. Si no
                                   existe, responde como "invitado" (no falla)
- POST   /students               → registra un estudiante nuevo (admin)
- GET    /students               → lista los estudiantes registrados (admin)
- DELETE /students/{codigo}      → borra un estudiante (admin)

Corre en el puerto 8002 por defecto:
  - llm       → 8000
  - notes     → 8001
  - students  → 8002
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError

# Carga el .env que vive al lado de este archivo (students/.env),
# sin importar desde qué directorio se ejecute uvicorn.
load_dotenv(Path(__file__).resolve().parent / ".env")

# =============================================
# Config
# =============================================
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "edusign")
STUDENTS_COLLECTION = os.getenv("STUDENTS_COLLECTION", "students")

if not MONGO_URI:
    raise RuntimeError("Falta MONGO_URI en variables de entorno.")

# Mongo client (Stable API v1)
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = mongo_client[MONGO_DB]
students_collection = db[STUDENTS_COLLECTION]

# Índice único sobre `codigo` para no permitir duplicados.
# create_index es idempotente: no falla si ya existe.
students_collection.create_index([("codigo", ASCENDING)], unique=True)


# =============================================
# App
# =============================================
app = FastAPI(title="EduSign Students Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================
# Helpers
# =============================================
def normalizar_codigo(codigo: str) -> str:
    """
    Normaliza el código del estudiante.

    El código son los números 1, 2, 3, ... que el profesor le asigna al niño.
    Aceptamos cualquier forma razonable que pueda escribir: " 1 ", "01", "001"
    todos resuelven al mismo registro ("1"). Si llega algo no numérico se
    devuelve tal cual (mayúsculas, sin espacios) para no romper compatibilidad.
    """
    limpio = re.sub(r"\s+", "", codigo or "")
    if limpio.isdigit():
        # Quita ceros a la izquierda: "001" -> "1", pero "0" sigue siendo "0".
        return str(int(limpio))
    return limpio.upper()


def normalizar_curso(curso: str) -> str:
    """Normaliza el curso: '6', '6°', '6 ' -> '6'."""
    if not curso:
        return ""
    return re.sub(r"[^0-9]", "", curso)


# =============================================
# Models
# =============================================
class StudentCreate(BaseModel):
    codigo: str = Field(..., description="Código único del estudiante (ej. '1', '2', '3')")
    nombre: str = Field(..., description="Primer nombre del estudiante (ej. 'Sofia')")
    apellido: str = Field(..., description="Apellido del estudiante (ej. 'Martinez')")
    curso: str = Field(..., description="Curso del estudiante: '6' o '7'")


class StudentResponse(BaseModel):
    codigo: str
    nombre: str
    apellido: str
    curso: str
    created_at: str


class StudentLookupResponse(BaseModel):
    """
    Respuesta del endpoint de login: siempre 200, con `found` indicando
    si el estudiante está registrado o si entra como invitado.
    """
    found: bool = Field(..., description="True si el código está registrado")
    codigo: str = Field(..., description="Código consultado (normalizado)")
    nombre: str = Field(..., description="Primer nombre del estudiante o 'Invitado'")
    apellido: str = Field("", description="Apellido del estudiante (vacío si invitado)")
    curso: Optional[str] = Field(None, description="Curso si está registrado, sino None")
    mensaje: str = Field(..., description="Mensaje de bienvenida listo para mostrar en VR")


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
        "service": "students",
        "mongo_ok": mongo_ok,
        "db": MONGO_DB,
        "collection": STUDENTS_COLLECTION,
    }


@app.get("/students/{codigo}", response_model=StudentLookupResponse)
def lookup_student(codigo: str):
    """
    Busca un estudiante por código.

    - Si el código existe en la base de datos, devuelve sus datos.
    - Si NO existe, responde como "Invitado" (HTTP 200, found=False).
      El cliente VR decide si lo deja entrar o no.
    """
    codigo_norm = normalizar_codigo(codigo)
    if not codigo_norm:
        raise HTTPException(status_code=400, detail="El código no puede estar vacío")

    try:
        doc = students_collection.find_one({"codigo": codigo_norm})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando estudiante: {e}")

    if doc is None:
        # No existe: entra como invitado, sin curso preasignado.
        return StudentLookupResponse(
            found=False,
            codigo=codigo_norm,
            nombre="Invitado",
            apellido="",
            curso=None,
            mensaje="Bienvenido, Invitado",
        )

    nombre = doc.get("nombre", "Invitado")
    apellido = doc.get("apellido", "")
    curso = doc.get("curso")
    return StudentLookupResponse(
        found=True,
        codigo=codigo_norm,
        nombre=nombre,
        apellido=apellido,
        curso=curso,
        mensaje=f"Bienvenido, {nombre}",
    )


@app.post("/students", response_model=StudentResponse, status_code=201)
def create_student(student: StudentCreate):
    """Registra un estudiante nuevo. Endpoint administrativo."""
    codigo_norm = normalizar_codigo(student.codigo)
    if not codigo_norm:
        raise HTTPException(status_code=400, detail="El código no puede estar vacío")

    curso_norm = normalizar_curso(student.curso)
    if curso_norm not in {"6", "7"}:
        raise HTTPException(
            status_code=400,
            detail=f"Curso inválido '{student.curso}'. Debe ser '6' o '7'.",
        )

    nombre = (student.nombre or "").strip()
    if not nombre:
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")

    apellido = (student.apellido or "").strip()
    if not apellido:
        raise HTTPException(status_code=400, detail="El apellido no puede estar vacío")

    doc = {
        "codigo": codigo_norm,
        "nombre": nombre,
        "apellido": apellido,
        "curso": curso_norm,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        students_collection.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un estudiante con código '{codigo_norm}'",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando estudiante: {e}")

    return StudentResponse(
        codigo=codigo_norm,
        nombre=nombre,
        apellido=apellido,
        curso=curso_norm,
        created_at=doc["created_at"].isoformat(),
    )


@app.get("/students", response_model=List[StudentResponse])
def list_students(curso: Optional[str] = None, limit: int = 200):
    """
    Lista los estudiantes registrados. Endpoint administrativo.
    Filtrable por curso ('6' o '7').
    """
    try:
        query = {}
        if curso:
            curso_norm = normalizar_curso(curso)
            if curso_norm:
                query["curso"] = curso_norm
        cursor = students_collection.find(query).sort("codigo", ASCENDING).limit(limit)
        estudiantes = []
        for doc in cursor:
            estudiantes.append(
                StudentResponse(
                    codigo=doc.get("codigo", ""),
                    nombre=doc.get("nombre", ""),
                    apellido=doc.get("apellido", ""),
                    curso=doc.get("curso", ""),
                    created_at=(
                        doc["created_at"].isoformat()
                        if doc.get("created_at")
                        else ""
                    ),
                )
            )
        return estudiantes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando estudiantes: {e}")


@app.delete("/students/{codigo}", status_code=204)
def delete_student(codigo: str):
    """Borra un estudiante por su código. Endpoint administrativo."""
    codigo_norm = normalizar_codigo(codigo)
    try:
        result = students_collection.delete_one({"codigo": codigo_norm})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando estudiante: {e}")
