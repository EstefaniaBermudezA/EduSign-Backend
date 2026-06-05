"""Microservicio REST de estudiantes de EduSign.

Expone una API FastAPI para registrar, consultar, listar y borrar estudiantes,
con el caso de uso principal de resolver un código en un mensaje de bienvenida
para la experiencia VR. Los datos se persisten en MongoDB usando ``codigo`` como
clave única. Se ejecuta con un servidor ASGI, p. ej.
``uvicorn students.app:app --reload``, y requiere ``MONGO_URI`` (más las
opcionales ``MONGO_DB`` y ``STUDENTS_COLLECTION``).
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

load_dotenv(Path(__file__).resolve().parent / ".env")

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "edusign")
STUDENTS_COLLECTION = os.getenv("STUDENTS_COLLECTION", "students")

if not MONGO_URI:
    raise RuntimeError("Falta MONGO_URI en variables de entorno.")

mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = mongo_client[MONGO_DB]
students_collection = db[STUDENTS_COLLECTION]

students_collection.create_index([("codigo", ASCENDING)], unique=True)

app = FastAPI(title="EduSign Students Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalizar_codigo(codigo: str) -> str:
    """Normaliza un código: quita espacios, elimina ceros a la izquierda en los
    numéricos ('007' -> '7') y pasa a mayúsculas los alfanuméricos."""
    limpio = re.sub(r"\s+", "", codigo or "")
    if limpio.isdigit():
        return str(int(limpio))
    return limpio.upper()


def normalizar_curso(curso: str) -> str:
    """Normaliza el curso: '6', '6°', '6 ' -> '6'."""
    if not curso:
        return ""
    return re.sub(r"[^0-9]", "", curso)

class StudentCreate(BaseModel):
    """Cuerpo de la petición para registrar un estudiante."""

    codigo: str = Field(..., description="Código único del estudiante (ej. '1', '2', '3')")
    nombre: str = Field(..., description="Primer nombre del estudiante (ej. 'Sofia')")
    apellido: str = Field(..., description="Apellido del estudiante (ej. 'Martinez')")
    curso: str = Field(..., description="Curso del estudiante: '6' o '7'")


class StudentResponse(BaseModel):
    """Representación de un estudiante tal como se devuelve al cliente."""

    codigo: str
    nombre: str
    apellido: str
    curso: str
    created_at: str


class StudentLookupResponse(BaseModel):
    """Resultado de consultar un código, con datos del estudiante o de invitado."""

    found: bool = Field(..., description="True si el código está registrado")
    codigo: str = Field(..., description="Código consultado (normalizado)")
    nombre: str = Field(..., description="Primer nombre del estudiante o 'Invitado'")
    apellido: str = Field("", description="Apellido del estudiante (vacío si invitado)")
    curso: Optional[str] = Field(None, description="Curso si está registrado, sino None")
    mensaje: str = Field(..., description="Mensaje de bienvenida listo para mostrar en VR")

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
        "service": "students",
        "mongo_ok": mongo_ok,
        "db": MONGO_DB,
        "collection": STUDENTS_COLLECTION,
    }


@app.get("/students/{codigo}", response_model=StudentLookupResponse)
def lookup_student(codigo: str):
    """Resuelve un código en un mensaje de bienvenida para la experiencia VR.

    Si el código está registrado devuelve sus datos; si no, responde como
    'Invitado' (con ``found=False``) en lugar de error, para no bloquear el flujo.

    Raises:
        HTTPException: 400 si el código queda vacío; 500 ante fallos de MongoDB.
    """
    codigo_norm = normalizar_codigo(codigo)
    if not codigo_norm:
        raise HTTPException(status_code=400, detail="El código no puede estar vacío")

    try:
        doc = students_collection.find_one({"codigo": codigo_norm})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando estudiante: {e}")

    if doc is None:
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
    """Registra un estudiante tras validar y normalizar sus datos.

    Raises:
        HTTPException: 400 si el código, nombre o apellido quedan vacíos o el
            curso no es '6' ni '7'; 409 si el código ya existe; 500 en otros
            fallos de MongoDB.
    """
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
    """Lista estudiantes ordenados por código, opcionalmente filtrados por curso.

    Args:
        curso: Si se indica, solo devuelve los estudiantes de ese curso ('6'/'7').
        limit: Número máximo de estudiantes a devolver.

    Raises:
        HTTPException: 500 si falla la consulta a MongoDB.
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
    """Borra un estudiante por su código.

    Raises:
        HTTPException: 404 si el código no existe; 500 ante otros errores.
    """
    codigo_norm = normalizar_codigo(codigo)
    try:
        result = students_collection.delete_one({"codigo": codigo_norm})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error borrando estudiante: {e}")
