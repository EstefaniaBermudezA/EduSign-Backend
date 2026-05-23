"""
Seed de estudiantes de ejemplo para la demo de tesis.

Uso:
    python -m students.seed_students            # inserta los de la lista
    python -m students.seed_students --reset    # borra todo antes de insertar

Para agregar estudiantes reales a mano, edita la lista ESTUDIANTES_DEMO
y vuelve a correr el script (los que ya existen no se duplican).
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

# Mismo .env que usa el servicio (students/.env)
load_dotenv(Path(__file__).resolve().parent / ".env")


# ─────────────────────────────────────────────────────────────────────
# EDITAR AQUÍ: lista de estudiantes que tendrán acceso a EduSign VR.
# Códigos en formato libre, se normalizan a mayúsculas sin espacios.
# Curso debe ser "6" o "7".
# ─────────────────────────────────────────────────────────────────────
ESTUDIANTES_DEMO = [
    {"codigo": "1", "nombre": "Sofia",       "apellido": "Martinez",  "curso": "6"},
    {"codigo": "2", "nombre": "Juan Camilo", "apellido": "Ruiz",      "curso": "6"},
    {"codigo": "3", "nombre": "Valentina",   "apellido": "Lopez",     "curso": "6"},
    {"codigo": "4", "nombre": "Mateo",       "apellido": "Gonzalez",  "curso": "7"},
    {"codigo": "5", "nombre": "Isabella",    "apellido": "Torres",    "curso": "7"},
    {"codigo": "6", "nombre": "Samuel",      "apellido": "Beltran",   "curso": "7"},
]


def normalizar_codigo(codigo: str) -> str:
    """Misma lógica que el servicio: numérico sin ceros a la izquierda."""
    limpio = "".join((codigo or "").split())
    if limpio.isdigit():
        return str(int(limpio))
    return limpio.upper()


def main():
    parser = argparse.ArgumentParser(description="Seed de estudiantes de EduSign")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra TODOS los estudiantes antes de insertar",
    )
    args = parser.parse_args()

    mongo_uri = os.getenv("MONGO_URI", "")
    mongo_db = os.getenv("MONGO_DB", "edusign")
    coll_name = os.getenv("STUDENTS_COLLECTION", "students")

    if not mongo_uri:
        raise SystemExit("Falta MONGO_URI en variables de entorno (.env)")

    client = MongoClient(mongo_uri, server_api=ServerApi("1"))
    coll = client[mongo_db][coll_name]
    coll.create_index([("codigo", ASCENDING)], unique=True)

    if args.reset:
        borrados = coll.delete_many({}).deleted_count
        print(f"[reset] Borrados {borrados} estudiantes previos")

    insertados = 0
    saltados = 0
    for est in ESTUDIANTES_DEMO:
        doc = {
            "codigo": normalizar_codigo(est["codigo"]),
            "nombre": est["nombre"].strip(),
            "apellido": est["apellido"].strip(),
            "curso": est["curso"].strip(),
            "created_at": datetime.now(timezone.utc),
        }
        try:
            coll.insert_one(doc)
            insertados += 1
            nombre_completo = f"{doc['nombre']} {doc['apellido']}"
            print(f"  + {doc['codigo']:<6} {nombre_completo:<30} curso {doc['curso']}")
        except DuplicateKeyError:
            saltados += 1
            print(f"  · {doc['codigo']:<6} ya existía, sin cambios")

    print(f"\nListo. Insertados: {insertados} · Saltados: {saltados}")
    print(f"Total en colección: {coll.count_documents({})}")


if __name__ == "__main__":
    main()
