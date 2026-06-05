# EduSign Backend — Módulo de estudiantes

Microservicio FastAPI que gestiona el manejo de estudiantes con acceso a la experiencia VR de EduSign. Se usa en la pantalla de bienvenida del cliente UE5: el niño escribe su código, UE5 consulta `GET /students/{codigo}` y muestra "Bienvenido, [Nombre]".

Si el código no existe en la base de datos, el endpoint no falla: responde con `found=false` y `nombre="Invitado"`, permitiendo entrar como invitado.

Hace parte del backend monolítico modular de EduSign; corre de forma independiente con su propio `uvicorn`.

## Stack

| Tecnología | Propósito |
|------------|-----------|
| FastAPI | Framework web asíncrono |
| Uvicorn | Servidor ASGI |
| Pydantic | Modelos y validación |
| PyMongo | Cliente de MongoDB Atlas |
| python-dotenv | Carga de variables de entorno |

## Puertos del backend

| Servicio | Puerto |
| --- | --- |
| `llm` | 8000 |
| `notes` | 8001 |
| **`students`** | **8002** |

## Variables de entorno

Comparte el mismo `.env` que `notes/` (a nivel de repositorio):

```
MONGO_URI=mongodb+srv://...
MONGO_DB=edusign
STUDENTS_COLLECTION=students
```

## Instalación

```bash
cd students
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

Desde la raíz del repositorio `EduSign-Backend/`:

```bash
uvicorn students.app:app --host 0.0.0.0 --port 8002 --reload
```

Documentación interactiva en `http://localhost:8002/docs`.

### Sembrar datos de demo

```bash
python -m students.seed_students            # inserta los de la lista, salta los existentes
python -m students.seed_students --reset    # borra todo y vuelve a insertar
```

Para agregar estudiantes reales, edita la lista `ESTUDIANTES_DEMO` en `seed_students.py`.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio y de MongoDB |
| GET | `/students/{codigo}` | Login: resuelve un código (siempre responde 200; `found` indica si existe) |
| POST | `/students` | Registra un estudiante (admin) |
| GET | `/students` | Lista estudiantes, filtrable por `curso` (admin) |
| DELETE | `/students/{codigo}` | Borra un estudiante (admin) |

### Ejemplos

```bash
# Estudiante registrado
curl http://localhost:8002/students/1
# {"found":true,"codigo":"1","nombre":"Sofia Martinez","curso":"6","mensaje":"Bienvenido, Sofia Martinez"}

# No registrado (entra como invitado)
curl http://localhost:8002/students/ABC-999
# {"found":false,"codigo":"ABC-999","nombre":"Invitado","curso":null,"mensaje":"Bienvenido, Invitado"}
```

El código se normaliza quitando ceros a la izquierda y espacios, así que `1`, `01`, `001` y ` 1 ` resuelven al mismo registro.

## Integración con UE5 (pantalla de bienvenida)

En el widget de bienvenida, al pulsar **Entrar**:

1. Tomar el texto del input "Código" y construir la URL `http://<host>:8002/students/<codigo>`.
2. Hacer una petición HTTP GET y parsear el JSON.
3. Leer `mensaje` y mostrarlo en el `TextBlock` de bienvenida.
4. Si `found=false`, marcar la sesión como invitada (no guardar progreso) y dejar entrar igual.
5. Hacer `Open Level` al menú principal.

## Estructura

```
students/
├── app.py            # API FastAPI (login + CRUD de estudiantes)
├── seed_students.py  # Script para sembrar datos de demo
├── requirements.txt
├── .env.example      # Plantilla de variables de entorno
└── README.md
```
