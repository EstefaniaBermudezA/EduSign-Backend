# EduSign Students Backend

Microservicio FastAPI que gestiona el padrón de estudiantes con acceso a la experiencia VR de EduSign. Se usa en la pantalla de bienvenida del cliente UE5: el niño escribe su código y curso, UE5 consulta `GET /students/{codigo}` y muestra "Bienvenido, [Nombre]".

Si el código no existe en la base de datos, el endpoint **no falla**: responde con `found=false` y `nombre="Invitado"`, permitiendo entrar como invitado.

## Puertos del backend

| Servicio | Puerto |
| --- | --- |
| `llm` | 8000 |
| `notes` | 8001 |
| `students` | 8002 |

## Variables de entorno

Mismo `.env` que usa `notes/` (compartido a nivel de repo):

```
MONGO_URI=mongodb+srv://...
MONGO_DB=edusign
STUDENTS_COLLECTION=students
```

## Instalación

```bash
cd EduSign-Backend
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r students/requirements.txt
```

## Sembrar datos de demo

Para la primera corrida (o para resetear la lista antes de la demo de tesis):

```bash
python -m students.seed_students            # inserta los de la lista, salta los ya existentes
python -m students.seed_students --reset    # borra todo y vuelve a insertar
```

Para agregar estudiantes reales, edita la lista `ESTUDIANTES_DEMO` dentro de `seed_students.py` y vuelve a correr el script.

## Correr el servicio

Desde la raíz del repo `EduSign-Backend/`:

```bash
uvicorn students.app:app --host 0.0.0.0 --port 8002 --reload
```

Docs interactivas en http://localhost:8002/docs

## Endpoints

### `GET /health`

Health check del servicio y de la conexión a MongoDB.

```bash
curl http://localhost:8002/health
```

### `GET /students/{codigo}` — endpoint de login (lo que llama UE5)

Devuelve **siempre 200**. El campo `found` indica si el estudiante está registrado.

```bash
# Estudiante registrado
curl http://localhost:8002/students/1
# {"found":true,"codigo":"1","nombre":"Sofia Martinez","curso":"6","mensaje":"Bienvenido, Sofia Martinez"}

# Estudiante NO registrado (entra como invitado)
curl http://localhost:8002/students/ABC-999
# {"found":false,"codigo":"ABC-999","nombre":"Invitado","curso":null,"mensaje":"Bienvenido, Invitado"}
```

El código es un número entero (`1`, `2`, `3`, ...). Se normaliza quitando ceros a la izquierda y espacios, así que `1`, `01`, `001` y ` 1 ` resuelven al mismo registro.

### `POST /students` — registrar estudiante (admin)

```bash
curl -X POST http://localhost:8002/students \
  -H "Content-Type: application/json" \
  -d '{"codigo":"7","nombre":"Diana Pérez","curso":"6"}'
```

### `GET /students` — listar estudiantes (admin)

```bash
curl http://localhost:8002/students
curl "http://localhost:8002/students?curso=7"
```

### `DELETE /students/{codigo}` — borrar (admin)

```bash
curl -X DELETE http://localhost:8002/students/7
```

## Próximo paso (lado UE5)

En el widget de bienvenida (`WBP_Welcome` o similar), al pulsar **Entrar**:

1. Tomar el texto del input "Código" y construir la URL `http://<host>:8002/students/<codigo>`.
2. Hacer una petición HTTP GET (Blueprint: `HTTP Request` del plugin `HTTP Blueprint`, o C++ con `FHttpModule`).
3. Parsear el JSON, leer `mensaje` y mostrarlo en el `TextBlock` de bienvenida.
4. Si `found=false`, marcar la sesión como invitada (no guardar progreso) y dejar entrar igual.
5. Hacer `Open Level` al `M_MainMenu`.
