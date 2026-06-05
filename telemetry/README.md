# EduSign Backend — Módulo de telemetría

Analiza los archivos CSV que el `UTelemetryLogger` de Unreal Engine 5 genera en `Edusign_VR_Final/Saved/Telemetry/` y produce las figuras y resúmenes usados en la tesis.

Hace parte del backend de EduSign; se ejecuta como script independiente (no es un servicio).

## Stack

| Tecnología | Propósito |
|------------|-----------|
| pandas | Carga y procesamiento de los CSV |
| NumPy | Cálculo numérico |
| Matplotlib | Generación de figuras |
| seaborn | Gráficos estadísticos |

## Instalación

```bash
cd telemetry
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

Desde la carpeta `telemetry/`:

```bash
python analyze_sessions.py
```

Por defecto busca los CSV en `C:/Users/estef/.../Edusign_VR_Final/Saved/Telemetry/`. Para usar otra ruta:

```bash
python analyze_sessions.py --telemetry-dir "<ruta>" --output-dir figures
```

## Salidas

Todas las figuras se guardan en `--output-dir` (por defecto `./figures/`):

| Archivo | Qué muestra |
|---|---|
| `accuracy_por_personaje.png` | Porcentaje de señas correctas por personaje |
| `matriz_confusion_<personaje>.png` | Matriz expected vs predicted para cada personaje |
| `confianza_por_personaje.png` | Boxplot de confianza del modelo por personaje |
| `confianza_por_sena.png` | Violinplot de confianza por seña predicha |
| `latencia_por_personaje.png` | Latencia entre pregunta y respuesta del estudiante |
| `tiempo_por_escena.png` | Tiempo total acumulado por escena |
| `heatmap_hmd_<escena>.png` | Heatmap 2D de posición del HMD por escena |
| `trayectoria_manos_<escena>.png` | Trayectoria 2D de HMD y manos |
| `flujo_navegacion.png` | Diagrama de transiciones entre escenas |
| `resumen_sesiones.csv` | Tabla con estadísticas por sesión |

## Notas

- La primera seña de cada sesión tiene `expected_label=""` y `latency_ms=-1` porque no hay un `LogSignAttempt` previo; el script las ignora en los cálculos de accuracy y latencia.
- Las sesiones sin datos (solo cabeceras) se omiten silenciosamente.
- Las escenas principales que se grafican individualmente son `TarakMap`, `AnubisMap` y `MagnusMap`.

## Estructura

```
telemetry/
├── analyze_sessions.py # Script de análisis (genera las figuras)
├── figures/            # Figuras y resúmenes generados
├── requirements.txt
└── README.md
```
