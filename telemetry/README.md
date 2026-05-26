# Telemetry analytics — EduSign VR

Analiza los CSVs que UTelemetryLogger (UE5) genera en
`Edusign_VR_Final/Saved/Telemetry/` y produce figuras listas para la tesis.

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

Desde la carpeta `telemetry/`:

```bash
python analyze_sessions.py
```

Por defecto busca los CSVs en
`C:/Users/estef/OneDrive/Documents/Unreal Projects/Edusign_VR_Final/Saved/Telemetry/`.

Si tu proyecto esta en otra ruta:

```bash
python analyze_sessions.py --telemetry-dir "<ruta>" --output-dir figures
```

## Salidas

Todas las figuras se guardan en `--output-dir` (por defecto `./figures/`):

| Archivo | Que muestra |
|---|---|
| `accuracy_por_personaje.png` | Porcentaje de senas correctas por personaje |
| `matriz_confusion_<personaje>.png` | Matriz expected vs predicted para cada personaje |
| `confianza_por_personaje.png` | Boxplot de confianza del modelo por personaje |
| `confianza_por_sena.png` | Violinplot de confianza por sena predicha |
| `latencia_por_personaje.png` | Latencia entre pregunta y respuesta del estudiante |
| `tiempo_por_escena.png` | Tiempo total acumulado por escena |
| `heatmap_hmd_<escena>.png` | Heatmap 2D de posicion del HMD por escena principal |
| `trayectoria_manos_<escena>.png` | Trayectoria 2D de HMD y manos |
| `flujo_navegacion.png` | Diagrama de transiciones entre escenas |
| `resumen_sesiones.csv` | Tabla con stats por sesion |

## Notas

- La primera sena de cada sesion siempre tiene `expected_label=""` y
  `latency_ms=-1` porque no hay un `LogSignAttempt` previo. El script las
  ignora automaticamente en los calculos de accuracy y latencia.
- Sesiones sin datos (solo headers) se omiten silenciosamente.
- Las escenas principales que se grafican individualmente son
  `TarakMap`, `AnubisMap`, `MagnusMap`.
