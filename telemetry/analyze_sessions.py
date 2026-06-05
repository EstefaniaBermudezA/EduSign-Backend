"""Analiza los CSV de telemetria de EduSign VR y genera figuras de evaluacion.

Lee los archivos ``session_*_events|tracking|signs.csv`` exportados por la app de
Unreal, construye un resumen por sesion y produce las figuras del experimento
(accuracy, confianza y latencia del reconocimiento, matrices de confusion,
tiempos por escena, heatmaps/trayectorias del HMD y manos, y flujo de navegacion).

Se ejecuta desde la linea de comandos::

    python analyze_sessions.py --telemetry-dir <carpeta_csv> --output-dir <carpeta_figuras>

``--telemetry-dir`` apunta a la carpeta con los CSV (por defecto se autodetecta la
ruta de exportacion de Unreal); ``--output-dir`` indica donde guardar las figuras y
el ``resumen_sesiones.csv`` (se crea si no existe).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "DejaVu Sans"

PERSONAJE_COLORS = {
    "Tarak": "#A0522D",
    "Anubis": "#DAA520",
    "Magnus": "#4682B4",
}

SCENE_COLORS = {
    "M_LogIn": "#9E9E9E",
    "M_MainMenu": "#BDBDBD",
    "L_SeleccionMundos": "#E0E0E0",
    "TarakMap": PERSONAJE_COLORS["Tarak"],
    "AnubisMap": PERSONAJE_COLORS["Anubis"],
    "MagnusMap": PERSONAJE_COLORS["Magnus"],
    "NotesMap": "#616161",
}

PRINCIPAL_SCENES = ["TarakMap", "AnubisMap", "MagnusMap"]

def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Lee un CSV; retorna None si esta vacio o si hay error."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except pd.errors.EmptyDataError:
        return None
    except Exception as exc:
        print(f"  ! No se pudo leer {path.name}: {exc}", file=sys.stderr)
        return None


def load_all_sessions(telemetry_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y concatena todos los CSV de telemetria de la carpeta indicada.

    Returns:
        Tupla ``(events, tracking, signs)``; cada DataFrame queda vacio si no hay
        archivos de ese grupo o si todos estaban vacios.
    """
    if not telemetry_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {telemetry_dir}")

    def _read_group(suffix: str) -> pd.DataFrame:
        files = sorted(telemetry_dir.glob(f"session_*_{suffix}.csv"))
        if not files:
            return pd.DataFrame()
        dfs = []
        for f in files:
            df = _safe_read_csv(f)
            if df is not None:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    events = _read_group("events")
    tracking = _read_group("tracking")
    signs = _read_group("signs")

    return events, tracking, signs

def build_session_summary(events: pd.DataFrame, tracking: pd.DataFrame, signs: pd.DataFrame) -> pd.DataFrame:
    """Resume cada sesion: duracion, escenas visitadas y conteos de signs/tracking.

    Returns:
        DataFrame con una fila por ``session_id``, ordenado por dicho identificador.
    """
    rows = []
    for sid, group in events.groupby("session_id"):
        start = group[group["event_type"] == "session_start"]["timestamp_ms"].min()
        end = group[group["event_type"] == "session_end"]["timestamp_ms"].max()
        # Si faltan los eventos de inicio/fin, se usan los timestamps extremos disponibles.
        if pd.isna(start):
            start = group["timestamp_ms"].min()
        if pd.isna(end):
            end = group["timestamp_ms"].max()

        scenes_visited = group[group["event_type"] == "scene_enter"]["escena"].nunique()
        n_signs = (signs["session_id"] == sid).sum() if not signs.empty else 0
        n_tracking = (tracking["session_id"] == sid).sum() if not tracking.empty else 0
        duracion_s = (end - start) / 1000 if pd.notna(start) and pd.notna(end) else np.nan

        rows.append({
            "session_id": sid,
            "duracion_s": round(duracion_s, 1) if pd.notna(duracion_s) else None,
            "escenas_visitadas": scenes_visited,
            "n_signs": n_signs,
            "n_tracking_samples": n_tracking,
        })
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)

def plot_accuracy_por_personaje(signs: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un barras de accuracy del reconocimiento de senas por personaje."""
    if signs.empty:
        print("  [skip] accuracy_por_personaje: no hay datos de signs")
        return

    # Solo cuentan las filas con etiqueta esperada (las de practica libre no la traen).
    valid = signs[(signs["expected_label"].notna()) & (signs["expected_label"] != "")].copy()
    if valid.empty:
        print("  [skip] accuracy_por_personaje: no hay filas con expected_label")
        return

    # Comparacion case-insensitive para evitar falsos negativos por mayusculas.
    valid["correct"] = valid["expected_label"].str.lower() == valid["predicted_label"].str.lower()
    stats = valid.groupby("personaje").agg(
        total=("correct", "size"),
        correctas=("correct", "sum"),
    ).reset_index()
    stats["accuracy_pct"] = (stats["correctas"] / stats["total"]) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [PERSONAJE_COLORS.get(p, "#888888") for p in stats["personaje"]]
    bars = ax.bar(stats["personaje"], stats["accuracy_pct"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, n_total, n_ok in zip(bars, stats["total"], stats["correctas"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{n_ok}/{n_total}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Personaje")
    ax.set_title("Accuracy del reconocimiento de senas por personaje")
    fig.savefig(output_dir / "accuracy_por_personaje.png")
    plt.close(fig)
    print("  [ok] accuracy_por_personaje.png")

def plot_confusion_matrices(signs: pd.DataFrame, output_dir: Path) -> None:
    """Grafica una matriz de confusion (esperado vs predicho) por cada personaje."""
    if signs.empty:
        return

    valid = signs[(signs["expected_label"].notna()) & (signs["expected_label"] != "")].copy()
    if valid.empty:
        print("  [skip] matriz_confusion: no hay expected_label")
        return

    for personaje, df in valid.groupby("personaje"):
        # Union de etiquetas esperadas y predichas para que la matriz sea cuadrada.
        labels = sorted(set(df["expected_label"]).union(set(df["predicted_label"])))
        if not labels:
            continue
        mat = pd.crosstab(df["expected_label"], df["predicted_label"]).reindex(
            index=labels, columns=labels, fill_value=0
        )
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.7)))
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Esperado")
        ax.set_title(f"Matriz de confusion - {personaje}")
        fig.savefig(output_dir / f"matriz_confusion_{personaje}.png")
        plt.close(fig)
        print(f"  [ok] matriz_confusion_{personaje}.png")

def plot_confianza_por_personaje(signs: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un boxplot de la confianza del modelo por personaje."""
    if signs.empty:
        return
    df = signs[signs["confidence"].notna()].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = [p for p in ["Tarak", "Anubis", "Magnus"] if p in df["personaje"].unique()]
    palette = {p: PERSONAJE_COLORS.get(p, "#888888") for p in order}
    sns.boxplot(data=df, x="personaje", y="confidence", order=order,
                hue="personaje", hue_order=order, palette=palette, legend=False,
                ax=ax, fliersize=3, width=0.5)
    sns.stripplot(data=df, x="personaje", y="confidence", order=order,
                  color="black", alpha=0.35, size=3, jitter=0.2, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confianza del modelo")
    ax.set_xlabel("Personaje")
    ax.set_title("Distribucion de confianza del reconocimiento")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    fig.savefig(output_dir / "confianza_por_personaje.png")
    plt.close(fig)
    print("  [ok] confianza_por_personaje.png")


def plot_confianza_por_sena(signs: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un violinplot de la confianza para las senas predichas mas frecuentes."""
    if signs.empty:
        return
    df = signs[signs["confidence"].notna() & signs["predicted_label"].notna()].copy()
    if df.empty:
        return

    # Se limita a las 12 senas mas frecuentes para mantener el eje X legible.
    top_senas = df["predicted_label"].value_counts().head(12).index.tolist()
    df = df[df["predicted_label"].isin(top_senas)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x="predicted_label", y="confidence",
                   inner="quartile", linewidth=1, ax=ax,
                   order=sorted(df["predicted_label"].unique()))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confianza")
    ax.set_xlabel("Sena predicha")
    ax.set_title("Distribucion de confianza por sena")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(output_dir / "confianza_por_sena.png")
    plt.close(fig)
    print("  [ok] confianza_por_sena.png")

def plot_latencia_por_personaje(signs: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un boxplot de la latencia (pregunta -> sena del estudiante) por personaje."""
    if signs.empty:
        return
    # Latencias <= 0 son invalidas (sena sin pregunta previa) y se descartan.
    df = signs[(signs["latency_ms"].notna()) & (signs["latency_ms"] > 0)].copy()
    if df.empty:
        print("  [skip] latencia: no hay filas con latencia > 0")
        return
    df["latency_s"] = df["latency_ms"] / 1000

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = [p for p in ["Tarak", "Anubis", "Magnus"] if p in df["personaje"].unique()]
    palette = {p: PERSONAJE_COLORS.get(p, "#888888") for p in order}
    sns.boxplot(data=df, x="personaje", y="latency_s", order=order,
                hue="personaje", hue_order=order, palette=palette, legend=False,
                ax=ax, fliersize=3, width=0.5)
    sns.stripplot(data=df, x="personaje", y="latency_s", order=order,
                  color="black", alpha=0.35, size=3, jitter=0.2, ax=ax)
    ax.set_ylabel("Latencia de respuesta (s)")
    ax.set_xlabel("Personaje")
    ax.set_title("Latencia entre pregunta y sena del estudiante")
    fig.savefig(output_dir / "latencia_por_personaje.png")
    plt.close(fig)
    print("  [ok] latencia_por_personaje.png")

def _scene_durations(events: pd.DataFrame) -> pd.DataFrame:
    """Calcula la permanencia en cada escena por sesion.

    La duracion de una escena es el tiempo hasta el siguiente ``scene_enter`` (o el
    ``session_end``), por eso ambos tipos de evento se intercalan ordenados.

    Returns:
        DataFrame con columnas ``session_id``, ``escena``, ``personaje``, ``epoca`` y
        ``duracion_s`` (una fila por tramo de permanencia).
    """
    rows = []
    for sid, g in events.sort_values("timestamp_ms").groupby("session_id"):
        g = g.reset_index(drop=True)
        scene_rows = g[g["event_type"].isin(["scene_enter", "session_end"])].reset_index(drop=True)
        for i in range(len(scene_rows) - 1):
            actual = scene_rows.iloc[i]
            siguiente = scene_rows.iloc[i + 1]
            # Solo los scene_enter abren un tramo; session_end solo cierra el anterior.
            if actual["event_type"] == "scene_enter":
                dur = (siguiente["timestamp_ms"] - actual["timestamp_ms"]) / 1000
                rows.append({
                    "session_id": sid,
                    "escena": actual["escena"],
                    "personaje": actual.get("personaje", ""),
                    "epoca": actual.get("epoca", ""),
                    "duracion_s": dur,
                })
    return pd.DataFrame(rows)


def plot_tiempo_por_escena(events: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un barras horizontales del tiempo total acumulado en cada escena."""
    if events.empty:
        return
    durations = _scene_durations(events)
    if durations.empty:
        print("  [skip] tiempo_por_escena: no hay duraciones calculables")
        return
    agg = durations.groupby("escena")["duracion_s"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [SCENE_COLORS.get(e, "#888888") for e in agg.index]
    ax.barh(agg.index, agg.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, (escena, secs) in enumerate(agg.items()):
        ax.text(secs + max(agg.values) * 0.01, i, f"{secs:.1f}s", va="center", fontsize=10)
    ax.set_xlabel("Tiempo total acumulado (s)")
    ax.set_title("Tiempo total por escena (todas las sesiones)")
    fig.savefig(output_dir / "tiempo_por_escena.png")
    plt.close(fig)
    print("  [ok] tiempo_por_escena.png")

def plot_hmd_heatmaps(tracking: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un heatmap 2D de la posicion del HMD por cada escena principal."""
    if tracking.empty:
        return
    for escena in PRINCIPAL_SCENES:
        df = tracking[tracking["escena"] == escena]
        # Se omiten escenas con muy pocas muestras: el histograma no seria representativo.
        if df.empty or len(df) < 20:
            continue

        x = df["hmd_x"].values
        y = df["hmd_y"].values

        fig, ax = plt.subplots(figsize=(6, 5.5))
        h = ax.hist2d(x, y, bins=60, cmap="hot", cmin=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title(f"Heatmap de posicion HMD - {escena}  (n={len(df)})")
        fig.colorbar(h[3], ax=ax, label="frecuencia")
        fig.savefig(output_dir / f"heatmap_hmd_{escena}.png")
        plt.close(fig)
        print(f"  [ok] heatmap_hmd_{escena}.png")

def plot_trayectoria_manos(tracking: pd.DataFrame, output_dir: Path) -> None:
    """Grafica la trayectoria XY del HMD y de ambas manos por cada escena principal."""
    if tracking.empty:
        return
    for escena in PRINCIPAL_SCENES:
        # Se ordena por tiempo para que la linea siga el recorrido real, no el orden de filas.
        df = tracking[tracking["escena"] == escena].sort_values("timestamp_ms")
        if df.empty or len(df) < 20:
            continue

        fig, ax = plt.subplots(figsize=(6, 5.5))

        ax.plot(df["hmd_x"], df["hmd_y"], color="#888888", alpha=0.4, linewidth=0.8, label="HMD")

        ax.plot(df["lhand_x"], df["lhand_y"], color="#1f77b4", alpha=0.6, linewidth=0.7, label="Mano izquierda")
        ax.plot(df["rhand_x"], df["rhand_y"], color="#d62728", alpha=0.6, linewidth=0.7, label="Mano derecha")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title(f"Trayectoria HMD y manos - {escena}")
        ax.legend(loc="best", fontsize=9)
        fig.savefig(output_dir / f"trayectoria_manos_{escena}.png")
        plt.close(fig)
        print(f"  [ok] trayectoria_manos_{escena}.png")

def plot_flujo_navegacion(events: pd.DataFrame, output_dir: Path) -> None:
    """Grafica un diagrama de flujo con las transiciones entre escenas y su frecuencia."""
    if events.empty:
        return
    df = events[events["event_type"] == "scene_enter"].sort_values(["session_id", "timestamp_ms"]).copy()
    if df.empty:
        return

    # Cada transicion es (escena actual -> siguiente escena de la misma sesion).
    df["next_escena"] = df.groupby("session_id")["escena"].shift(-1)
    transitions = df.dropna(subset=["next_escena"])
    if transitions.empty:
        return
    pair_counts = transitions.groupby(["escena", "next_escena"]).size().reset_index(name="count")

    # Posicion fija (columna, fila) de cada escena conocida para un diagrama estable.
    layout_cols = {
        "M_LogIn": 0,
        "M_MainMenu": 1,
        "L_SeleccionMundos": 2,
        "TarakMap": 3,
        "AnubisMap": 3,
        "MagnusMap": 3,
        "NotesMap": 3,
    }
    layout_rows = {
        "M_LogIn": 2,
        "M_MainMenu": 2,
        "L_SeleccionMundos": 2,
        "TarakMap": 0,
        "AnubisMap": 1,
        "MagnusMap": 2,
        "NotesMap": 3,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    all_scenes = sorted(set(df["escena"].unique()) | set(transitions["next_escena"].unique()))
    positions = {}
    for s in all_scenes:
        if s in layout_cols:
            positions[s] = (layout_cols[s], layout_rows[s])
        else:
            positions[s] = (4, len(positions))

    for s, (x, y) in positions.items():
        color = SCENE_COLORS.get(s, "#888888")
        ax.scatter([x], [y], s=1200, color=color, edgecolor="black", linewidth=1, zorder=3)
        ax.text(x, y, s.replace("Map", "").replace("M_", "").replace("L_", ""),
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    max_count = pair_counts["count"].max()
    for _, row in pair_counts.iterrows():
        src = positions.get(row["escena"])
        dst = positions.get(row["next_escena"])
        if src is None or dst is None or src == dst:
            continue
        # Grosor y tamano de flecha proporcionales a la frecuencia de la transicion.
        width = 0.5 + 3.5 * (row["count"] / max_count)
        arrow = FancyArrowPatch(src, dst, arrowstyle="->",
                                mutation_scale=15 + 15 * (row["count"] / max_count),
                                linewidth=width, color="#444444", alpha=0.6,
                                connectionstyle="arc3,rad=0.15", zorder=2)
        ax.add_patch(arrow)
        midx, midy = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
        ax.text(midx, midy + 0.1, str(row["count"]), ha="center", fontsize=8,
                color="#222222", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1, max(layout_rows.values()) + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Flujo de navegacion entre escenas (numero de transiciones)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output_dir / "flujo_navegacion.png")
    plt.close(fig)
    print("  [ok] flujo_navegacion.png")

def default_telemetry_dir() -> Path:
    """Ruta default: ../../Unreal Projects/Edusign_VR_Final/Saved/Telemetry/ relativo al script."""
    script = Path(__file__).resolve().parent
    candidates = [
        script.parents[2] / "Documents" / "Unreal Projects" / "Edusign_VR_Final" / "Saved" / "Telemetry",
        Path.home() / "Documents" / "Unreal Projects" / "Edusign_VR_Final" / "Saved" / "Telemetry",
        Path("C:/Users/estef/OneDrive/Documents/Unreal Projects/Edusign_VR_Final/Saved/Telemetry"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def main() -> int:
    """Punto de entrada CLI: carga la telemetria, escribe el resumen y genera las figuras.

    Returns:
        Codigo de salida ``0`` si todo corrio bien, ``1`` si no se encontraron datos.
    """
    parser = argparse.ArgumentParser(description="Analiza CSVs de telemetria EduSign VR.")
    parser.add_argument("--telemetry-dir", type=Path, default=default_telemetry_dir(),
                        help="Carpeta con los session_*_events|tracking|signs.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures",
                        help="Carpeta donde guardar las figuras (se crea si no existe)")
    args = parser.parse_args()

    print(f"Telemetry dir: {args.telemetry_dir}")
    print(f"Output dir:    {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events, tracking, signs = load_all_sessions(args.telemetry_dir)
    if events.empty and tracking.empty and signs.empty:
        print("No se encontraron datos. Verifica --telemetry-dir.")
        return 1

    summary = build_session_summary(events, tracking, signs)
    summary.to_csv(args.output_dir / "resumen_sesiones.csv", index=False)
    print("\n=== Resumen general ===")
    print(f"Sesiones encontradas:  {events['session_id'].nunique() if not events.empty else 0}")
    print(f"Eventos totales:       {len(events)}")
    print(f"Snapshots tracking:    {len(tracking)}")
    print(f"Senas reconocidas:     {len(signs)}")
    if not summary.empty:
        print(f"Duracion promedio:     {summary['duracion_s'].dropna().mean():.1f} s")
        print(f"Duracion total:        {summary['duracion_s'].dropna().sum():.1f} s")

    print("\n=== Generando figuras ===")
    plot_accuracy_por_personaje(signs, args.output_dir)
    plot_confusion_matrices(signs, args.output_dir)
    plot_confianza_por_personaje(signs, args.output_dir)
    plot_confianza_por_sena(signs, args.output_dir)
    plot_latencia_por_personaje(signs, args.output_dir)
    plot_tiempo_por_escena(events, args.output_dir)
    plot_hmd_heatmaps(tracking, args.output_dir)
    plot_trayectoria_manos(tracking, args.output_dir)
    plot_flujo_navegacion(events, args.output_dir)

    print(f"\nListo. Figuras y resumen en: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
