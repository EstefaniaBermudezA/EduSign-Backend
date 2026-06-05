"""
Benchmark de latencia del clasificador CNN 1D de senas LSC en CPU.

Mide tres aspectos relevantes para el RNF de tiempo de respuesta del Anexo B
y para la afirmacion de "modelo ligero" del Anexo C:

  1. Latencia de inferencia por muestra individual (batch=1).
     Reporta p50, p90, p95, p99, media, max sobre N_RUNS corridas
     descartando un warmup inicial.

  2. Throughput por lote (batch=32 y batch=128).
     Mide muestras por segundo.

  3. Pipeline lite end-to-end (sin captura de camara): normalizacion z-score
     + permutacion + forward + softmax + argmax. Reporta p50, p95, p99.

Adicionalmente reporta:
  - Numero de parametros del modelo.
  - Tamano del archivo .pth serializado en disco.
  - Informacion del entorno (CPU, threads de torch).

Salidas (en evaluation/results/):
  - latency_benchmark.json    -> todas las metricas
  - latency_histogram.png     -> histograma de latencia por muestra (batch=1)

Uso (desde sign_recognition_eval/):
    python evaluation/latency_benchmark.py
"""

import os
import sys
import json
import time
import platform

import numpy as np
import torch
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from train_model import SignCNN, SEQ_LENGTH, NUM_FEATURES  # noqa: E402

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "signs_cnn.pth")
WARMUP_RUNS = 50
N_RUNS = 500
BATCH_SIZES = [1, 32, 128]


# ----------------------------------------------------------------------------
# Carga del modelo
# ----------------------------------------------------------------------------

def load_model(model_path):
    """Carga el modelo serializado por train_model.py con sus stats z-score."""
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = payload["num_classes"]
    model = SignCNN(num_classes=num_classes)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    feat_mean = torch.tensor(payload["feat_mean"], dtype=torch.float32)
    feat_std = torch.tensor(payload["feat_std"], dtype=torch.float32)
    return model, feat_mean, feat_std, num_classes, payload.get("idx_to_label", {})


# ----------------------------------------------------------------------------
# Mediciones
# ----------------------------------------------------------------------------

def time_perf_ms():
    return time.perf_counter() * 1000.0


def measure_inference_latency(model, batch_size, n_runs, warmup):
    """Latencia pura de forward sobre un tensor sintetico de tamano batch."""
    x = torch.randn(batch_size, NUM_FEATURES, SEQ_LENGTH, dtype=torch.float32)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    # Medicion
    samples_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time_perf_ms()
            _ = model(x)
            t1 = time_perf_ms()
            samples_ms.append(t1 - t0)
    return np.array(samples_ms)


def measure_pipeline_latency(model, feat_mean, feat_std, n_runs, warmup):
    """Pipeline 'lite' end-to-end (sin MediaPipe): normalizacion + forward + softmax + argmax."""
    samples_ms = []
    raw = np.random.randn(SEQ_LENGTH, NUM_FEATURES).astype(np.float32)

    def step(raw_np):
        t0 = time_perf_ms()
        x = torch.from_numpy(raw_np)
        x = (x - feat_mean) / feat_std
        x = x.permute(1, 0).unsqueeze(0)  # (1, F, T)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            _ = probs.argmax(dim=1).item()
        t1 = time_perf_ms()
        return t1 - t0

    for _ in range(warmup):
        step(raw)
    for _ in range(n_runs):
        samples_ms.append(step(raw))
    return np.array(samples_ms)


def summarize(samples_ms):
    return {
        "mean": float(np.mean(samples_ms)),
        "std": float(np.std(samples_ms, ddof=1)),
        "p50": float(np.percentile(samples_ms, 50)),
        "p90": float(np.percentile(samples_ms, 90)),
        "p95": float(np.percentile(samples_ms, 95)),
        "p99": float(np.percentile(samples_ms, 99)),
        "min": float(np.min(samples_ms)),
        "max": float(np.max(samples_ms)),
        "n_runs": int(len(samples_ms)),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Cargando modelo: {MODEL_PATH}")
    model, feat_mean, feat_std, num_classes, idx_to_label = load_model(MODEL_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = os.path.getsize(MODEL_PATH)
    print(f"  num_classes      : {num_classes}")
    print(f"  parametros       : {n_params:,}")
    print(f"  archivo (.pth)   : {model_size_bytes / 1024:.1f} KB")

    print(f"\nEntorno:")
    print(f"  Python           : {sys.version.split()[0]}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  Plataforma       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  CPU              : {platform.processor() or 'desconocido'}")
    print(f"  Threads de torch : {torch.get_num_threads()}")

    results = {
        "model": {
            "path": MODEL_PATH,
            "num_classes": num_classes,
            "num_parameters": int(n_params),
            "size_kb": round(model_size_bytes / 1024, 1),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "processor": platform.processor() or "desconocido",
            "torch_threads": torch.get_num_threads(),
        },
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "n_runs": N_RUNS,
            "batch_sizes": BATCH_SIZES,
            "seq_length": SEQ_LENGTH,
            "num_features": NUM_FEATURES,
        },
        "inference_only_ms": {},
        "throughput_samples_per_s": {},
        "pipeline_end_to_end_ms": {},
    }

    # --- 1. Latencia pura de inferencia por batch ---
    print(f"\nLatencia de inferencia ({N_RUNS} runs por batch, warmup={WARMUP_RUNS}):")
    samples_b1 = None
    for bs in BATCH_SIZES:
        samples = measure_inference_latency(model, bs, N_RUNS, WARMUP_RUNS)
        stats = summarize(samples)
        results["inference_only_ms"][f"batch_{bs}"] = stats
        results["throughput_samples_per_s"][f"batch_{bs}"] = {
            "mean": bs / (stats["mean"] / 1000.0),
            "p50": bs / (stats["p50"] / 1000.0),
        }
        print(f"  batch={bs:>3}: mean={stats['mean']:6.2f} ms  p50={stats['p50']:6.2f}  "
              f"p95={stats['p95']:6.2f}  p99={stats['p99']:6.2f}  "
              f"throughput~{bs / (stats['mean']/1000.0):.0f} samples/s")
        if bs == 1:
            samples_b1 = samples

    # --- 2. Pipeline end-to-end (norm + forward + softmax + argmax) ---
    print(f"\nPipeline end-to-end (batch=1) sobre {N_RUNS} runs:")
    pipeline_samples = measure_pipeline_latency(model, feat_mean, feat_std, N_RUNS, WARMUP_RUNS)
    pipeline_stats = summarize(pipeline_samples)
    results["pipeline_end_to_end_ms"] = pipeline_stats
    print(f"  mean={pipeline_stats['mean']:6.2f} ms  p50={pipeline_stats['p50']:6.2f}  "
          f"p95={pipeline_stats['p95']:6.2f}  p99={pipeline_stats['p99']:6.2f}")

    # --- 3. Cumplimiento del RNF (< 2000 ms desde fin de gesto) ---
    rnf_threshold_ms = 2000.0
    p99 = pipeline_stats["p99"]
    results["rnf_compliance"] = {
        "threshold_ms": rnf_threshold_ms,
        "p99_pipeline_ms": p99,
        "ok": bool(p99 < rnf_threshold_ms),
        "headroom_ms": rnf_threshold_ms - p99,
    }
    print(f"\nRNF Anexo B (<{rnf_threshold_ms:.0f} ms): "
          f"p99 pipeline = {p99:.2f} ms -> {'CUMPLE' if p99 < rnf_threshold_ms else 'NO CUMPLE'} "
          f"(margen {rnf_threshold_ms - p99:.1f} ms)")

    # --- Guardar JSON ---
    json_path = os.path.join(out_dir, "latency_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    # --- Histograma ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(samples_b1, bins=40, color="#3a6ea5", edgecolor="white")
    ax.axvline(np.percentile(samples_b1, 50), color="black", linestyle="--",
               label=f"p50 = {np.percentile(samples_b1, 50):.2f} ms")
    ax.axvline(np.percentile(samples_b1, 95), color="orange", linestyle="--",
               label=f"p95 = {np.percentile(samples_b1, 95):.2f} ms")
    ax.axvline(np.percentile(samples_b1, 99), color="red", linestyle="--",
               label=f"p99 = {np.percentile(samples_b1, 99):.2f} ms")
    ax.set_xlabel("Latencia (ms)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Latencia de inferencia (batch=1, {N_RUNS} runs en CPU)")
    ax.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, "latency_histogram.png")
    plt.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"PNG : {png_path}")


if __name__ == "__main__":
    main()
