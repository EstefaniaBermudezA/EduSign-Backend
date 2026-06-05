"""
Lineas base MLP para comparar contra la CNN 1D del clasificador de senas LSC.

Compara dos arquitecturas alternativas, ambas evaluadas con el mismo protocolo
5-fold x 3 seeds sin leakage definido en kfold_evaluation.py:

  - MLP_flat   : aplana la secuencia (60 x 132 = 7920) y la pasa por
                 capas densas. Tamano calibrado para ~176k parametros,
                 comparable con la CNN 1D (~178k).
  - MLP_pooled : agrega temporalmente la secuencia (mean pooling sobre los 60
                 cuadros) y clasifica el vector de pose promedio (~26k params).
                 Sirve para evaluar cuanto aporta la informacion temporal.

Para cada modelo se reporta accuracy, F1 macro (media +/- std combinando 15
entrenamientos), numero de parametros y latencia de inferencia (batch=1).
Adicionalmente, se comparan los resultados contra la CNN 1D leyendo
kfold_summary.json del experimento anterior.

Salidas (en evaluation/results/):
  - baseline_per_fold.csv     -> una fila por modelo x seed x fold
  - baseline_comparison.csv   -> tabla CNN 1D vs MLP_flat vs MLP_pooled
  - baseline_comparison.json  -> mismo contenido con metadatos

Uso (desde sign_recognition_eval/):
    python evaluation/baseline_mlp.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, SRC_DIR)

from kfold_evaluation import (  # noqa: E402
    set_seed,
    load_raw_dataset,
    stratified_kfold_indices,
    prepare_fold_data,
    per_class_counts,
    macro_average,
)
from train_model import (  # noqa: E402
    SEQ_LENGTH,
    NUM_FEATURES,
    BATCH_SIZE,
    LEARNING_RATE,
)

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
N_SPLITS = 5
SEEDS = [42, 123, 2025]
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 50
SCHEDULER_PATIENCE = 25
SCHEDULER_FACTOR = 0.5
WEIGHT_DECAY = 1e-4

# Latency benchmark inline
LAT_WARMUP = 50
LAT_RUNS = 200


# ----------------------------------------------------------------------------
# Arquitecturas MLP
# ----------------------------------------------------------------------------

class SignMLPFlat(nn.Module):
    """MLP sobre el vector aplanado (60 x 132 = 7920).

    Calibrado para ~176k parametros, comparable con la CNN 1D (~178k).
    """

    def __init__(self, num_features=NUM_FEATURES, seq_length=SEQ_LENGTH, num_classes=9):
        super().__init__()
        input_dim = num_features * seq_length  # 7920
        hidden1 = 22  # 7920 * 22 ~ 174k params
        hidden2 = 64
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        # Entrada: (batch, NUM_FEATURES, SEQ_LENGTH) -> aplana ambos ejes
        return self.net(x)


class SignMLPPooled(nn.Module):
    """MLP sobre el vector pose promedio (mean-pool sobre el eje temporal).

    Pierde la informacion temporal: clasifica solo la "pose promedio".
    Sirve para medir cuanto aporta la dinamica de la sena al rendimiento.
    """

    def __init__(self, num_features=NUM_FEATURES, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Entrada: (batch, NUM_FEATURES, SEQ_LENGTH) -> pool sobre tiempo
        x = x.mean(dim=2)  # (batch, NUM_FEATURES)
        return self.net(x)


MODELS = {
    "MLP_flat": SignMLPFlat,
    "MLP_pooled": SignMLPPooled,
}


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------

def train_one_fold(model_ctor, X_train, y_train, X_val, y_val, num_classes, seed):
    set_seed(seed)

    X_train_t = torch.tensor(X_train).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val).permute(0, 2, 1)
    y_val_t = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    weights_tensor = torch.tensor(class_weights)

    model = model_ctor(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1).numpy()

    return val_preds, best_epoch, best_val_loss


# ----------------------------------------------------------------------------
# Benchmark inline de latencia (batch=1)
# ----------------------------------------------------------------------------

def quick_latency(model_ctor, num_classes):
    model = model_ctor(num_classes=num_classes)
    model.eval()
    x = torch.randn(1, NUM_FEATURES, SEQ_LENGTH, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(LAT_WARMUP):
            _ = model(x)
        samples = []
        for _ in range(LAT_RUNS):
            t0 = time.perf_counter()
            _ = model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(samples)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def count_params(model_ctor, num_classes):
    m = model_ctor(num_classes=num_classes)
    return int(sum(p.numel() for p in m.parameters()))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_model(name, ctor, raw_X, raw_y, num_classes):
    print(f"\n========== {name} ==========")
    print(f"  parametros: {count_params(ctor, num_classes):,}")
    accuracies, f1s, rows = [], [], []
    t0 = time.time()
    for seed in SEEDS:
        for fold_idx, (train_idx, val_idx) in enumerate(
            stratified_kfold_indices(raw_y, N_SPLITS, seed)
        ):
            X_tr, y_tr, X_va, y_va = prepare_fold_data(raw_X, raw_y, train_idx, val_idx)
            val_preds, best_epoch, best_val_loss = train_one_fold(
                ctor, X_tr, y_tr, X_va, y_va, num_classes, seed
            )
            acc = float((val_preds == y_va).mean()) * 100
            pcs = per_class_counts(y_va, val_preds, num_classes)
            _, _, mf1 = macro_average(pcs)
            accuracies.append(acc)
            f1s.append(mf1)
            rows.append({
                "model": name, "seed": seed, "fold": fold_idx + 1,
                "accuracy_pct": acc, "f1_macro_pct": mf1,
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            })
            print(f"  seed {seed} fold {fold_idx + 1}: acc={acc:5.2f}%  F1={mf1:5.2f}%")
    elapsed = time.time() - t0
    print(f"  Tiempo: {elapsed:.0f}s")
    print(f"  Acc media : {np.mean(accuracies):.2f}% +/- {np.std(accuracies, ddof=1):.2f}%")
    print(f"  F1 macro  : {np.mean(f1s):.2f}% +/- {np.std(f1s, ddof=1):.2f}%")
    return rows, accuracies, f1s, elapsed


def main():
    features_root = os.path.join(PROJECT_ROOT, "data", "features")
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    raw_X, raw_y, class_names = load_raw_dataset(features_root)
    num_classes = len(class_names)
    print(f"Dataset: {len(raw_X)} muestras crudas en {num_classes} clases")

    all_rows = []
    per_model_summary = {}
    t_total = time.time()
    for name, ctor in MODELS.items():
        rows, accs, f1s, elapsed = run_model(name, ctor, raw_X, raw_y, num_classes)
        all_rows.extend(rows)
        lat = quick_latency(ctor, num_classes)
        per_model_summary[name] = {
            "num_parameters": count_params(ctor, num_classes),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "latency_ms": lat,
            "elapsed_seconds": round(elapsed, 1),
        }
    total_elapsed = time.time() - t_total

    # --- CSV detallado ---
    detail_csv = os.path.join(out_dir, "baseline_per_fold.csv")
    with open(detail_csv, "w", encoding="utf-8") as f:
        fieldnames = ["model", "seed", "fold", "accuracy_pct", "f1_macro_pct",
                      "best_epoch", "best_val_loss", "n_train", "n_val"]
        f.write(",".join(fieldnames) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")

    # --- Leer resultados de la CNN 1D para comparacion ---
    cnn_row = None
    kfold_json = os.path.join(out_dir, "kfold_summary.json")
    cnn_latency_json = os.path.join(out_dir, "latency_benchmark.json")
    if os.path.exists(kfold_json):
        with open(kfold_json, encoding="utf-8") as f:
            kfold = json.load(f)
        cnn_acc = kfold["global_metrics_pct"]["accuracy"]
        cnn_f1 = kfold["global_metrics_pct"]["f1_macro"]
        cnn_row = {
            "model": "CNN_1D",
            "num_parameters": 178697,  # del benchmark de latencia
            "accuracy_mean": cnn_acc["mean"],
            "accuracy_std": cnn_acc["std"],
            "f1_macro_mean": cnn_f1["mean"],
            "f1_macro_std": cnn_f1["std"],
        }
        if os.path.exists(cnn_latency_json):
            with open(cnn_latency_json, encoding="utf-8") as f:
                lat_json = json.load(f)
            b1 = lat_json["inference_only_ms"]["batch_1"]
            cnn_row["latency_ms"] = {
                "mean_ms": b1["mean"], "p50_ms": b1["p50"],
                "p95_ms": b1["p95"], "p99_ms": b1["p99"],
            }
            cnn_row["num_parameters"] = lat_json["model"]["num_parameters"]

    # --- Tabla comparativa CSV ---
    comp_csv = os.path.join(out_dir, "baseline_comparison.csv")
    with open(comp_csv, "w", encoding="utf-8") as f:
        f.write("model,num_parameters,accuracy_mean,accuracy_std,"
                "f1_macro_mean,f1_macro_std,latency_mean_ms,latency_p99_ms\n")

        def write_row(name, info):
            lat = info.get("latency_ms", {})
            f.write(f"{name},{info['num_parameters']},"
                    f"{info['accuracy_mean']:.2f},{info['accuracy_std']:.2f},"
                    f"{info['f1_macro_mean']:.2f},{info['f1_macro_std']:.2f},"
                    f"{lat.get('mean_ms', 0):.3f},{lat.get('p99_ms', 0):.3f}\n")

        if cnn_row is not None:
            write_row("CNN_1D", cnn_row)
        for name in MODELS:
            write_row(name, per_model_summary[name])

    # --- JSON comparativo ---
    comp_json_path = os.path.join(out_dir, "baseline_comparison.json")
    payload = {
        "config": {
            "n_splits": N_SPLITS, "seeds": SEEDS,
            "max_epochs": MAX_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
            "seq_length": SEQ_LENGTH, "num_features": NUM_FEATURES,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "latency_warmup": LAT_WARMUP, "latency_runs": LAT_RUNS,
        },
        "results": {
            **({"CNN_1D": cnn_row} if cnn_row is not None else {}),
            **per_model_summary,
        },
        "total_elapsed_seconds": round(total_elapsed, 1),
    }
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # --- Resumen final ---
    print("\n" + "=" * 88)
    print(f"{'Modelo':<13}{'Params':>10}{'Accuracy':>22}{'F1 macro':>22}{'Lat. p99':>11}")
    print("=" * 88)
    if cnn_row is not None:
        lat = cnn_row.get("latency_ms", {})
        print(f"{'CNN_1D':<13}{cnn_row['num_parameters']:>10,}"
              f"{cnn_row['accuracy_mean']:>10.2f}% +/- {cnn_row['accuracy_std']:>4.2f}%"
              f"{cnn_row['f1_macro_mean']:>10.2f}% +/- {cnn_row['f1_macro_std']:>4.2f}%"
              f"{lat.get('p99_ms', 0):>8.2f} ms")
    for name in MODELS:
        info = per_model_summary[name]
        lat = info["latency_ms"]
        print(f"{name:<13}{info['num_parameters']:>10,}"
              f"{info['accuracy_mean']:>10.2f}% +/- {info['accuracy_std']:>4.2f}%"
              f"{info['f1_macro_mean']:>10.2f}% +/- {info['f1_macro_std']:>4.2f}%"
              f"{lat['p99_ms']:>8.2f} ms")
    print(f"\nTiempo total: {total_elapsed:.0f}s")
    print(f"Salidas en:")
    print(f"  {detail_csv}")
    print(f"  {comp_csv}")
    print(f"  {comp_json_path}")


if __name__ == "__main__":
    main()
