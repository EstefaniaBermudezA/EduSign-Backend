"""
Validacion cruzada estratificada SIN leakage para el clasificador CNN 1D de senas LSC.

Diferencias clave frente a evaluate_model.py:
 - El split se hace sobre las 99 muestras CRUDAS (no aumentadas).
 - El data augmentation se aplica SOLO a las muestras de train de cada fold.
 - La normalizacion z-score se calcula SOLO sobre train de cada fold y se aplica a val.
 - Se repite todo con varias semillas para reportar media +/- desviacion estandar.

Salidas (en evaluation/results/):
 - kfold_per_fold.csv         -> una fila por seed x fold x clase con TP/FP/FN/precision/recall/F1/support
 - kfold_per_fold_global.csv  -> una fila por seed x fold con accuracy y F1 macro
 - kfold_summary.json         -> resumen agregado (media +/- std combinando seeds y folds)
 - kfold_confusion_matrix.png -> matriz de confusion agregada sobre TODOS los folds y seeds

Uso (desde sign_recognition_eval/):
    python evaluation/kfold_evaluation.py
"""

import os
import sys
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Reutilizar arquitectura y augmentations del modulo de entrenamiento original
# ----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from train_model import (  # noqa: E402
    SignCNN,
    normalize_sequence,
    augment_sample,
    SEQ_LENGTH,
    NUM_FEATURES,
    BATCH_SIZE,
    LEARNING_RATE,
    AUGMENTATION_PER_SAMPLE,
)

# ----------------------------------------------------------------------------
# Configuracion del experimento
# ----------------------------------------------------------------------------
N_SPLITS = 5
SEEDS = [42, 123, 2025]
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 50
SCHEDULER_PATIENCE = 25
SCHEDULER_FACTOR = 0.5
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 25


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def set_seed(seed):
    """Fija todas las fuentes de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_dataset(features_root):
    """Carga las secuencias .npy originales (sin augmentation, sin normalizar).

    Devuelve:
        raw_X: lista de arrays con shapes variables (n_frames, NUM_FEATURES)
        raw_y: np.array (n_samples,) con etiquetas enteras
        class_names: lista ordenada de nombres de clase
    """
    class_names = sorted([
        d for d in os.listdir(features_root)
        if os.path.isdir(os.path.join(features_root, d))
    ])
    if not class_names:
        raise SystemExit(f"[ERROR] No se encontraron clases en: {features_root}")

    label_map = {name: i for i, name in enumerate(class_names)}
    raw_X = []
    raw_y = []
    for class_name in class_names:
        class_dir = os.path.join(features_root, class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        for f in files:
            seq = np.load(os.path.join(class_dir, f))
            raw_X.append(seq)
            raw_y.append(label_map[class_name])

    raw_y = np.array(raw_y, dtype=np.int64)
    return raw_X, raw_y, class_names


def stratified_kfold_indices(y, n_splits, seed):
    """StratifiedKFold manual (sin dependencia de sklearn).

    Para cada clase: baraja sus indices y los reparte en n_splits cubos
    rotativos. Devuelve un generador de (train_idx, val_idx) por fold.
    """
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    cubes = [[] for _ in range(n_splits)]
    for c in range(n_classes):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            cubes[i % n_splits].append(idx)
    cubes = [np.array(c) for c in cubes]
    for k in range(n_splits):
        val_idx = cubes[k]
        train_idx = np.concatenate([cubes[j] for j in range(n_splits) if j != k])
        yield train_idx, val_idx


def prepare_fold_data(raw_X, raw_y, train_idx, val_idx):
    """Construye X_train (aumentado + z-score) y X_val (z-score con stats de train).

    Sin leakage: las copias aumentadas viven solo en train; la media y la desviacion
    estandar se calculan SOLO sobre las muestras de train de este fold.
    """
    # --- Train: original + AUGMENTATION_PER_SAMPLE copias aumentadas por muestra ---
    train_seqs = []
    train_labels = []
    for i in train_idx:
        normalized = normalize_sequence(raw_X[i], SEQ_LENGTH)
        train_seqs.append(normalized)
        train_labels.append(raw_y[i])
        for _ in range(AUGMENTATION_PER_SAMPLE):
            aug = augment_sample(raw_X[i], SEQ_LENGTH)
            train_seqs.append(aug)
            train_labels.append(raw_y[i])
    X_train = np.array(train_seqs, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)

    # --- Val: solo normalizacion temporal, sin augmentation ---
    val_seqs = [normalize_sequence(raw_X[i], SEQ_LENGTH) for i in val_idx]
    X_val = np.array(val_seqs, dtype=np.float32)
    y_val = raw_y[val_idx].copy()

    # --- Z-score con estadisticas calculadas SOLO sobre train ---
    feat_mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    feat_std = X_train.reshape(-1, NUM_FEATURES).std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    return X_train, y_train, X_val, y_val


def train_one_fold(X_train, y_train, X_val, y_val, num_classes, seed):
    """Entrena un SignCNN sobre un fold y devuelve predicciones de validacion."""
    set_seed(seed)

    X_train_t = torch.tensor(X_train).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val).permute(0, 2, 1)
    y_val_t = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Pesos por clase (inverso de la frecuencia, normalizado)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    weights_tensor = torch.tensor(class_weights)

    model = SignCNN(num_classes=num_classes)
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
            val_preds = val_output.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

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

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"        epoch {epoch:3d} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_probs = torch.softmax(val_output, dim=1).numpy()
        val_preds = val_output.argmax(dim=1).numpy()

    return val_preds, val_probs, best_epoch, best_val_loss


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------

def per_class_counts(y_true, y_pred, num_classes):
    """Devuelve dict por clase con TP, FP, FN, support y P/R/F1 (en %)."""
    out = {}
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[c] = {
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision_pct": precision * 100,
            "recall_pct": recall * 100,
            "f1_pct": f1 * 100,
        }
    return out


def macro_average(per_class):
    """Promedio macro de P/R/F1 a partir del dict por clase."""
    ps = [m["precision_pct"] for m in per_class.values()]
    rs = [m["recall_pct"] for m in per_class.values()]
    f1s = [m["f1_pct"] for m in per_class.values()]
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, output_path, title):
    """Guarda una matriz de confusion en PNG, con conteos y porcentajes por fila."""
    cm_pct = cm.astype(float)
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_pct = cm_pct / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% por fila")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)",
                    ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    project_root = PROJECT_ROOT
    features_root = os.path.join(project_root, "data", "features")
    out_dir = os.path.join(project_root, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(features_root):
        raise SystemExit(f"[ERROR] No se encontro: {features_root}")

    print(f"Cargando dataset crudo desde: {features_root}")
    raw_X, raw_y, class_names = load_raw_dataset(features_root)
    num_classes = len(class_names)
    print(f"  {len(raw_X)} muestras crudas en {num_classes} clases")
    for c, name in enumerate(class_names):
        print(f"    {c} {name:12s}: {int((raw_y == c).sum())} muestras")

    per_fold_rows = []         # detallado: una fila por seed x fold x clase
    per_fold_global_rows = []  # resumen: una fila por seed x fold
    aggregate_cm = np.zeros((num_classes, num_classes), dtype=int)
    accuracies = []
    macro_p_list, macro_r_list, macro_f1_list = [], [], []

    t0 = time.time()
    for seed in SEEDS:
        print(f"\n========== SEED {seed} ==========")
        for fold_idx, (train_idx, val_idx) in enumerate(
            stratified_kfold_indices(raw_y, N_SPLITS, seed)
        ):
            print(f"  -- fold {fold_idx + 1}/{N_SPLITS} | train={len(train_idx)} val={len(val_idx)}")
            X_tr, y_tr, X_va, y_va = prepare_fold_data(raw_X, raw_y, train_idx, val_idx)
            val_preds, val_probs, best_epoch, best_val_loss = train_one_fold(
                X_tr, y_tr, X_va, y_va, num_classes, seed
            )

            # --- Metricas del fold ---
            acc = float((val_preds == y_va).mean())
            pcs = per_class_counts(y_va, val_preds, num_classes)
            mp, mr, mf1 = macro_average(pcs)
            cm = confusion_matrix(y_va, val_preds, num_classes)
            aggregate_cm += cm

            accuracies.append(acc)
            macro_p_list.append(mp)
            macro_r_list.append(mr)
            macro_f1_list.append(mf1)

            per_fold_global_rows.append({
                "seed": seed,
                "fold": fold_idx + 1,
                "accuracy_pct": acc * 100,
                "precision_macro_pct": mp,
                "recall_macro_pct": mr,
                "f1_macro_pct": mf1,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
            })

            for c in range(num_classes):
                row = {"seed": seed, "fold": fold_idx + 1,
                       "class_id": c, "class_name": class_names[c]}
                row.update(pcs[c])
                per_fold_rows.append(row)

            print(f"     acc={acc*100:.2f}%  F1_macro={mf1:.2f}%  "
                  f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")

    t1 = time.time()

    # ------------------- Guardar CSVs -------------------
    detail_csv = os.path.join(out_dir, "kfold_per_fold.csv")
    with open(detail_csv, "w", encoding="utf-8") as f:
        fieldnames = ["seed", "fold", "class_id", "class_name",
                      "tp", "fp", "fn", "support",
                      "precision_pct", "recall_pct", "f1_pct"]
        f.write(",".join(fieldnames) + "\n")
        for r in per_fold_rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")

    global_csv = os.path.join(out_dir, "kfold_per_fold_global.csv")
    with open(global_csv, "w", encoding="utf-8") as f:
        fieldnames = ["seed", "fold", "accuracy_pct",
                      "precision_macro_pct", "recall_macro_pct", "f1_macro_pct",
                      "best_epoch", "best_val_loss", "n_train", "n_val"]
        f.write(",".join(fieldnames) + "\n")
        for r in per_fold_global_rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")

    # ------------------- Resumen JSON -------------------
    def ms(arr):
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}

    per_class_agg = defaultdict(lambda: {"precision_pct": [], "recall_pct": [], "f1_pct": []})
    for r in per_fold_rows:
        per_class_agg[r["class_name"]]["precision_pct"].append(r["precision_pct"])
        per_class_agg[r["class_name"]]["recall_pct"].append(r["recall_pct"])
        per_class_agg[r["class_name"]]["f1_pct"].append(r["f1_pct"])

    summary = {
        "config": {
            "n_splits": N_SPLITS,
            "seeds": SEEDS,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_factor": SCHEDULER_FACTOR,
            "augmentation_per_sample": AUGMENTATION_PER_SAMPLE,
            "seq_length": SEQ_LENGTH,
            "num_features": NUM_FEATURES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "dataset": {
            "num_classes": num_classes,
            "class_names": class_names,
            "n_raw_samples_total": int(len(raw_X)),
            "n_raw_samples_per_class": {n: int((raw_y == i).sum()) for i, n in enumerate(class_names)},
        },
        "global_metrics_pct": {
            "accuracy": ms([a * 100 for a in accuracies]),
            "precision_macro": ms(macro_p_list),
            "recall_macro": ms(macro_r_list),
            "f1_macro": ms(macro_f1_list),
        },
        "per_class_metrics_pct": {
            name: {
                "precision": ms(per_class_agg[name]["precision_pct"]),
                "recall": ms(per_class_agg[name]["recall_pct"]),
                "f1": ms(per_class_agg[name]["f1_pct"]),
            }
            for name in class_names
        },
        "elapsed_seconds": round(t1 - t0, 1),
    }

    summary_json = os.path.join(out_dir, "kfold_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------- Matriz de confusion agregada -------------------
    cm_png = os.path.join(out_dir, "kfold_confusion_matrix.png")
    title = (f"Matriz de confusion agregada\n"
             f"{N_SPLITS}-fold x {len(SEEDS)} seeds | "
             f"acc {summary['global_metrics_pct']['accuracy']['mean']:.2f}% "
             f"+/- {summary['global_metrics_pct']['accuracy']['std']:.2f}%")
    plot_confusion_matrix(aggregate_cm, class_names, cm_png, title)

    # ------------------- Resumen en consola -------------------
    g = summary["global_metrics_pct"]
    print("\n" + "=" * 60)
    print("RESUMEN k-fold (5 folds x 3 seeds = 15 trainings)")
    print("=" * 60)
    print(f"  Accuracy        : {g['accuracy']['mean']:.2f}% +/- {g['accuracy']['std']:.2f}%")
    print(f"  Precision macro : {g['precision_macro']['mean']:.2f}% +/- {g['precision_macro']['std']:.2f}%")
    print(f"  Recall macro    : {g['recall_macro']['mean']:.2f}% +/- {g['recall_macro']['std']:.2f}%")
    print(f"  F1 macro        : {g['f1_macro']['mean']:.2f}% +/- {g['f1_macro']['std']:.2f}%")
    print(f"\n  Tiempo total    : {summary['elapsed_seconds']} s")
    print(f"\nSalidas en:")
    print(f"  {detail_csv}")
    print(f"  {global_csv}")
    print(f"  {summary_json}")
    print(f"  {cm_png}")


if __name__ == "__main__":
    main()
