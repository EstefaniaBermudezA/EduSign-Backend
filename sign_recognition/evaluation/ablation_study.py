"""
Estudio de ablacion para el clasificador CNN 1D de senas LSC.

Reutiliza la metodologia 5-fold x 3 seeds SIN leakage definida en
kfold_evaluation.py y evalua cuatro variantes:

  - full           : pipeline completo (augmentation + z-score + class weights)
  - no_augment     : sin data augmentation (solo las 99 muestras crudas)
  - no_zscore      : sin normalizacion z-score (entrada cruda)
  - no_class_w     : sin pesos por clase en la entropia cruzada

Para cada variante se reporta accuracy y F1 macro (media +/- std combinando
folds y seeds) y el delta vs la variante 'full'.

Salidas (en evaluation/results/):
 - ablation_per_fold.csv     -> una fila por variante x seed x fold
 - ablation_summary.csv      -> una fila por variante con metricas agregadas
 - ablation_comparison.json  -> deltas vs full (accuracy y F1)

Uso (desde sign_recognition_eval/):
    python evaluation/ablation_study.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reutilizar utilidades del kfold (que a su vez importa del modulo original)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from kfold_evaluation import (  # noqa: E402
    set_seed,
    load_raw_dataset,
    stratified_kfold_indices,
    per_class_counts,
    macro_average,
)
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
# Config
# ----------------------------------------------------------------------------
N_SPLITS = 5
SEEDS = [42, 123, 2025]
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 50
SCHEDULER_PATIENCE = 25
SCHEDULER_FACTOR = 0.5
WEIGHT_DECAY = 1e-4

VARIANTS = [
    "full",
    "no_augment",
    "no_zscore",
    "no_class_w",
]


# ----------------------------------------------------------------------------
# Preparacion de datos parametrizada por variante
# ----------------------------------------------------------------------------

def prepare_fold_data_variant(raw_X, raw_y, train_idx, val_idx,
                              use_augment, use_zscore):
    """Construye X_train y X_val activando/desactivando augment y z-score."""
    # --- Train ---
    train_seqs = []
    train_labels = []
    for i in train_idx:
        normalized = normalize_sequence(raw_X[i], SEQ_LENGTH)
        train_seqs.append(normalized)
        train_labels.append(raw_y[i])
        if use_augment:
            for _ in range(AUGMENTATION_PER_SAMPLE):
                aug = augment_sample(raw_X[i], SEQ_LENGTH)
                train_seqs.append(aug)
                train_labels.append(raw_y[i])
    X_train = np.array(train_seqs, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int64)

    # --- Val ---
    val_seqs = [normalize_sequence(raw_X[i], SEQ_LENGTH) for i in val_idx]
    X_val = np.array(val_seqs, dtype=np.float32)
    y_val = raw_y[val_idx].copy()

    # --- Z-score (opcional) ---
    if use_zscore:
        feat_mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
        feat_std = X_train.reshape(-1, NUM_FEATURES).std(axis=0) + 1e-8
        X_train = (X_train - feat_mean) / feat_std
        X_val = (X_val - feat_mean) / feat_std

    return X_train, y_train, X_val, y_val


def train_one_fold_variant(X_train, y_train, X_val, y_val,
                           num_classes, seed, use_class_w):
    """Entrena un SignCNN sobre un fold con/sin pesos por clase."""
    set_seed(seed)

    X_train_t = torch.tensor(X_train).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val).permute(0, 2, 1)
    y_val_t = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    if use_class_w:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * num_classes
        weights_tensor = torch.tensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    model = SignCNN(num_classes=num_classes)
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
        val_output = model(X_val_t)
        val_preds = val_output.argmax(dim=1).numpy()

    return val_preds, best_epoch, best_val_loss


# ----------------------------------------------------------------------------
# Orquestacion por variante
# ----------------------------------------------------------------------------

def run_variant(variant_name, raw_X, raw_y, num_classes):
    """Corre 5-fold x 3 seeds para una variante y devuelve metricas."""
    use_augment = variant_name != "no_augment"
    use_zscore = variant_name != "no_zscore"
    use_class_w = variant_name != "no_class_w"

    print(f"\n========== VARIANTE: {variant_name} "
          f"(augment={use_augment}, zscore={use_zscore}, class_w={use_class_w}) ==========")

    accuracies, macro_f1s, rows = [], [], []
    t0 = time.time()
    for seed in SEEDS:
        for fold_idx, (train_idx, val_idx) in enumerate(
            stratified_kfold_indices(raw_y, N_SPLITS, seed)
        ):
            X_tr, y_tr, X_va, y_va = prepare_fold_data_variant(
                raw_X, raw_y, train_idx, val_idx, use_augment, use_zscore
            )
            val_preds, best_epoch, best_val_loss = train_one_fold_variant(
                X_tr, y_tr, X_va, y_va, num_classes, seed, use_class_w
            )
            acc = float((val_preds == y_va).mean())
            pcs = per_class_counts(y_va, val_preds, num_classes)
            mp, mr, mf1 = macro_average(pcs)
            accuracies.append(acc * 100)
            macro_f1s.append(mf1)
            rows.append({
                "variant": variant_name, "seed": seed, "fold": fold_idx + 1,
                "accuracy_pct": acc * 100, "f1_macro_pct": mf1,
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            })
            print(f"  seed {seed} fold {fold_idx + 1}: acc={acc*100:5.2f}%  F1={mf1:5.2f}%")
    elapsed = time.time() - t0
    print(f"  Tiempo {variant_name}: {elapsed:.0f}s")
    print(f"  Acc media   : {np.mean(accuracies):.2f}% +/- {np.std(accuracies, ddof=1):.2f}%")
    print(f"  F1 macro    : {np.mean(macro_f1s):.2f}% +/- {np.std(macro_f1s, ddof=1):.2f}%")
    return rows, accuracies, macro_f1s, elapsed


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    features_root = os.path.join(PROJECT_ROOT, "data", "features")
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    raw_X, raw_y, class_names = load_raw_dataset(features_root)
    num_classes = len(class_names)
    print(f"Dataset: {len(raw_X)} muestras crudas en {num_classes} clases")

    all_rows = []
    summary_by_variant = {}
    t_total = time.time()
    for variant in VARIANTS:
        rows, accs, f1s, elapsed = run_variant(variant, raw_X, raw_y, num_classes)
        all_rows.extend(rows)
        summary_by_variant[variant] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "elapsed_seconds": round(elapsed, 1),
        }
    total_elapsed = time.time() - t_total

    # --- CSV detallado por fold ---
    detail_csv = os.path.join(out_dir, "ablation_per_fold.csv")
    with open(detail_csv, "w", encoding="utf-8") as f:
        fieldnames = ["variant", "seed", "fold", "accuracy_pct", "f1_macro_pct",
                      "best_epoch", "best_val_loss", "n_train", "n_val"]
        f.write(",".join(fieldnames) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")

    # --- CSV resumen por variante ---
    summary_csv = os.path.join(out_dir, "ablation_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("variant,accuracy_mean,accuracy_std,f1_macro_mean,f1_macro_std,"
                "delta_accuracy_vs_full,delta_f1_vs_full,elapsed_seconds\n")
        full = summary_by_variant["full"]
        for variant in VARIANTS:
            s = summary_by_variant[variant]
            d_acc = s["accuracy_mean"] - full["accuracy_mean"]
            d_f1 = s["f1_macro_mean"] - full["f1_macro_mean"]
            f.write(f"{variant},{s['accuracy_mean']:.2f},{s['accuracy_std']:.2f},"
                    f"{s['f1_macro_mean']:.2f},{s['f1_macro_std']:.2f},"
                    f"{d_acc:+.2f},{d_f1:+.2f},{s['elapsed_seconds']}\n")

    # --- JSON con deltas ---
    full = summary_by_variant["full"]
    comparison = {
        "config": {
            "n_splits": N_SPLITS, "seeds": SEEDS,
            "max_epochs": MAX_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
            "augmentation_per_sample": AUGMENTATION_PER_SAMPLE,
            "seq_length": SEQ_LENGTH, "num_features": NUM_FEATURES,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "variants": {
            variant: {
                **summary_by_variant[variant],
                "delta_accuracy_vs_full": summary_by_variant[variant]["accuracy_mean"] - full["accuracy_mean"],
                "delta_f1_vs_full": summary_by_variant[variant]["f1_macro_mean"] - full["f1_macro_mean"],
            }
            for variant in VARIANTS
        },
        "total_elapsed_seconds": round(total_elapsed, 1),
    }
    comparison_json = os.path.join(out_dir, "ablation_comparison.json")
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # --- Resumen final en consola ---
    print("\n" + "=" * 70)
    print(f"{'Variante':<14}{'Accuracy':>22}{'F1 macro':>22}{'Delta F1':>10}")
    print("=" * 70)
    for variant in VARIANTS:
        s = summary_by_variant[variant]
        d_f1 = s["f1_macro_mean"] - full["f1_macro_mean"]
        print(f"{variant:<14}"
              f"{s['accuracy_mean']:>10.2f}% +/- {s['accuracy_std']:>4.2f}%"
              f"{s['f1_macro_mean']:>10.2f}% +/- {s['f1_macro_std']:>4.2f}%"
              f"{d_f1:>+10.2f}")
    print(f"\nTiempo total: {total_elapsed:.0f}s")
    print(f"Salidas en:")
    print(f"  {detail_csv}")
    print(f"  {summary_csv}")
    print(f"  {comparison_json}")


if __name__ == "__main__":
    main()
