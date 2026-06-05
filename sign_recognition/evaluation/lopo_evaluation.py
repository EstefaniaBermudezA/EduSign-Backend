"""
Evaluación Leave-One-Participant-Out (LOPO) del clasificador CNN 1D de señas LSC.

Responde a la pregunta relevante para el despliegue en VR: ¿qué tan bien
generaliza el modelo a una persona nueva que nunca vio?

A diferencia de kfold_evaluation.py (que reparte muestras al azar y por tanto
mezcla a la misma persona entre train y validación), aquí cada fold deja fuera
a todas las señas de un participante: se entrena con los otros 10 y se evalúa
con el participante retenido. Se rota por los 11 participantes (p01..p11).

Mantiene el mismo protocolo sin leakage del k-fold (reutiliza sus utilidades):
 - El split se hace sobre las muestras crudas, por participante.
 - El data augmentation se aplica solo a las muestras de train.
 - La normalización z-score se calcula solo sobre train y se aplica a val.
 - Se repite con varias semillas para promediar el ruido de inicialización.

Salidas (en evaluation/results/):
 - lopo_per_fold.csv          -> una fila por seed x participante con acc/F1
 - lopo_per_participant.csv   -> una fila por participante (media +/- std de seeds)
 - lopo_summary.json          -> resumen agregado (global y por participante/clase)
 - lopo_confusion_matrix.png  -> matriz de confusión agregada sobre todos los folds

Uso (desde sign_recognition/):
    python evaluation/lopo_evaluation.py
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# Reutilizar el protocolo sin leakage definido en kfold_evaluation.py
# ----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, SRC_DIR)

from kfold_evaluation import ( 
    prepare_fold_data,
    train_one_fold,
    per_class_counts,
    macro_average,
    confusion_matrix,
    plot_confusion_matrix,
)

# ----------------------------------------------------------------------------
# Configuración del experimento
# ----------------------------------------------------------------------------
SEEDS = [42, 123, 2025]


# ----------------------------------------------------------------------------
# Carga de datos con identificador de participante
# ----------------------------------------------------------------------------

def load_raw_dataset_with_participants(features_root):
    """Carga las secuencias .npy crudas y el participante de cada una.

    El participante se infiere del prefijo del nombre de archivo (p.ej.
    "p07_nilo.npy" -> "p07").

    Devuelve:
        raw_X: lista de arrays (n_frames, NUM_FEATURES)
        raw_y: np.array (n_samples,) con etiquetas enteras
        participants: lista (n_samples,) con el id de participante
        class_names: lista ordenada de nombres de clase
    """
    class_names = sorted([
        d for d in os.listdir(features_root)
        if os.path.isdir(os.path.join(features_root, d))
    ])
    if not class_names:
        raise SystemExit(f"[ERROR] No se encontraron clases en: {features_root}")

    label_map = {name: i for i, name in enumerate(class_names)}
    raw_X, raw_y, participants = [], [], []
    for class_name in class_names:
        class_dir = os.path.join(features_root, class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        for f in files:
            pid = f.split("_")[0]  # "p07_nilo.npy" -> "p07"
            raw_X.append(np.load(os.path.join(class_dir, f)))
            raw_y.append(label_map[class_name])
            participants.append(pid)

    return raw_X, np.array(raw_y, dtype=np.int64), participants, class_names


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    features_root = os.path.join(PROJECT_ROOT, "data", "features")
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(features_root):
        raise SystemExit(f"[ERROR] No se encontro: {features_root}")

    print(f"Cargando dataset crudo desde: {features_root}")
    raw_X, raw_y, participants, class_names = load_raw_dataset_with_participants(features_root)
    num_classes = len(class_names)
    participants_unique = sorted(set(participants))
    n_part = len(participants_unique)
    print(f"  {len(raw_X)} muestras crudas | {num_classes} clases | {n_part} participantes")
    for p in participants_unique:
        print(f"    {p}: {sum(1 for q in participants if q == p)} muestras")

    per_fold_rows = []          # una fila por seed x participante
    aggregate_cm = np.zeros((num_classes, num_classes), dtype=int)
    # acumuladores por participante (combinando seeds)
    acc_by_part = defaultdict(list)
    f1_by_part = defaultdict(list)
    # acumuladores globales (todas las combinaciones seed x participante)
    all_acc, all_f1 = [], []
    per_class_agg = defaultdict(lambda: {"precision_pct": [], "recall_pct": [], "f1_pct": []})

    t0 = time.time()
    for seed in SEEDS:
        print(f"\n========== SEED {seed} ==========")
        for pid in participants_unique:
            val_idx = np.array([i for i, p in enumerate(participants) if p == pid])
            train_idx = np.array([i for i, p in enumerate(participants) if p != pid])

            X_tr, y_tr, X_va, y_va = prepare_fold_data(raw_X, raw_y, train_idx, val_idx)
            val_preds, _val_probs, best_epoch, best_val_loss = train_one_fold(
                X_tr, y_tr, X_va, y_va, num_classes, seed
            )

            acc = float((val_preds == y_va).mean())
            pcs = per_class_counts(y_va, val_preds, num_classes)
            mp, mr, mf1 = macro_average(pcs)
            cm = confusion_matrix(y_va, val_preds, num_classes)
            aggregate_cm += cm

            acc_by_part[pid].append(acc * 100)
            f1_by_part[pid].append(mf1)
            all_acc.append(acc * 100)
            all_f1.append(mf1)
            for c in range(num_classes):
                name = class_names[c]
                per_class_agg[name]["precision_pct"].append(pcs[c]["precision_pct"])
                per_class_agg[name]["recall_pct"].append(pcs[c]["recall_pct"])
                per_class_agg[name]["f1_pct"].append(pcs[c]["f1_pct"])

            per_fold_rows.append({
                "seed": seed,
                "participant": pid,
                "accuracy_pct": acc * 100,
                "precision_macro_pct": mp,
                "recall_macro_pct": mr,
                "f1_macro_pct": mf1,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
            })

            print(f"  hold-out {pid}: acc={acc*100:6.2f}%  F1_macro={mf1:6.2f}%  "
                  f"(best_epoch={best_epoch})")

    t1 = time.time()

    # ------------------- Helpers -------------------
    def ms(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    # ------------------- CSV por fold -------------------
    detail_csv = os.path.join(out_dir, "lopo_per_fold.csv")
    with open(detail_csv, "w", encoding="utf-8") as f:
        fieldnames = ["seed", "participant", "accuracy_pct",
                      "precision_macro_pct", "recall_macro_pct", "f1_macro_pct",
                      "best_epoch", "best_val_loss", "n_train", "n_val"]
        f.write(",".join(fieldnames) + "\n")
        for r in per_fold_rows:
            f.write(",".join(str(r[k]) for k in fieldnames) + "\n")

    # ------------------- CSV por participante -------------------
    part_csv = os.path.join(out_dir, "lopo_per_participant.csv")
    with open(part_csv, "w", encoding="utf-8") as f:
        f.write("participant,accuracy_mean,accuracy_std,f1_macro_mean,f1_macro_std,n_seeds\n")
        for pid in participants_unique:
            a = ms(acc_by_part[pid])
            fa = ms(f1_by_part[pid])
            f.write(f"{pid},{a['mean']:.2f},{a['std']:.2f},"
                    f"{fa['mean']:.2f},{fa['std']:.2f},{len(acc_by_part[pid])}\n")

    # ------------------- Resumen JSON -------------------
    summary = {
        "config": {
            "method": "leave-one-participant-out",
            "seeds": SEEDS,
            "n_participants": n_part,
            "n_folds_total": n_part * len(SEEDS),
        },
        "dataset": {
            "num_classes": num_classes,
            "class_names": class_names,
            "n_raw_samples_total": int(len(raw_X)),
            "participants": participants_unique,
        },
        "global_metrics_pct": {
            "accuracy": ms(all_acc),
            "f1_macro": ms(all_f1),
        },
        "per_participant_accuracy_pct": {
            pid: ms(acc_by_part[pid]) for pid in participants_unique
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
    summary_json = os.path.join(out_dir, "lopo_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------- Matriz de confusión agregada -------------------
    cm_png = os.path.join(out_dir, "lopo_confusion_matrix.png")
    g = summary["global_metrics_pct"]
    title = (f"Matriz de confusion LOPO agregada\n"
             f"{n_part} participantes x {len(SEEDS)} seeds | "
             f"acc {g['accuracy']['mean']:.2f}% +/- {g['accuracy']['std']:.2f}%")
    plot_confusion_matrix(aggregate_cm, class_names, cm_png, title)

    # ------------------- Resumen en consola -------------------
    print("\n" + "=" * 64)
    print(f"RESUMEN LOPO ({n_part} participantes x {len(SEEDS)} seeds "
          f"= {n_part * len(SEEDS)} entrenamientos)")
    print("=" * 64)
    print(f"  Accuracy (persona nueva) : {g['accuracy']['mean']:.2f}% +/- {g['accuracy']['std']:.2f}%")
    print(f"  F1 macro                 : {g['f1_macro']['mean']:.2f}% +/- {g['f1_macro']['std']:.2f}%")
    print(f"\n  Accuracy por participante (media de seeds):")
    for pid in participants_unique:
        a = ms(acc_by_part[pid])
        print(f"    {pid}: {a['mean']:6.2f}% +/- {a['std']:5.2f}%")
    print(f"\n  Tiempo total: {summary['elapsed_seconds']} s")
    print(f"\nSalidas en:")
    for path in (detail_csv, part_csv, summary_json, cm_png):
        print(f"  {path}")


if __name__ == "__main__":
    main()
