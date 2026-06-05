"""
Evaluación offline (holdout) del modelo signs_cnn.pth ya entrenado.

Reproduce EXACTAMENTE el split de validación de train_model.py (mismo
RANDOM_SEED=42 y mismo prepare_dataset sin leakage) y evalúa el modelo
guardado sobre esas muestras de validación, que NO se usaron para entrenar.
Produce, en evaluation/results/:
  - metrics_summary.json    -> métricas globales
  - metrics_per_class.csv   -> precision/recall/F1/soporte por clase
  - confusion_matrix.png    -> matriz de confusión

Relación con kfold_evaluation.py:
    Este script es un check de un solo split (~20 muestras de validación), por
    lo que su número es honesto pero ruidoso. La métrica de referencia para
    reportar generalización es la de kfold_evaluation.py (5-fold x 3 seeds),
    que promedia 15 entrenamientos. Ambos comparten el mismo protocolo sin
    leakage; este sirve para inspeccionar el modelo concreto que se despliega.

Uso (desde sign_recognition/):
    python evaluation/evaluate_model.py
"""

import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Importar arquitectura y pipeline de datos desde train_model.py (en ../src)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from train_model import (
    SignCNN, prepare_dataset, RANDOM_SEED,
)


def per_class_metrics(y_true, y_pred, num_classes):
    """Devuelve precisión, recall, f1 y soporte por clase (en %)."""
    out = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out.append({
            "class_id": c,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "support": support,
        })
    return out


def weighted_avg(per_class, key, total_support):
    return sum(m[key] * m["support"] for m in per_class) / total_support if total_support > 0 else 0.0


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion (validacion)")
    thr = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_root = os.path.join(project_root, "data", "features")
    model_path = os.path.join(project_root, "models", "signs_cnn.pth")
    output_dir = os.path.join(project_root, "evaluation", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Reproducir el mismo split que en entrenamiento
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Reproduciendo dataset de validacion...")
    (X_train, y_train, X_val, y_val,
     label_map, num_classes, feat_mean, feat_std) = prepare_dataset(features_root)

    print(f"\nCargando modelo desde {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SignCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_label = checkpoint["idx_to_label"]
    class_names = [idx_to_label[i] for i in range(num_classes)]

    X_val_t = torch.tensor(X_val).permute(0, 2, 1).float()
    y_val_t = torch.tensor(y_val)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        output = model(X_val_t)
        val_loss = criterion(output, y_val_t).item()
        preds = output.argmax(dim=1).numpy()

    y_true = y_val.astype(np.int64)
    y_pred = preds.astype(np.int64)

    # Métricas
    accuracy = float((y_pred == y_true).mean())
    per_class = per_class_metrics(y_true, y_pred, num_classes)
    total_support = sum(m["support"] for m in per_class)
    precision_w = weighted_avg(per_class, "precision", total_support) / 100
    recall_w = weighted_avg(per_class, "recall", total_support) / 100
    f1_w = weighted_avg(per_class, "f1", total_support) / 100

    metrics_summary = {
        "accuracy_global_pct": round(accuracy * 100, 2),
        "precision_ponderada_pct": round(precision_w * 100, 2),
        "recall_ponderado_pct": round(recall_w * 100, 2),
        "f1_ponderado_pct": round(f1_w * 100, 2),
        "perdida_validacion_final": round(val_loss, 4),
        "muestras_validacion": int(len(y_val)),
        "num_classes": int(num_classes),
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    csv_lines = ["clase,precision,recall,f1,soporte"]
    for name, m in zip(class_names, per_class):
        csv_lines.append(
            f"{name},{m['precision']:.2f},{m['recall']:.2f},{m['f1']:.2f},{m['support']}"
        )
    with open(os.path.join(output_dir, "metrics_per_class.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # Matriz de confusión
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))

    print("\n" + "=" * 60)
    print("METRICAS GLOBALES")
    print("=" * 60)
    print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))
    print("\nMETRICAS POR CLASE")
    print("=" * 60)
    print("\n".join(csv_lines))
    print(f"\nArchivos generados en: {output_dir}/")
    print("  - metrics_summary.json")
    print("  - metrics_per_class.csv")
    print("  - confusion_matrix.png")


if __name__ == "__main__":
    main()
