"""
Entrenamiento de un modelo 1D-CNN para reconocer senas en LSC.

Descubre automaticamente las carpetas en data/features/ y asigna una clase
por carpeta. Incluye augmentation robusto y normalizacion z-score para
generalizar mejor a datos en tiempo real.

Uso:
    python src/train_model.py
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Configuracion ----------
SEQ_LENGTH = 60          # Frames por secuencia (mas para capturar movimiento)
NUM_FEATURES = 132       # Features por frame
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
AUGMENTATION_PER_SAMPLE = 8

# ---------------------------------------------------------------------------
# Modelo: 1D-CNN multi-clase
# ---------------------------------------------------------------------------

class SignCNN(nn.Module):
    """Red 1D-CNN para clasificacion multi-clase de secuencias de landmarks."""

    def __init__(self, num_features=NUM_FEATURES, seq_length=SEQ_LENGTH, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: captura patrones locales
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Bloque 2: patrones de mediano alcance
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Bloque 3
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Bloque 4: compresion
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------

def normalize_sequence(seq, target_length=SEQ_LENGTH):
    """Interpola una secuencia de N frames a target_length frames."""
    n_frames = len(seq)
    if n_frames == target_length:
        return seq.copy() if isinstance(seq, np.ndarray) else seq
    n_feat = seq.shape[1] if len(seq.shape) > 1 else NUM_FEATURES
    indices = np.linspace(0, n_frames - 1, target_length)
    result = np.zeros((target_length, n_feat))
    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = min(low + 1, n_frames - 1)
        frac = idx - low
        result[i] = seq[low] * (1 - frac) + seq[high] * frac
    return result


# ---------------------------------------------------------------------------
# Data Augmentation robusto
# ---------------------------------------------------------------------------

def aug_gaussian_noise(seq, sigma=0.03):
    """Ruido gaussiano."""
    return seq + np.random.randn(*seq.shape) * sigma


def aug_scale(seq, low=0.85, high=1.15):
    """Escalar coordenadas (simula persona mas cerca/lejos)."""
    scale = np.random.uniform(low, high)
    return seq * scale


def aug_time_warp(seq):
    """Deformacion temporal no lineal (acelera/desacelera partes)."""
    n = len(seq)
    # Crear indices no lineales
    x = np.linspace(0, 1, n)
    # Punto de quiebre aleatorio
    warp_point = np.random.uniform(0.2, 0.8)
    warp_amount = np.random.uniform(0.5, 2.0)

    warped = np.where(
        x < warp_point,
        x * warp_amount / (warp_amount * warp_point + (1 - warp_point)),
        (x - warp_point) / (warp_amount * warp_point + (1 - warp_point)) +
        warp_amount * warp_point / (warp_amount * warp_point + (1 - warp_point))
    )
    warped = np.clip(warped * (n - 1), 0, n - 1)

    result = np.zeros_like(seq)
    for i, idx in enumerate(warped):
        low = int(np.floor(idx))
        high = min(low + 1, n - 1)
        frac = idx - low
        result[i] = seq[low] * (1 - frac) + seq[high] * frac
    return result


def aug_speed_change(seq_raw, target_length):
    """Cambia la velocidad: toma un sub-segmento y lo estira a target_length."""
    n = len(seq_raw)
    # Tomar entre 60% y 100% de la secuencia original
    crop_ratio = np.random.uniform(0.6, 1.0)
    crop_len = max(int(n * crop_ratio), target_length)
    crop_len = min(crop_len, n)
    start = np.random.randint(0, n - crop_len + 1)
    cropped = seq_raw[start:start + crop_len]
    return normalize_sequence(cropped, target_length)


def aug_spatial_jitter(seq, jitter=0.015):
    """Desplazamiento espacial constante (simula persona en distinta posicion)."""
    offset = np.random.randn(seq.shape[1]) * jitter
    return seq + offset


def aug_mirror_hands(seq):
    """Intercambia mano izquierda y derecha (indices 0-62 <-> 63-125)."""
    mirrored = seq.copy()
    mirrored[:, :63], mirrored[:, 63:126] = seq[:, 63:126].copy(), seq[:, :63].copy()
    # Invertir eje X de las manos (negar la coordenada x de cada landmark)
    for hand_start in [0, 63]:
        for lm in range(21):
            x_idx = hand_start + lm * 3
            mirrored[:, x_idx] = -mirrored[:, x_idx]
    # Invertir eje X de hombros tambien
    mirrored[:, 126], mirrored[:, 129] = -seq[:, 129].copy(), -seq[:, 126].copy()
    mirrored[:, 127], mirrored[:, 130] = seq[:, 130].copy(), seq[:, 127].copy()
    mirrored[:, 128], mirrored[:, 131] = seq[:, 131].copy(), seq[:, 128].copy()
    return mirrored


def aug_drop_hand(seq):
    """Pone una mano a cero aleatoriamente (simula oclusion parcial)."""
    result = seq.copy()
    if np.random.rand() > 0.5:
        result[:, :63] = 0  # Pone mano izquierda a cero
    else:
        result[:, 63:126] = 0  # Pone mano derecha a cero
    return result


def augment_sample(seq_raw, target_length):
    """Aplica una combinacion aleatoria de augmentations a una secuencia raw."""
    # Primero, variacion de velocidad (opera sobre la secuencia raw)
    if np.random.rand() > 0.4:
        normalized = aug_speed_change(seq_raw, target_length)
    else:
        normalized = normalize_sequence(seq_raw, target_length)

    # Luego, augmentations sobre la secuencia normalizada
    if np.random.rand() > 0.4:
        normalized = aug_time_warp(normalized)

    if np.random.rand() > 0.5:
        normalized = aug_gaussian_noise(normalized, sigma=np.random.uniform(0.01, 0.05))

    if np.random.rand() > 0.5:
        normalized = aug_scale(normalized)

    if np.random.rand() > 0.6:
        normalized = aug_spatial_jitter(normalized)

    if np.random.rand() > 0.8:
        normalized = aug_mirror_hands(normalized)

    if np.random.rand() > 0.85:
        normalized = aug_drop_hand(normalized)

    return normalized


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_features(features_dir):
    """Carga todos los archivos .npy de un directorio."""
    sequences = []
    files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
    for f in files:
        data = np.load(os.path.join(features_dir, f))
        sequences.append(data)
        print(f"    {f}: {data.shape}")
    return sequences


def discover_classes(features_root):
    """Descubre automaticamente las clases a partir de subcarpetas."""
    class_dirs = sorted([
        d for d in os.listdir(features_root)
        if os.path.isdir(os.path.join(features_root, d))
    ])
    return class_dirs


def prepare_dataset(features_root, validation_split=VALIDATION_SPLIT, random_seed=RANDOM_SEED):
    """Carga las clases y separa train/val SIN data leakage.

    Orden correcto (sin fuga de informacion de validacion):
      1. Separa las secuencias CRUDAS en train/val de forma estratificada,
         ANTES de aumentar nada.
      2. Aplica data augmentation SOLO a las muestras de train. Validacion
         conserva unicamente la version normalizada de cada gesto crudo.
      3. Calcula el z-score con estadisticas de TRAIN y lo aplica a train y val.

    Asi ninguna copia aumentada de un gesto de validacion entra al entrenamiento
    y la normalizacion no "ve" datos de validacion. Es el mismo protocolo que
    kfold_evaluation.py, pero con un unico split (mas rapido, mas ruidoso).
    """
    class_names = discover_classes(features_root)
    if not class_names:
        print(f"[ERROR] No se encontraron carpetas de clases en: {features_root}")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"Clases encontradas ({num_classes}): {class_names}")

    label_map = {name: i for i, name in enumerate(class_names)}

    # 1. Cargar secuencias crudas y separarlas en train/val ANTES de aumentar.
    #    Se usa un RandomState propio para que el split sea reproducible e
    #    independiente del resto de llamadas a np.random (augmentation, etc.).
    rng = np.random.RandomState(random_seed)
    train_raw, train_lbl = [], []
    val_raw, val_lbl = [], []
    for class_name in class_names:
        class_id = label_map[class_name]
        class_dir = os.path.join(features_root, class_name)
        print(f"\n  [{class_name}] (clase {class_id}):")
        seqs = load_features(class_dir)

        order = np.arange(len(seqs))
        rng.shuffle(order)
        n_train = int(round(len(seqs) * (1 - validation_split)))
        if len(seqs) > 1:
            n_train = max(1, min(n_train, len(seqs) - 1))  # garantizar >=1 en val
        for rank, i in enumerate(order):
            if rank < n_train:
                train_raw.append(seqs[i]); train_lbl.append(class_id)
            else:
                val_raw.append(seqs[i]); val_lbl.append(class_id)

    # 2. Augmentation SOLO en train; validacion solo normalizada.
    train_X, train_y = [], []
    for seq, lbl in zip(train_raw, train_lbl):
        train_X.append(normalize_sequence(seq, SEQ_LENGTH))
        train_y.append(lbl)
        for _ in range(AUGMENTATION_PER_SAMPLE):
            train_X.append(augment_sample(seq, SEQ_LENGTH))
            train_y.append(lbl)

    X_train = np.array(train_X, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.int64)
    X_val = np.array([normalize_sequence(seq, SEQ_LENGTH) for seq in val_raw],
                     dtype=np.float32)
    y_val = np.array(val_lbl, dtype=np.int64)

    # 3. Z-score con estadisticas calculadas SOLO sobre train.
    feat_mean = X_train.reshape(-1, NUM_FEATURES).mean(axis=0)
    feat_std = X_train.reshape(-1, NUM_FEATURES).std(axis=0) + 1e-8
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    print(f"\nZ-score (stats de train): mean~{feat_mean.mean():.4f}, std~{feat_std.mean():.4f}")

    # Resumen
    print(f"\n{'='*50}")
    print(f"DATASET (split sin leakage, val={validation_split:.0%})")
    print(f"{'='*50}")
    for name, idx in label_map.items():
        tr_raw = sum(1 for l in train_lbl if l == idx)
        va = sum(1 for l in val_lbl if l == idx)
        print(f"  Clase {idx} ({name:12s}): train_raw={tr_raw:3d} (+aug)  val={va:3d}")
    print(f"  Total: train={len(X_train)} (de {len(train_raw)} crudas)  val={len(X_val)}")

    return X_train, y_train, X_val, y_val, label_map, num_classes, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(features_root, model_output_path):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    (X_train, y_train, X_val, y_val,
     label_map, num_classes, feat_mean, feat_std) = prepare_dataset(features_root)

    # Tensores: (batch, features, seq_length) para Conv1d
    X_train_t = torch.tensor(X_train).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val).permute(0, 2, 1)
    y_val_t = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Pesos para balancear clases
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    weights_tensor = torch.tensor(class_weights)

    # Modelo
    model = SignCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {total_params:,} parametros | {num_classes} clases")
    print(f"Entrenando por {EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    EARLY_STOP_PATIENCE = 50

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_X)
            preds = output.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            val_preds = val_output.argmax(dim=1)
            val_acc = (val_preds == y_val_t).sum().item() / len(y_val_t)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Guardar mejor modelo basado en val_loss (no solo accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping en epoch {epoch} (sin mejora por {EARLY_STOP_PATIENCE} epochs)")
            break

    # Guardar mejor modelo
    model.load_state_dict(best_state)

    # Confusion matrix simple en validacion
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_preds = val_output.argmax(dim=1).numpy()

    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"\n{'='*50}")
    print("MATRIZ DE CONFUSION (Validacion)")
    print(f"{'='*50}")
    header = "Real\\Pred  " + "".join(f"{idx_to_label[i][:5]:>6s}" for i in range(num_classes))
    print(header)
    for real_cls in range(num_classes):
        row = f"{idx_to_label[real_cls][:9]:9s}  "
        for pred_cls in range(num_classes):
            count = ((y_val == real_cls) & (val_preds == pred_cls)).sum()
            row += f"{count:6d}"
        print(row)

    save_data = {
        'model_state_dict': best_state,
        'seq_length': SEQ_LENGTH,
        'num_features': NUM_FEATURES,
        'num_classes': num_classes,
        'label_map': label_map,
        'idx_to_label': idx_to_label,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'best_val_acc': best_val_acc,
    }
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(save_data, model_output_path)

    print(f"\nMejor Val Accuracy: {best_val_acc:.3f} | Val Loss: {best_val_loss:.4f}")
    print(f"Clases: {idx_to_label}")
    print(f"Modelo guardado en: {model_output_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_root = os.path.join(project_root, "data", "features")
    model_path = os.path.join(project_root, "models", "signs_cnn.pth")

    if not os.path.isdir(features_root):
        print(f"[ERROR] No se encontro: {features_root}")
        print("  Ejecuta primero: python src/extract_features.py")
        sys.exit(1)

    train(features_root, model_path)


if __name__ == "__main__":
    main()
