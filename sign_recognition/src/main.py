"""Detector de señas LSC en tiempo real desde la webcam (con visualización).

Captura video, extrae landmarks de manos y hombros con MediaPipe y los acumula
en un buffer. Cuando detecta que las manos se quedan quietas (variancia baja
sostenida), clasifica la secuencia capturada con la SignCNN y muestra la seña
reconocida junto con las probabilidades sobre el frame. Se ejecuta con:
    python src/main.py [--camera 0] [--threshold ...] [--confidence ...]
"""

import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import SignCNN, SEQ_LENGTH, NUM_FEATURES, normalize_sequence

VARIANCE_THRESHOLD = 0.0004
STILLNESS_DURATION = 0.5
MIN_ACTIVE_FRAMES = 15
MAX_BUFFER_FRAMES = 300
CONFIDENCE_THRESHOLD = 0.55
COOLDOWN_SECONDS = 2.0

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

GREEN = (0, 200, 0)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 220)
BLACK = (0, 0, 0)


def load_model(model_path):
    """Carga el checkpoint de la SignCNN y sus metadatos de inferencia.

    Returns:
        Tupla (model en modo eval, idx_to_label, feat_mean, feat_std). Los dos
        últimos son los estadísticos z-score con que se normalizó el entreno y
        son necesarios para clasificar coherentemente en producción.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    num_classes = checkpoint.get('num_classes', 9)
    idx_to_label = checkpoint.get('idx_to_label', {})
    feat_mean = checkpoint.get('feat_mean', None)
    feat_std = checkpoint.get('feat_std', None)

    model = SignCNN(
        num_features=checkpoint.get('num_features', NUM_FEATURES),
        seq_length=checkpoint.get('seq_length', SEQ_LENGTH),
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Modelo cargado: {model_path}")
    print(f"  Clases: {idx_to_label}")
    print(f"  Val accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    if feat_mean is not None:
        print(f"  Z-score: mean~{feat_mean.mean():.4f}, std~{feat_std.mean():.4f}")

    return model, idx_to_label, feat_mean, feat_std


def create_hand_landmarker(model_path):
    """Crea el detector de manos de MediaPipe en modo VIDEO (hasta 2 manos)."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_pose_landmarker(model_path):
    """Crea el detector de pose de MediaPipe en modo VIDEO (1 persona)."""
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(hand_result, pose_result):
    """Construye el vector de 132 features de un frame (manos + hombros).

    Replica el mismo encoding usado en el entrenamiento: coordenadas relativas
    al centro de los hombros, layout mano izquierda (63) + derecha (63) +
    hombros (6). Devuelve None si no hay pose (frame descartado).
    """
    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        return None

    pose_lms = pose_result.pose_landmarks[0]
    left_sh = pose_lms[LEFT_SHOULDER_IDX]
    right_sh = pose_lms[RIGHT_SHOULDER_IDX]

    # Origen = centro de los hombros; hace la seña invariante a la posición.
    origin = np.array([
        (left_sh.x + right_sh.x) / 2,
        (left_sh.y + right_sh.y) / 2,
        (left_sh.z + right_sh.z) / 2,
    ])

    left_hand_lms = None
    right_hand_lms = None

    # "Left"/"Right" de MediaPipe están en vista de cámara (espejados) respecto
    # a las manos reales de la persona.
    if hand_result.hand_landmarks and hand_result.handedness:
        for i, handedness_list in enumerate(hand_result.handedness):
            label = handedness_list[0].category_name
            if label == "Left":
                right_hand_lms = hand_result.hand_landmarks[i]
            elif label == "Right":
                left_hand_lms = hand_result.hand_landmarks[i]

    def hand_to_array(lms, num_points):
        # Mano ausente -> ceros para conservar la dimensión del vector.
        if lms is None:
            return np.zeros(num_points * 3)
        coords = []
        for lm in lms:
            coords.extend([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])
        return np.array(coords)

    left_hand = hand_to_array(left_hand_lms, 21)
    right_hand = hand_to_array(right_hand_lms, 21)

    shoulders = np.array([
        left_sh.x - origin[0], left_sh.y - origin[1], left_sh.z - origin[2],
        right_sh.x - origin[0], right_sh.y - origin[1], right_sh.z - origin[2],
    ])

    return np.concatenate([left_hand, right_hand, shoulders])


def compute_hand_variance(buffer, window=15):
    """Variancia media del movimiento de manos en los últimos `window` frames.

    Se usa como detector de quietud: un valor bajo indica que las manos dejaron
    de moverse, señal de que la seña terminó. Solo considera los 126 features
    de manos (ignora hombros). Devuelve inf si aún no hay suficientes frames.
    """
    if len(buffer) < window:
        return float('inf')
    recent = np.array(list(buffer)[-window:])
    hands = recent[:, :126]
    return np.var(hands, axis=0).mean()


def classify_sequence(model, sequence, idx_to_label, feat_mean, feat_std):
    """Clasifica una secuencia de frames capturada con la SignCNN.

    Reinterpola la secuencia a SEQ_LENGTH, aplica la normalización z-score del
    entrenamiento y ejecuta la red.

    Returns:
        Tupla (label, confidence, class_idx, probs) con la seña predicha, su
        probabilidad softmax y el vector de probabilidades de todas las clases.
    """
    normalized = normalize_sequence(np.array(sequence), SEQ_LENGTH)

    if feat_mean is not None and feat_std is not None:
        normalized = (normalized - feat_mean) / feat_std

    # unsqueeze(0) añade la dimensión de batch; permute -> (batch, feat, seq).
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.permute(0, 2, 1)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    class_idx = probs.argmax().item()
    confidence = probs[class_idx].item()
    label = idx_to_label.get(class_idx, f"Clase_{class_idx}")

    return label, confidence, class_idx, probs.numpy()


def draw_status(frame, text, color, confidence=None, variance=None, fps=None, probs=None, idx_to_label=None):
    """Dibuja sobre el frame el estado del detector, las barras de probabilidad y métricas (HUD)."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 70), BLACK, -1)
    cv2.putText(frame, text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    if probs is not None and idx_to_label is not None:
        bar_x = w - 200
        bar_y_start = 85
        cv2.rectangle(frame, (bar_x - 10, bar_y_start - 20),
                       (w - 5, bar_y_start + len(probs) * 30 + 5), BLACK, -1)
        for i, prob in enumerate(probs):
            label = idx_to_label.get(i, f"C{i}")
            y = bar_y_start + i * 30
            bar_w = int(prob * 150)
            bar_color = GREEN if prob > CONFIDENCE_THRESHOLD else (100, 100, 100)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 18), bar_color, -1)
            cv2.putText(frame, f"{label}: {prob:.0%}", (bar_x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    info_parts = []
    if confidence is not None:
        info_parts.append(f"Conf: {confidence:.0%}")
    if variance is not None:
        info_parts.append(f"Var: {variance:.6f}")
    if fps is not None:
        info_parts.append(f"FPS: {fps:.0f}")

    if info_parts:
        info_text = " | ".join(info_parts)
        cv2.rectangle(frame, (0, h - 30), (w, h), BLACK, -1)
        cv2.putText(frame, info_text, (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


def main():
    """Bucle principal del detector: captura, detección de quietud y clasificación.

    Implementa una máquina de estados sobre la variancia de manos: acumula
    frames mientras hay movimiento y, cuando se detecta quietud sostenida (y no
    se está en cooldown), clasifica la seña, la muestra y reinicia el buffer.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Deteccion en tiempo real de senas LSC")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara")
    parser.add_argument("--threshold", type=float, default=VARIANCE_THRESHOLD,
                        help="Umbral de variancia para detectar quietud")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Umbral de confianza para clasificar una sena")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    model_path = os.path.join(models_dir, "signs_cnn.pth")
    if not os.path.isfile(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        print("  Ejecuta primero: python src/train_model.py")
        sys.exit(1)
    model, idx_to_label, feat_mean, feat_std = load_model(model_path)

    hand_landmarker = create_hand_landmarker(os.path.join(models_dir, "hand_landmarker.task"))
    pose_landmarker = create_pose_landmarker(os.path.join(models_dir, "pose_landmarker.task"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sign_names = list(idx_to_label.values())
    print(f"\n{'='*50}")
    print(f"DETECCION EN TIEMPO REAL - LSC")
    print(f"Senas: {', '.join(sign_names)}")
    print(f"{'='*50}")
    print(f"Umbral variancia: {args.threshold}")
    print(f"Umbral confianza: {args.confidence}")
    print("Presiona 'q' o ESC para salir")
    print("=" * 50)

    buffer = deque(maxlen=MAX_BUFFER_FRAMES)
    stillness_start = None
    last_detection_time = 0
    display_text = "Esperando sena..."
    display_color = WHITE
    display_confidence = None
    display_probs = None
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp_ms += 33

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = extract_landmarks(hand_result, pose_result)

        if landmarks is not None:
            buffer.append(landmarks)
            variance = compute_hand_variance(buffer)
            now = time.time()
            in_cooldown = (now - last_detection_time) < COOLDOWN_SECONDS

            if variance < args.threshold:
                if stillness_start is None:
                    stillness_start = now

                stillness_duration = now - stillness_start

                if stillness_duration >= STILLNESS_DURATION and not in_cooldown:
                    # Descarta los últimos 15 frames (la fase de quietud final)
                    # para clasificar solo la parte activa de la seña.
                    active_frames = list(buffer)[:-15]

                    if len(active_frames) >= MIN_ACTIVE_FRAMES:
                        label, prob, class_idx, probs = classify_sequence(
                            model, active_frames, idx_to_label,
                            feat_mean, feat_std
                        )

                        if prob >= args.confidence:
                            display_text = f"Sena: {label.upper()}"
                            display_color = GREEN
                        else:
                            display_text = "No reconocido"
                            display_color = RED

                        display_confidence = prob
                        display_probs = probs
                        last_detection_time = now
                        buffer.clear()
                        stillness_start = None
            else:
                stillness_start = None
                if not in_cooldown and len(buffer) > MIN_ACTIVE_FRAMES:
                    display_text = "Capturando sena..."
                    display_color = YELLOW
                    display_confidence = None
                    display_probs = None

            if in_cooldown and (now - last_detection_time) >= COOLDOWN_SECONDS:
                display_text = "Esperando sena..."
                display_color = WHITE
                display_confidence = None
                display_probs = None
        else:
            variance = float('inf')

        draw_status(
            frame, display_text, display_color,
            confidence=display_confidence,
            variance=variance if variance != float('inf') else None,
            fps=current_fps,
            probs=display_probs,
            idx_to_label=idx_to_label,
        )

        cv2.imshow("LSC - Detector de Senas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()
    print("\nSesion finalizada.")


if __name__ == "__main__":
    main()
