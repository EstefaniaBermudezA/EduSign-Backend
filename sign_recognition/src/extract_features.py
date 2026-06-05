"""Extrae los features de landmarks (manos + hombros) de videos de señas LSC.

Recorre los videos de una clase, ejecuta los detectores de MediaPipe (Hand
Landmarker y Pose Landmarker) frame por frame y guarda la secuencia de
landmarks de cada video como un archivo .npy que luego consume el
entrenamiento. Se ejecuta por clase:
    python src/extract_features.py --videos_dir videos/Quien --output_dir data/features/Quien
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
NUM_HAND_LANDMARKS = 21
FEATURES_PER_FRAME = 132


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


def extract_landmarks_from_results(hand_result, pose_result):
    """Construye el vector de features de un frame a partir de manos y pose.

    Toma el punto medio de los hombros como origen y expresa todas las
    coordenadas de forma relativa a él, de modo que la seña sea invariante a
    la posición del cuerpo dentro del encuadre. El layout de los 132 features
    es: mano izquierda (63), mano derecha (63) y hombros (6).

    Args:
        hand_result: Resultado del Hand Landmarker para el frame.
        pose_result: Resultado del Pose Landmarker para el frame.

    Returns:
        np.ndarray de 132 features, o None si no se detectó pose (sin pose no
        hay origen de referencia y el frame se descarta).
    """
    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        return None

    pose_lms = pose_result.pose_landmarks[0]
    left_sh = pose_lms[LEFT_SHOULDER_IDX]
    right_sh = pose_lms[RIGHT_SHOULDER_IDX]

    # Origen = centro de los hombros; ancla el sistema de coordenadas al torso.
    origin = np.array([
        (left_sh.x + right_sh.x) / 2,
        (left_sh.y + right_sh.y) / 2,
        (left_sh.z + right_sh.z) / 2,
    ])

    left_hand_lms = None
    right_hand_lms = None

    # MediaPipe etiqueta la lateralidad desde la vista de la cámara (espejada),
    # por eso "Left" del detector corresponde a la mano derecha de la persona.
    if hand_result.hand_landmarks and hand_result.handedness:
        for i, handedness_list in enumerate(hand_result.handedness):
            label = handedness_list[0].category_name
            if label == "Left":
                right_hand_lms = hand_result.hand_landmarks[i]
            elif label == "Right":
                left_hand_lms = hand_result.hand_landmarks[i]

    def hand_to_array(lms, num_points):
        # Mano no detectada -> vector de ceros para mantener dimensión constante.
        if lms is None:
            return np.zeros(num_points * 3)
        coords = []
        for lm in lms:
            coords.extend([lm.x - origin[0], lm.y - origin[1], lm.z - origin[2]])
        return np.array(coords)

    left_hand = hand_to_array(left_hand_lms, NUM_HAND_LANDMARKS)
    right_hand = hand_to_array(right_hand_lms, NUM_HAND_LANDMARKS)

    shoulders = np.array([
        left_sh.x - origin[0], left_sh.y - origin[1], left_sh.z - origin[2],
        right_sh.x - origin[0], right_sh.y - origin[1], right_sh.z - origin[2],
    ])

    return np.concatenate([left_hand, right_hand, shoulders])


def process_video(video_path, hand_landmarker, pose_landmarker):
    """Procesa un video completo y devuelve su secuencia de features.

    Returns:
        np.ndarray de forma (n_frames_con_landmarks, 132), o None si el video
        no se pudo abrir o no se detectaron landmarks en ningún frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f} | Frames totales: {total_frames}")

    all_frames = []
    frames_read = 0
    frames_with_landmarks = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = extract_landmarks_from_results(hand_result, pose_result)
        if landmarks is not None:
            all_frames.append(landmarks)
            frames_with_landmarks += 1

    cap.release()

    if not all_frames:
        print(f"  [WARN] No se detectaron landmarks en ningun frame")
        return None

    detection_rate = frames_with_landmarks / frames_read * 100
    print(f"  Frames leidos: {frames_read} | Con landmarks: {frames_with_landmarks} "
          f"({detection_rate:.1f}%)")

    return np.array(all_frames)


def main():
    """Procesa todos los videos de la carpeta de entrada y guarda un .npy por video."""
    parser = argparse.ArgumentParser(
        description="Extrae landmarks de manos y hombros de videos de senas (LSC)")
    parser.add_argument("--videos_dir", default="videos/Quien",
                        help="Carpeta con los videos .mp4")
    parser.add_argument("--output_dir", default="data/features/Quien",
                        help="Carpeta de salida para archivos .npy")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    videos_dir = os.path.join(project_root, args.videos_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    models_dir = os.path.join(project_root, "models")

    hand_model = os.path.join(models_dir, "hand_landmarker.task")
    pose_model = os.path.join(models_dir, "pose_landmarker.task")

    for model, name in [(hand_model, "hand_landmarker"), (pose_model, "pose_landmarker")]:
        if not os.path.isfile(model):
            print(f"[ERROR] Modelo no encontrado: {model}")
            print(f"  Descargalo de https://developers.google.com/mediapipe/solutions/vision/{name}")
            sys.exit(1)

    if not os.path.isdir(videos_dir):
        print(f"[ERROR] No se encontro el directorio de videos: {videos_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(videos_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    if not video_files:
        print(f"[ERROR] No se encontraron videos en: {videos_dir}")
        sys.exit(1)

    print(f"Videos encontrados: {len(video_files)}")
    print(f"Salida: {output_dir}")
    print("=" * 60)

    summary = []

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(videos_dir, video_file)
        name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_dir, f"{name}.npy")

        print(f"\n[{i}/{len(video_files)}] Procesando: {video_file}")

        hand_landmarker = create_hand_landmarker(hand_model)
        pose_landmarker = create_pose_landmarker(pose_model)

        try:
            features = process_video(video_path, hand_landmarker, pose_landmarker)

            if features is not None:
                np.save(output_path, features)
                print(f"  Guardado: {output_path}")
                print(f"  Shape: {features.shape}")
                summary.append((name, features.shape))
            else:
                summary.append((name, None))
        finally:
            hand_landmarker.close()
            pose_landmarker.close()

    print("\n" + "=" * 60)
    print("RESUMEN DE EXTRACCION")
    print("=" * 60)
    for name, shape in summary:
        if shape is not None:
            print(f"  {name}: {shape[0]} frames x {shape[1]} features")
        else:
            print(f"  {name}: FALLO")

    successful = sum(1 for _, s in summary if s is not None)
    print(f"\nExitosos: {successful}/{len(summary)}")


if __name__ == "__main__":
    main()
