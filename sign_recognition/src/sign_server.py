"""
Servidor WebSocket para integracion con Unreal Engine 5.

Corre el detector de senas LSC en un thread de fondo y empuja cada
deteccion a los clientes UE5 conectados, en formato JSON:

    {"label": "quien", "confidence": 0.87, "timestamp": 1712345678.9}

Modelo de captura controlada por UE5 (boton "Pregunta"):
  - Default: captura APAGADA. El worker no procesa frames.
  - Cuando el nino pulsa el boton, UE5 envia {"action":"start_capture"}.
    Python enciende la captura.
  - Python procesa frames y, cuando detecta una sena con suficiente
    confianza, la envia a UE5 (pero sigue capturando por si la deteccion
    no fue la que el nino queria).
  - Cuando UE5 reconoce la sena y arranca la animacion, envia
    {"action":"animation_started"}. Python apaga la captura
    automaticamente.
  - La captura solo vuelve a encenderse cuando el nino pulsa "Pregunta"
    de nuevo.

Caso de falso positivo (UE5 no tiene animacion para ese label):
  - UE5 no envia animation_started. Python mantiene la captura encendida
    y el nino puede intentar la sena de nuevo sin tener que volver a
    pulsar el boton.

Uso:
    pip install websockets
    python src/sign_server.py
    python src/sign_server.py --host 0.0.0.0 --port 8765 --camera 0

Desde UE5 conectas a: ws://127.0.0.1:8765
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import websockets
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import SEQ_LENGTH, NUM_FEATURES, normalize_sequence, SignCNN
from main import (
    VARIANCE_THRESHOLD, STILLNESS_DURATION, MIN_ACTIVE_FRAMES,
    MAX_BUFFER_FRAMES, CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS,
    load_model, create_hand_landmarker, create_pose_landmarker,
    extract_landmarks, compute_hand_variance, classify_sequence,
)

# Cola thread-safe de detecciones (worker -> event loop)
detection_queue: "asyncio.Queue" = None
connected_clients = set()
stop_event = threading.Event()

# Captura controlada por UE5. Default apagada hasta que el nino pulse
# el boton "Pregunta" y UE5 envie start_capture. Se apaga cuando UE5
# confirma que una sena fue reconocida (animation_started).
capture_enabled = threading.Event()


def detection_worker(args, loop):
    """Thread que corre el detector y encola cada seña reconocida."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    model_path = os.path.join(models_dir, "signs_cnn.pth")
    model, idx_to_label, feat_mean, feat_std = load_model(model_path)

    hand_landmarker = create_hand_landmarker(os.path.join(models_dir, "hand_landmarker.task"))
    pose_landmarker = create_pose_landmarker(os.path.join(models_dir, "pose_landmarker.task"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[Worker] Detector activo. Senas: {list(idx_to_label.values())}")
    print(f"[Worker] Captura inicialmente APAGADA. Esperando boton 'Pregunta' desde UE5...")

    buffer = deque(maxlen=MAX_BUFFER_FRAMES)
    stillness_start = None
    last_detection_time = 0
    timestamp_ms = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp_ms += 33

        # Si la captura no esta habilitada, drenamos el buffer para no
        # acumular movimiento parcial que se procesaria al reactivarse.
        if not capture_enabled.is_set():
            buffer.clear()
            stillness_start = None
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = extract_landmarks(hand_result, pose_result)
        if landmarks is None:
            continue

        buffer.append(landmarks)
        variance = compute_hand_variance(buffer)
        now = time.time()
        in_cooldown = (now - last_detection_time) < COOLDOWN_SECONDS

        if variance < args.threshold:
            if stillness_start is None:
                stillness_start = now
            if (now - stillness_start) >= STILLNESS_DURATION and not in_cooldown:
                active_frames = list(buffer)[:-15]
                if len(active_frames) >= MIN_ACTIVE_FRAMES:
                    label, prob, _, _ = classify_sequence(
                        model, active_frames, idx_to_label, feat_mean, feat_std
                    )
                    if prob >= args.confidence:
                        payload = {
                            "label": label,
                            "confidence": float(prob),
                            "timestamp": now,
                        }
                        print(f"[Worker] Detectado: {label} ({prob:.0%}). Enviado a UE5. Captura sigue activa hasta animation_started.")
                        asyncio.run_coroutine_threadsafe(
                            detection_queue.put(payload), loop
                        )
                    last_detection_time = now
                    buffer.clear()
                    stillness_start = None
        else:
            stillness_start = None

    cap.release()
    hand_landmarker.close()
    pose_landmarker.close()
    print("[Worker] Detenido.")


async def broadcaster():
    """Consume la cola y reenvia cada deteccion a todos los clientes UE5."""
    while True:
        payload = await detection_queue.get()
        message = json.dumps(payload)
        if connected_clients:
            await asyncio.gather(
                *[ws.send(message) for ws in connected_clients],
                return_exceptions=True,
            )


async def handler(websocket):
    print(f"[WS] Cliente conectado: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        # Mensajes entrantes desde UE5:
        #   start_capture     -> enciende la captura (nino pulso "Pregunta")
        #   animation_started -> apaga la captura (UE5 reconocio la sena)
        #   stop_capture      -> apaga la captura explicitamente (opcional)
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            action = data.get("action") or data.get("event")

            if action in ("start_capture", "ask"):
                if not capture_enabled.is_set():
                    capture_enabled.set()
                    print("[WS] start_capture recibido. Captura ACTIVA.")
                else:
                    print("[WS] start_capture recibido pero captura ya estaba activa.")

            elif action in ("animation_started", "sign_recognized"):
                if capture_enabled.is_set():
                    capture_enabled.clear()
                    print("[WS] animation_started recibido. Captura APAGADA.")

            elif action in ("stop_capture",):
                if capture_enabled.is_set():
                    capture_enabled.clear()
                    print("[WS] stop_capture recibido. Captura APAGADA.")
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        # Si no quedan clientes, apagamos la captura por seguridad para
        # que no quede procesando frames sin nadie escuchando.
        if not connected_clients and capture_enabled.is_set():
            capture_enabled.clear()
            print("[WS] Sin clientes conectados. Captura APAGADA.")
        print(f"[WS] Cliente desconectado: {websocket.remote_address}")


async def main_async(args):
    global detection_queue
    detection_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    worker = threading.Thread(target=detection_worker, args=(args, loop), daemon=True)
    worker.start()

    asyncio.create_task(broadcaster())

    print(f"[WS] Servidor escuchando en ws://{args.host}:{args.port}")
    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=VARIANCE_THRESHOLD)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        stop_event.set()
        print("\nServidor detenido.")
