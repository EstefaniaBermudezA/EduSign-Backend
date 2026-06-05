"""Servidor WebSocket que transmite las senas reconocidas a Unreal Engine 5.

Corre el detector de senas en un hilo aparte y expone un WebSocket: cuando UE5
envia "start_capture" se activa la captura, y cada sena reconocida con
confianza suficiente se reenvia como JSON a los clientes conectados. La captura
se apaga al recibir "animation_started"/"stop_capture", evitando detecciones
mientras el avatar reproduce la respuesta. Se ejecuta con:
    python src/sign_server.py [--host ...] [--port 8765] [--camera 0]
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

# Estado compartido entre el hilo del detector y el loop asyncio del WebSocket.
detection_queue: "asyncio.Queue" = None   # senas detectadas pendientes de enviar
connected_clients = set()
stop_event = threading.Event()            # senaliza al worker que debe terminar

# Interruptor de captura: el worker solo procesa frames cuando esta activo;
# lo controlan los mensajes de UE5 (start/stop) desde el handler.
capture_enabled = threading.Event()


def detection_worker(args, loop):
    """Hilo que captura camara, detecta senas y las encola hacia el WebSocket.

    Reutiliza el pipeline de main.py (deteccion de quietud + clasificacion),
    pero solo procesa frames mientras `capture_enabled` esta activo. Cada sena
    reconocida con confianza suficiente se encola en `detection_queue` mediante
    run_coroutine_threadsafe para cruzar del hilo al loop asyncio.

    Args:
        args: Argumentos de CLI (camara, umbrales de variancia y confianza).
        loop: Event loop de asyncio donde vive la cola, para encolar de forma
            segura entre hilos.
    """
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

        # Captura apagada: descarta el buffer para no mezclar frames viejos
        # con la proxima sesion de captura.
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
    """Consume la cola de detecciones y las difunde a todos los clientes conectados."""
    while True:
        payload = await detection_queue.get()
        message = json.dumps(payload)
        if connected_clients:
            await asyncio.gather(
                *[ws.send(message) for ws in connected_clients],
                return_exceptions=True,
            )


async def handler(websocket):
    """Atiende a un cliente WebSocket y traduce sus mensajes en control de captura.

    Registra al cliente y procesa sus acciones: start_capture/ask activan la
    captura; animation_started/sign_recognized y stop_capture la apagan. Al
    desconectarse el ultimo cliente tambien apaga la captura.
    """
    print(f"[WS] Cliente conectado: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
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
        if not connected_clients and capture_enabled.is_set():
            capture_enabled.clear()
            print("[WS] Sin clientes conectados. Captura APAGADA.")
        print(f"[WS] Cliente desconectado: {websocket.remote_address}")


async def main_async(args):
    """Arranca el worker de deteccion, el broadcaster y el servidor WebSocket."""
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
