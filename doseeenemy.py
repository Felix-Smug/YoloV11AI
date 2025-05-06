from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import pyautogui
import win32api, win32con
import math
import torch
import bettercam

# Load TensorRT-exported model
model = YOLO("valorant.engine")

# Just for user feedback
if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available. TensorRT will not work properly.")

class_names = ['enemy', 'enemy_Head']

def keyshoot():
    # You can still use the 'K' key if your Valorant is bound to it
    win32api.keybd_event(0x4B, 0, 0, 0)
    win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)

camera = bettercam.create(output_color='BGR')

# Define screen region
screen_w, screen_h = pyautogui.size()
region_size = 150
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'P' to Stop TriggerBot")

# FPS tracker
frame_count = 0
start_time = time.time()

while not keyboard.is_pressed('p'):
    frame = camera.grab(region=(region_left, region_top, region_left + region_size, region_top + region_size))
    if frame is None:
        continue

    frame_np = np.array(frame)

    # Predict using TensorRT engine
    results = model.predict(
        source=frame_np,
        conf=0.5,
        verbose=False,
        device=0  # CUDA
    )

    # Draw detections
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Draw bounding box
            color = (0, 0, 255) if label == "enemy_Head" else (0, 255, 0)
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Triggerbot logic
            if label == "enemy_Head":
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = math.hypot(box_center_x - region_size / 2, box_center_y - region_size / 2)

                if distance < 30:  # Increased for tolerance
                    print("enemyHead detected → Shooting")
                    keyshoot()
                    time.sleep(0.2)
                    break

    # Show what AI sees
    cv2.imshow("TriggerBot Vision", frame_np)
    if cv2.waitKey(1) == ord('p'):
        break

    # FPS counter
    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

cv2.destroyAllWindows()
print("Stopped")
