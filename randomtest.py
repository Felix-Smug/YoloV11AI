import threading
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

# Load YOLO model
model = YOLO("ow2.engine")

if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

target_label = "EnemyHead"
aim_assist_enabled = [True]
triggerbot_enabled = True
aim_region_size = 150
aim_smoothness = 1.5
aim_max_move = 10
trigger_radius = 9
click_cooldown_time = 0.15
frame_count = 0
start_time = time.time()

def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

def smooth_move(dx, dy, scale=1.5, max_move=15):
    move_x = int(max(-max_move, min(max_move, dx * scale)))
    move_y = int(max(-max_move, min(max_move, dy * scale)))
    win32api.mouse_event(0x0001, move_x, move_y, 0, 0)

click_ready = [True]
def click_cooldown():
    click_ready[0] = False
    time.sleep(click_cooldown_time)
    click_ready[0] = True

# --- Setup ---
screen_w, screen_h = pyautogui.size()
center_x, center_y = screen_w // 2, screen_h // 2
camera = bettercam.create(output_color='BGR')
print("Press 'C' to toggle Aim Assist ON/OFF")
print("Press 'P' to stop script")

#main
last_toggle_time = 0
while not keyboard.is_pressed('p'):

    
    if keyboard.is_pressed('c') and time.time() - last_toggle_time > 0.5:
        aim_assist_enabled[0] = not aim_assist_enabled[0]
        print(f"Aim Assist {'ENABLED' if aim_assist_enabled[0] else 'DISABLED'}")
        last_toggle_time = time.time()

    
    frame = camera.grab()
    if frame is None:
        continue

    frame_np = np.array(frame)
    results = model.predict(source=frame_np, conf=0.1, verbose=False, device=0)

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == target_label:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                dx = box_center_x - center_x
                dy = box_center_y - center_y

                if aim_assist_enabled[0]:
                    smooth_move(dx, dy, scale=aim_smoothness, max_move=aim_max_move)
 
                # Fire if target is centered
                if math.hypot(dx, dy) < trigger_radius and click_ready[0]:
                    print("EnemyHead Detected — Shooting")
                    click()
                    threading.Thread(target=click_cooldown, daemon=True).start()
                break  # Only shoot 1 target per frame
    
    
    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

print("STOP DA PROGRAM")
