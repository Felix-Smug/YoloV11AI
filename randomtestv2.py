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

# --- CONFIG ---
aim_color = (255, 0, 0)  # Set your target color here (R, G, B)
color_radius = 10        # Radius for triggerbot
color_threshold = 65     # Tolerance for RGB difference
click_cooldown_time = 0.15
aim_region_size = 150
aim_smoothness = 1.5
aim_max_move = 10
trigger_radius = 9

# --- YOLO SETUP ---
model = YOLO("ow2.engine")

if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

# --- GLOBAL FLAGS ---
aim_assist_enabled = [True]
click_ready = [True]

# --- Screen Setup ---
screen_w, screen_h = pyautogui.size()
center_x, center_y = screen_w // 2, screen_h // 2
camera = bettercam.create(output_color='BGR')

# --- Mouse Utility ---
def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

def smooth_move(dx, dy, scale=1.5, max_move=15):
    move_x = int(max(-max_move, min(max_move, dx * scale)))
    move_y = int(max(-max_move, min(max_move, dy * scale)))
    win32api.mouse_event(0x0001, move_x, move_y, 0, 0)

def click_cooldown():
    click_ready[0] = False
    time.sleep(click_cooldown_time)
    click_ready[0] = True

# --- Color TriggerBot Thread ---
def color_triggerbot():
    print("Color TriggerBot Started")
    region = (
        center_x - color_radius,
        center_y - color_radius,
        center_x + color_radius,
        center_y + color_radius,
    )
    while not keyboard.is_pressed('p'):
        try:
            frame = camera.grab(region=region)
            if frame is None:
                continue

            frame_np = np.array(frame).astype(np.int32)
            for x in range(0, color_radius * 2):
                for y in range(0, color_radius * 2):
                    pixel = frame_np[y, x]
                    if all(abs(pixel[i] - aim_color[i]) < color_threshold for i in range(3)):
                        if click_ready[0]:
                            print("Color match — Triggerbot fired")
                            click()
                            threading.Thread(target=click_cooldown, daemon=True).start()
                        raise StopIteration
        except StopIteration:
            continue
        except Exception as e:
            print(f"[TriggerBot Error]: {e}")
    print("Triggerbot Stopped")

# --- YOLO Main Loop ---
def yolo_aim_assist():
    frame_count = 0
    start_time = time.time()
    target_label = "EnemyHead"
    last_toggle_time = 0

    # Ensure region is always initialized
    region_left = max(0, center_x - aim_region_size // 2)
    region_top = max(0, center_y - aim_region_size // 2)
    region_right = min(screen_w, region_left + aim_region_size)
    region_bottom = min(screen_h, region_top + aim_region_size)
    region = (region_left, region_top, region_right, region_bottom)

    while not keyboard.is_pressed('p'):
        # Toggle Aim Assist
        if keyboard.is_pressed('c') and time.time() - last_toggle_time > 0.5:
            aim_assist_enabled[0] = not aim_assist_enabled[0]
            print(f"Aim Assist {'ENABLED' if aim_assist_enabled[0] else 'DISABLED'}")
            last_toggle_time = time.time()

        # --- Try grabbing frame ---
        try:
            frame = camera.grab(region=region)
        except Exception as e:
            print(f"[BetterCam Grab Error]: {e}")
            continue

        if frame is None:
            continue

        frame_np = np.array(frame)
        results = model.predict(source=frame_np, conf=0.2, verbose=False, device=0)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == target_label:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2

                    dx = box_center_x - aim_region_size / 2
                    dy = box_center_y - aim_region_size / 2

                    if aim_assist_enabled[0]:
                        smooth_move(dx, dy, scale=aim_smoothness, max_move=aim_max_move)

                    break  # Only track 1 target per frame

        # FPS Counter
        frame_count += 1
        if frame_count >= 60:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

    print("Aim Assist Stopped")


# --- MAIN ---
if __name__ == "__main__":
    print("Press 'C' to toggle Aim Assist. Press 'P' to quit.")
    threading.Thread(target=color_triggerbot, daemon=True).start()
    yolo_aim_assist()
