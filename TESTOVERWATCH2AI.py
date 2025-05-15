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

model = YOLO("ow2.engine")

if torch.cuda.is_available():
    print(f"Using GPU with TensorRT: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

class_names = ['EnemyHead']


def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)


# def keyshoot():
#     keyboard.press('k')

camera = bettercam.create(output_color='BGR')

screen_w, screen_h = pyautogui.size()
region_size = 150
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'C' to toggle TriggerBot ON/OFF")
print("Press 'P' to Stop TriggerBot")

# toggle
triggerbot_enabled = True
last_toggle_time = 0

frame_count = 0
start_time = time.time()

while not keyboard.is_pressed('p'):
    
    if keyboard.is_pressed('c') and time.time() - last_toggle_time > 0.5:
        triggerbot_enabled = not triggerbot_enabled
        status = "ENABLED" if triggerbot_enabled else "DISABLED"
        print(f"TriggerBot {status}")
        last_toggle_time = time.time()

    frame = camera.grab(region=(region_left, region_top, region_left + region_size, region_top + region_size))
    if frame is None:
        continue

    frame_np = np.array(frame)

    
    results = model.predict(
        source=frame_np,
        conf=0.2,
        verbose=False,
        device=0
    )


    if triggerbot_enabled and results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "EnemyHead":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = math.hypot(box_center_x - region_size / 2, box_center_y - region_size / 2)

                if distance < 9:
                    print("EnemyHead detected")
                    click()
                    time.sleep(0.2)
                    break

    frame_count += 1
    if frame_count >= 60:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

print("Stopped")
