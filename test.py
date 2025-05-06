from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import win32api, win32con
import math
import torch
import threading
from mss import mss

# Load model
model = YOLO("valorant.pt")

# Move model to GPU if available
if torch.cuda.is_available():
    model.to('cuda')
    model.half()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Define enemy classes
class_names = ['enemy', 'enemy_head']

# Shoot function
def keyshoot():
    win32api.keybd_event(0x4B, 0, 0, 0)  # Press K
    win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)  # Release K

# Cooldown function for toggle key
def cooldown(toggle_state, wait):
    time.sleep(wait)
    toggle_state[0] = True

# Screen capture region settings
MONITOR_WIDTH, MONITOR_HEIGHT = 1920, 1080  # Adjust if your resolution is different
MONITOR_SCALE = 5  # How much to downscale
region = (
    int(MONITOR_WIDTH/2 - MONITOR_WIDTH/MONITOR_SCALE/2),
    int(MONITOR_HEIGHT/2 - MONITOR_HEIGHT/MONITOR_SCALE/2),
    int(MONITOR_WIDTH/2 + MONITOR_WIDTH/MONITOR_SCALE/2),
    int(MONITOR_HEIGHT/2 + MONITOR_HEIGHT/MONITOR_SCALE/2)
)
x, y, width, height = region
screenshot_center = [int((width - x) / 2), int((height - y) / 2)]

# Initialize triggerbot state
triggerbot = False
triggerbot_toggle = [True]  # Cooldown control list

# FPS counter setup
start_time = time.time()
frame_counter = 0

print("Press 'C' to Toggle TriggerBot, Press 'P' to Stop.")

with mss() as sct:
    while not keyboard.is_pressed('p'):
        # Capture screenshot
        screenshot = np.array(sct.grab(region))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # Convert 4 channel to 3 channel (fix)

        # Predict using the model
        with torch.no_grad():
            results = model.predict(
                source=screenshot,
                conf=0.5,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                half=True if torch.cuda.is_available() else False
            )


        frame_counter += 1
        if time.time() - start_time > 1:
            fps = frame_counter / (time.time() - start_time)
            print(f"FPS: {int(fps)}")
            frame_counter = 0
            start_time = time.time()

        # Handle toggle with cooldown
        if keyboard.is_pressed('c'):
            if triggerbot_toggle[0]:
                triggerbot = not triggerbot
                print(f"TriggerBot {'Enabled' if triggerbot else 'Disabled'}")
                triggerbot_toggle[0] = False
                threading.Thread(target=cooldown, args=(triggerbot_toggle, 0.2)).start()

        # Find the closest enemy head
        closest_distance = float('inf')
        closest_box = None

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "enemy_head":
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    distance = math.hypot(box_center_x - screenshot_center[0], box_center_y - screenshot_center[1])

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_box = (x1, y1, x2, y2)

        # Shoot if closest box is centered
        if triggerbot and closest_box is not None:
            x1, y1, x2, y2 = closest_box
            if (screenshot_center[0] >= x1 and screenshot_center[0] <= x2) and (screenshot_center[1] >= y1 and screenshot_center[1] <= y2):
                print("Enemy head detected, shooting!")
                keyshoot()
                time.sleep(0.2)  # Prevent double-shot

print("Stopped")
