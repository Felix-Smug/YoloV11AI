from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import pyautogui
import time
import win32api, win32con
import math
import random
import threading
import bettercam


model = YOLO("valorant.pt")
class_names = ['enemyBody', 'enemyHead']

#set shoot button to K
def keyshoot():
    win32api.keybd_event(0x4B, 0, 0, 0)                   
    win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)

#initialize camera
camera = bettercam.create(output_color='BGR')

# region for BetterCam
screen_w, screen_h = pyautogui.size()

# trigger bot region 
region_size = 150  
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'P' to Stop TriggerBot")
while not keyboard.is_pressed('p'):

    frame = camera.grab(region=(region_left, region_top, region_left + region_size, region_top + region_size))

    if frame is None:
        continue

    frame_np = np.array(frame)

    #yolov11 detection
    results = model.predict(source=frame_np, conf=0.5, verbose=False)

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "enemyHead":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # check if the center of the box is near crosshair
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = math.hypot(box_center_x - region_size / 2, box_center_y - region_size / 2)

                if distance < 20:  # i'll change the distance if doesn't work
                    print(f"enemyHead detected")
                    keyshoot()
                    #delay for no double shots
                    time.sleep(0.2) 
                    break

print("Stopped")