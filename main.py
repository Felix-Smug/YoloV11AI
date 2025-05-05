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

