import cv2
import numpy as np
from ultralytics import YOLO
import bettercam
import time
import torch

# Verify GPU setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load model
model = YOLO("valorant.pt")

# Move model to GPU with half-precision for better performance
if torch.cuda.is_available():
    model.to("cuda")
    # Enable half-precision if your GPU supports it (faster inference)
    model.half()  
    print("Using GPU for inference with half-precision.")
else:
    print("Using CPU for inference.")

# Setup capture
region_width, region_height = 320, 180
screen_w, screen_h = 1920, 1080
left = screen_w // 2 - region_width // 2
top = screen_h // 2 - region_height // 2

camera = bettercam.create(output_color='BGR')
prev_time = time.time()

while True:
    frame = camera.grab(region=(left, top, left + region_width, top + region_height))
    if frame is None:
        continue

    frame = np.array(frame)
    
    # Run model with GPU optimizations
    with torch.no_grad():  # Disable gradient calculation for inference
        results = model.predict(
            frame, 
            conf=0.4, 
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',  # Explicitly set device
            half=True if torch.cuda.is_available() else False  # Use half-precision if GPU
        )

    # Visualization (keep this on CPU)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label != "enemy_Head":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # FPS counter
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()