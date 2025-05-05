import cv2
import numpy as np
from ultralytics import YOLO
import mss


model = YOLO("valorant.pt")

class_names = ['enemyBody', 'enemyHead']


screen_width, screen_height = 1920, 1080

box_width, box_height = 640, 360
center_x, center_y = screen_width // 2, screen_height // 2
left = center_x - box_width // 2
top = center_y - box_height // 2

monitor = {
    "top": top,
    "left": left,
    "width": box_width,
    "height": box_height
}


sct = mss.mss()

while True:

    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    results = model.predict(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 0, 255) if label == 'enemyHead' else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

   
    cv2.imshow("Centered Detection Box", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
