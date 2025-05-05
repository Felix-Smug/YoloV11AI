import cv2
from ultralytics import YOLO

# Load the YOLOv11 model (from your trained model)
model = YOLO("valorant.pt")  # Make sure valorant.pt is in the same folder

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Define class names (from your data.yaml)
class_names = ['enemyBody', 'enemyHead']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, conf=0.4, verbose=False)

    # Draw results
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Choose color
            color = (0, 0, 255) if label == 'enemyHead' else (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("YOLOv11 - Live Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
