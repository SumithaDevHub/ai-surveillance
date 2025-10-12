# threat_detection.py
import cv2
import os
from ultralytics import YOLO

# Path to custom threat model
CUSTOM_MODEL_PATH = "models/threat/threat_best.pt"

# Check if custom model exists
if os.path.exists(CUSTOM_MODEL_PATH):
    print(f"✅ Loading custom threat model: {CUSTOM_MODEL_PATH}")
    model = YOLO(CUSTOM_MODEL_PATH)
else:
    print("⚠️ Custom model not found. Using pre-trained YOLOv8n for demo")
    model = YOLO("yolov8n.pt")  # small pre-trained YOLO model

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open webcam")

print("🎥 Threat Detection running... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO expects BGR image
    results = model(frame, verbose=False)[0]  # single frame

    for box in results.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Draw rectangle and label
        color = (0, 0, 255)  # red for threats
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Threat Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
