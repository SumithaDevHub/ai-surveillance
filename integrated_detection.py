import cv2
from logo_detection import detect_logos
from face_recognition import detect_faces
# from threat_detection import detect_threats   # Uncomment when threat model ready

def process_frame(frame):
    # 1️⃣ Face recognition
    frame = detect_faces(frame)

    # 2️⃣ Logo detection
    frame = detect_logos(frame)

    # 3️⃣ Threat detection (future)
    # frame = detect_threats(frame)

    return frame
