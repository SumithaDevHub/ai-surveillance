import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Directories
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_PATH = "models/face/encodings.pkl"

# Load or generate encodings
def generate_or_load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        print("✅ Loaded existing encodings.")
        return data["encodings"], data["names"]

    print("🔍 Generating new encodings...")
    known_encodings, known_names = [], []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, filename)
            try:
                rep = DeepFace.represent(img_path=path, model_name="Facenet512", enforce_detection=False)
                if rep:
                    known_encodings.append(rep[0]["embedding"])
                    known_names.append(person_name)
                    print(f"✅ Encoded {person_name} ({filename})")
            except Exception as e:
                print(f"⚠️ Failed for {filename}: {e}")

    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print("🎉 Saved encodings to", ENCODINGS_PATH)
    return known_encodings, known_names


def recognize_face(frame, known_encodings, known_names):
    try:
        rep = DeepFace.represent(frame, model_name="Facenet512", enforce_detection=False)
        if not rep:
            return "Unknown", None

        face_embedding = rep[0]["embedding"]

        # Compare with known faces
        distances = [np.linalg.norm(np.array(enc) - np.array(face_embedding)) for enc in known_encodings]
        if len(distances) == 0:
            return "Unknown", None

        best_match_idx = np.argmin(distances)
        if distances[best_match_idx] < 0.8:
            return known_names[best_match_idx], distances[best_match_idx]
        else:
            return "Unknown", distances[best_match_idx]
    except Exception as e:
        return "Error", str(e)
