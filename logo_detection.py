import cv2
import os
import numpy as np

# Path to your known logos folder
LOGO_DIR = "known_logos"
logo_data = []

# Load all logos once at startup
for filename in os.listdir(LOGO_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(LOGO_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Failed to load logo: {filename}")
            continue
        logo_data.append((filename.split('.')[0], img))
        print(f"✅ Loaded logo: {filename}")


def detect_logos(frame, threshold=0.8):
    """
    Detect logos in the given frame using template matching.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, logo_img in logo_data:
        logo_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            h, w = logo_gray.shape
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
            cv2.putText(frame, f"{name}", (pt[0], pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame
