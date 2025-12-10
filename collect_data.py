# collect_data.py
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ----------------------------
# CONFIGURATION
# ----------------------------
labels = [
    "Hello", "ThankYou", "Bye", "GoodMorning", "GoodNight",
    "Yes", "No", "Sorry", "Love", "Happy"  # Only 10 labels
]
data_path = "data"
os.makedirs(data_path, exist_ok=True)

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    arr = np.array(coords, dtype=np.float32).reshape(21, 3)
    wrist = arr[0]
    arr = arr - wrist
    arr = arr.flatten()
    arr = arr / (np.max(np.abs(arr)) + 1e-6)
    return arr

# ----------------------------
# COLLECT DATA
# ----------------------------
def collect_data():
    # Open webcam
    for i in range(0, 3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Using camera index {i}")
            break
        cap.release()
    else:
        print("Cannot access webcam.")
        return

    for label in labels:
        print(f"Collecting data for: {label}")
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            feats = []
            if result.multi_hand_landmarks:
                # Only take the first detected hand
                hand_lms = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                feats.append(extract_features(hand_lms))

            if len(feats) > 0:
                combined_feat = np.array(feats).reshape(1, -1)
            
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Collect Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if len(feats) > 0:
                    frames.append(combined_feat.flatten())
                    print(f"Captured frame: {len(frames)}")
            elif key == ord('q'):
                break

        # Save data for this label
        df = pd.DataFrame(frames)
        df.to_csv(os.path.join(data_path, f"{label}.csv"), index=False)
        print(f"Saved {len(frames)} frames for {label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
