# run_translator.py
import os
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import pyttsx3

# Path to your trained model and label encoder
model_path = "models"

# Only the 10 labels you want to detect
labels = [
    "Hello", "ThankYou", "Bye", "GoodMorning", "GoodNight",
    "Yes", "No", "Sorry", "Love", "Happy"
]

# Initialize Mediapipe Hands for single-hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Extract normalized features from hand landmarks
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

def run_translator():
    # Load trained model
    try:
        clf = load(os.path.join(model_path, "gesture_model.joblib"))
        le = load(os.path.join(model_path, "label_encoder.joblib"))
    except:
        print("Model not found. Train the model first.")
        return

    engine = pyttsx3.init()
    last_pred = ""  # Track last prediction to avoid repeating speech

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

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        pred_text = ""
        feats = []

        # Detect hand and extract features
        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            feats.append(extract_features(hand_lms))

        if len(feats) > 0:
            combined_feat = np.array(feats).reshape(1, -1)  # shape = (1, 63)
            pred = clf.predict(combined_feat)
            pred_text = le.inverse_transform(pred)[0]

            # Only allow predictions from the 10 selected labels
            if pred_text not in labels:
                pred_text = ""

            # Speak only when prediction changes
            if pred_text != "" and pred_text != last_pred:
                engine.say(pred_text)
                engine.runAndWait()
                last_pred = pred_text

        # Display prediction on screen
        cv2.putText(frame, f"Prediction: {pred_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AI Sign Language Translator", frame)

        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_translator()
