# app.py
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import pyttsx3

# ----------------------------
# CONFIGURATION
# ----------------------------
labels = [
    "Hello", "ThankYou", "Bye", "GoodMorning", "GoodNight", "Welcome",
    "Yes", "No", "Sorry", "Love", "Happy", "Sad",
    "HowAreYou", "What", "Who", "Stop", "Go", "Eat", "Drink", "Please",
    "One", "Two", "Three", "Four", "Five"
]
data_path = "data"
model_path = "models"

os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    arr = np.array(coords, dtype=np.float32)
    arr = arr.reshape(21, 3)
    wrist = arr[0]
    arr = arr - wrist
    arr = arr.flatten()
    arr = arr / (np.max(np.abs(arr)) + 1e-6)
    return arr

# ----------------------------
# COLLECT DATA
# ----------------------------
def collect_data():
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
                for hand_lms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    feats.append(extract_features(hand_lms))

            # Pad missing hands with zeros
            while len(feats) < 2:
                feats.append(np.zeros(63))

            combined_feat = np.concatenate(feats)

            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Collect Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frames.append(combined_feat)
                print(f"Captured frame: {len(frames)}")
            elif key == ord('q'):
                break

        df = pd.DataFrame(frames)
        df.to_csv(os.path.join(data_path, f"{label}.csv"), index=False)
        print(f"Saved {len(frames)} frames for {label}")

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model():
    X = []
    y = []

    for label in labels:
        file_path = os.path.join(data_path, f"{label}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            X.append(df.values)
            y += [label] * len(df)

    if len(X) == 0:
        print("No data found. Please collect data first.")
        return

    X = np.vstack(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y_encoded)

    dump(clf, os.path.join(model_path, "gesture_model.joblib"))
    dump(le, os.path.join(model_path, "label_encoder.joblib"))

    acc = clf.score(X, y_encoded)
    print(f"Training completed. Accuracy: {acc*100:.2f}%")

# ----------------------------
# RUN TRANSLATOR
# ----------------------------
def run_translator():
    try:
        clf = load(os.path.join(model_path, "gesture_model.joblib"))
        le = load(os.path.join(model_path, "label_encoder.joblib"))
    except:
        print("Model not found. Train the model first.")
        return

    engine = pyttsx3.init()

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
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                feats.append(extract_features(hand_lms))

        # Pad missing hands with zeros
        while len(feats) < 2:
            feats.append(np.zeros(63))

        if len(feats) > 0:
            combined_feat = np.concatenate(feats).reshape(1, -1)
            pred = clf.predict(combined_feat)
            pred_text = le.inverse_transform(pred)[0]

        cv2.putText(frame, f"Prediction: {pred_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Translator", frame)

        if pred_text != "":
            engine.say(pred_text)
            engine.runAndWait()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# MAIN MENU
# ----------------------------
if __name__ == "__main__":
    while True:
        print("\n===== AI-Powered Sign Language Translator =====")
        print("1. Collect Data")
        print("2. Train Model")
        print("3. Run Translator")
        print("4. Exit")
        choice = input("Enter option: ")

        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_translator()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")
