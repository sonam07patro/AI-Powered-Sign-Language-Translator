# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

data_path = "data"
model_path = "models"
os.makedirs(model_path, exist_ok=True)

def train_model():
    X = []
    y = []

    # Load all CSV files in data folder
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            label = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_path, file))

            # Filter only rows with exactly 63 features (single hand)
            for idx, row in df.iterrows():
                row_values = row.values
                if len(row_values) == 63:
                    X.append(row_values)
                    y.append(label)
                else:
                    print(f"Skipping row {idx} in {file} (size {len(row_values)})")

    if len(X) == 0:
        print("No valid single-hand data to train. Re-collect data using single hand only.")
        return

    X = np.vstack(X)
    y = np.array(y)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_enc)

    # Save model and label encoder
    dump(clf, os.path.join(model_path, "gesture_model.joblib"))
    dump(le, os.path.join(model_path, "label_encoder.joblib"))

    print(f"Training complete. Model saved in '{model_path}'.")

if __name__ == "__main__":
    train_model()
