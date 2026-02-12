import os
import pandas as pd
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# os.makedirs(MODEL_PATH, exist_ok=True)

# # Paths
# DATA_PATH = "../data/dataset.csv"
# MODEL_PATH = "../models/"

# Create models directory if not exists
os.makedirs(MODEL_PATH, exist_ok=True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)
# Load dataset
data = pd.read_csv(DATA_PATH)

# Combine features
data["Combined"] = data["Skills"] + " " + data["Interest"] + " " + data["Personality"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Combined"])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Career"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed")
print("Model Accuracy:", accuracy)

# Save model files

pickle.dump(model, open(os.path.join(MODEL_PATH, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_PATH, "vectorizer.pkl"), "wb"))
pickle.dump(label_encoder, open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "wb"))
print("Model saved inside models/ folder")


