import pickle
import numpy as np
import os
# L
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")

model = pickle.load(open(os.path.join(MODEL_PATH, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(MODEL_PATH, "vectorizer.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb"))

print("==============================================")
print(" INTELLIGENT CAREER RECOMMENDATION SYSTEM ")
print("==============================================")

# User input
skills = input("Enter your skills (space separated): ")
interest = input("Enter your interest area: ")
personality = input("Enter your personality traits: ")

# Combine input
user_data = skills + " " + interest + " " + personality

# Transform input
input_vector = vectorizer.transform([user_data])

# Get probabilities
probabilities = model.predict_proba(input_vector)[0]

# Top 3 predictions
top3_indices = np.argsort(probabilities)[-3:][::-1]

print("\nTop 3 Career Recommendations:\n")

for index in top3_indices:
    career_name = label_encoder.inverse_transform([index])[0]
    confidence = probabilities[index] * 100
    print(f"{career_name} (Confidence: {confidence:.2f}%)")
