import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model

# Define the dataset path
dataset_path = os.path.join(os.path.dirname(__file__), "hand_gesture_data.csv")

# Check if the file exists before proceeding
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: The file '{dataset_path}' was not found. Please check the path.")

# Load the data
data = pd.read_csv(dataset_path, header=None)

# Ensure the dataset is not empty
if data.empty:
    raise ValueError("Error: The dataset is empty. Please check the CSV file.")

# The first column is the label; the rest are features.
X = data.iloc[:, 1:].values  # Landmark data
y = data.iloc[:, 0].values   # Gesture labels

# Ensure the data is valid
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Error: The dataset does not contain enough data for training.")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model
model_path = os.path.join(os.path.dirname(__file__), "gesture_classifier.pkl")
joblib.dump(clf, model_path)
print(f"\nModel saved successfully at: {model_path}")
