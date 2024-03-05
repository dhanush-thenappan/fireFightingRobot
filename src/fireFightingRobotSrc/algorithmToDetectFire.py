import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
# Function to extract features from images
def extract_features(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Thresholding to isolate regions with high intensity and low saturation
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Calculate the percentage of pixels that satisfy the thresholding
    fire_percentage = np.sum(mask == 255) / (image.shape[0] * image.shape[1])
    
    return fire_percentage

# Load dataset
# Assuming you have two directories: 'fire' containing images with fire and 'no_fire' containing images without fire
# Load dataset
def load_dataset():
    X, y = [], []
    for label, folder in enumerate(['Dataset/fire', 'Dataset/no_fire']):
        for filename in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, filename))
            if image is not None:
                # Extract features from the image
                features = extract_features(image)
                X.append([features])  # Reshape to 2D array
                y.append(label)
    return np.array(X), np.array(y)

# Load dataset
X, y = load_dataset()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

with open('svm_classifier.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)