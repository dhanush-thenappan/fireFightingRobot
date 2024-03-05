import cv2
import numpy as np
from sklearn.svm import SVC
import os
import pickle

# Function to extract features from an image
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

# Load trained SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier_path = "svm_classifier.pkl"  # Change this to the path of your trained SVM classifier
if os.path.exists(svm_classifier_path):
    with open(svm_classifier_path, 'rb') as f:
        svm_classifier = pickle.load(f)
else:
    print("Trained SVM classifier not found. Please train a classifier first.")
    exit()

# Function to detect fire using the camera
def detect_fire():
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Preprocess image
        processed_image = cv2.resize(frame, (224, 224))  # Resize image to match input size of the model
        
        # Extract features
        features = extract_features(processed_image)
        
        # Make prediction using the trained model
        prediction = svm_classifier.predict([[features]])
        
        # Print result
        if prediction == 0:
            print("Fire detected!")
        else:
            print("No fire detected.")
        
        # Display the resulting frame
        cv2.imshow('Fire Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start fire detection using the camera
detect_fire()
