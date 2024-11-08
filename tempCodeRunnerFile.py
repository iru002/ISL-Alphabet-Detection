import cv2
import numpy as np
import os
from keras.models import load_model

# Load the trained model
model = load_model('isl_model.keras')

# Define the labels for the classes
class_labels = sorted(os.listdir('isl/Indian'))

# Function to predict the gesture
def predict_gesture(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the region of interest (ROI) for gesture recognition
    x, y, w, h = 100, 100, 300, 300
    roi = gray_frame[y:y+h, x:x+w]
    
    # Predict the gesture in the ROI
    predicted_label = predict_gesture(roi)
    
    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the predicted label in the box layout
    cv2.putText(frame, f'Predicted: {predicted_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Gesture Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
