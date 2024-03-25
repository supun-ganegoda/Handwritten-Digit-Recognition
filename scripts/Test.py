# Import required libraries
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

# Define the parameters
windowWidth = 640
windowHeight = 480
imageSize = (320,320,3)
threshold = 0.75

# Open the web camera
cap = cv2.VideoCapture(0)
cap.set(3,windowWidth)
cap.set(4, windowHeight)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Load trained model
with open('../models/model_trained.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)
# Load model weights
loaded_model.load_weights("../models/model.weights.h5")

# Pre-process the captured images
def preProcessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img  = img/ 255
    return img 

# Prediction
def prediction(image, model):
    img = np.asarray(image)
    img = cv2.resize(image, (32, 32))
    img = cv2.equalizeHist(img)
    img = img / 255
    img = img.reshape(1, 32, 32, 1)
    predict = model.predict(img)
    prob = np.amax(predict)
    class_index = np.argmax(predict, axis=1)
    result = class_index[0]
    if prob < threshold:
        result = 0
        prob = 0
    return result, prob

# Reading webcam feed
while True:
    success, imgOriginal = cap.read()
    imgOriginalCopy = imgOriginal.copy()

    if not success:
        print("Failed to read frame from camera")
        break
    if isinstance(imgOriginal, np.ndarray):
        bbox_size = (200,200)
        bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]
        img_cropped = imgOriginal[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (200, 200))
        cv2.imshow("cropped", img_gray)

        result, probability = prediction(img_gray, loaded_model)
        cv2.putText(imgOriginalCopy, f"Prediction: {result}", (40,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginalCopy, "Probability: "+"{0:.2f}".format(probability), (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2, cv2.LINE_AA)
        
        if probability > threshold:
            recColor = (0,255,0)
        else:
            recColor = (0,0,255)
        
        cv2.rectangle(imgOriginalCopy, bbox[0], bbox[1], recColor, 3)

    else:
        print("Invalid image format")

    cv2.imshow('Input', imgOriginalCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()