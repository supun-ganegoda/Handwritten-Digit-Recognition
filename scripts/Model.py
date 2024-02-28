# import libraries
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

# model settings
dataPath = '../data'
testDataRatio = 0.2
validationDataRatio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000

# import data
imageList = []
classList = []
rawData = os.listdir(dataPath)
noOfClasses = len(rawData)

print('\n\nImporting data...')
for path in range(0, noOfClasses):
    filePath = os.listdir(dataPath + '/' + str(path))
    for img in tqdm(filePath, desc=f"Processing Class {path}"):
        curImage = cv2.imread(dataPath+'/'+str(path)+'/'+img)
        curImage = cv2.resize(curImage, (imageDimensions[0], imageDimensions[1]))
        imageList.append(curImage)
        classList.append(path)

print("Data import finished...")
print("Total images imported: ",len(imageList))
print("Total image classes: ", len(classList))

imageList = np.array(imageList)
classList = np.array(classList)

print('Shape of the images: ',imageList.shape)
print('Shape of the classes',classList.shape)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(imageList, classList, test_size=testDataRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=validationDataRatio)

print("Shape of training set: ",X_train.shape)
print("Shape of test set: ",X_test.shape)
print("Shape of validation set: ",X_validation.shape,'\n')

# Plotting the class distribution
numOfSamples = []
for classNo in range(0,noOfClasses):
    numOfSamples.append(len(np.where(y_train==classNo)[0]))

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), numOfSamples)
plt.title("Sample Distribution")
plt.xlabel("Class no")
plt.ylabel("Sample count")
plt.show()

# Image pre-processing
def preProcessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img  = img/ 255
    return img 

X_train = np.array(list(map(preProcessImage,X_train)))
X_test = np.array(list(map(preProcessImage,X_test)))
X_validation = np.array(list(map(preProcessImage,X_validation)))

print('After pre-processing...\n')
print("Shape of training set: ",X_train.shape)
print("Shape of test set: ",X_test.shape)
print("Shape of validation set: ",X_validation.shape)

# Reshape pre-processed data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2],1)

# Create augmented image set
augmentedData = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   rotation_range=10)
augmentedData.fit(X_train)

# One hot encoding
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# Create neural network
def createModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape = (imageDimensions[0],imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

neuralNetwork = createModel()
print('\n\n-------------------------\n')
print("Model compilation done !\n\n")
print(neuralNetwork.summary())

# Train the model
generator = augmentedData.flow(X_train, y_train, batch_size=batchSizeVal)
history = neuralNetwork.fit(generator,
                            steps_per_epoch=stepsPerEpoch,
                            epochs=epochsVal,
                            validation_data=(X_validation, y_validation),
                            shuffle=1)

# Plotting model performance
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Display the score
print("-----------------------------\n\n")
score = neuralNetwork.evaluate(X_test,y_test,verbose=0)
print('Test score = ', score[0])
print('Test accuracy = ', score[1])

# Save the trained model
model_json = neuralNetwork.to_json()
with open("../models/model_trained.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
neuralNetwork.save_weights("model.weights.h5")