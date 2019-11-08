import numpy as np

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

def loadTrainingImages1():
    x_train = np.zeros((96, 32, 32, 3))

    imagePaths = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imagePaths)):

        img = cv2.imread(imagePaths[i])
        img = cv2.resize(img, (32, 32))
        img = img_to_array(img)
        x_train[i] = img

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = i

    return x_train, y_train

def loadTestingImages1():

    img = cv2.imread("testCNN.png")
    arr = img_to_array(img)
    cv2.imwrite("croppped.png", arr[58:180, 702:1230])

    arr = arr[58:180, 702:1230]

    cv2.imwrite("testData/output1.png", arr[57:145, 50:104])

    cv2.imwrite("testData/output2.png", arr[57:145, 109:163])

    cv2.imwrite("testData/output3.png", arr[57:145, 168:222])

    cv2.imwrite("testData/output4.png", arr[57:145, 227:281])

    cv2.imwrite("testData/output5.png", arr[57:145, 286:340])

    cv2.imwrite("testData/output6.png", arr[57:145, 345:399])

    cv2.imwrite("testData/output7.png", arr[57:145, 404:458])

    cv2.imwrite("testData/output8.png", arr[57:145, 463:517])
