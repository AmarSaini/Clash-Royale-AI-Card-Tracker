import numpy as np

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

from random import randint

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Used for Generating/Labeling Data
from shutil import copyfile
import os
from random import randint

def generateTrainingImages2():

    currentNumOfData = len(sorted(list(paths.list_images("generatedData/"))))

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    i = 0

    while (True):

        if (time.time()-startTime > 1):
            print("--------Captured Data--------")

            im = ImageGrab.grab()
            im.save("generatedData/input" + str(i+1+currentNumOfData) + ".png")
            i += 1

            startTime = time.time()

def labelTrainingData2():

    imagePaths = sorted(list(paths.list_images("generatedData/")))
    currentNumOfLabeledData = len(sorted(list(paths.list_images("trainData2/"))))

    root = tkinter.Tk()
    myFrame = tkinter.LabelFrame(root, text="Unlabeled Data", labelanchor="n")
    myFrame.pack()

    labeledCount = 0

    for i in range(len(imagePaths)):
        img = Image.open(imagePaths[i])
        img.thumbnail((1500, 1500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame, image = img)
        panel.image = img
        panel.grid(row=0, column=0)
        root.update()

        label = input()

        if (label != 'e'):
            copyfile(imagePaths[i], "trainData2/"+label+"input"+str(labeledCount+currentNumOfLabeledData)+".png")
            labeledCount += 1

        os.remove(imagePaths[i])

def loadTrainingImages2():

    imagePaths = sorted(list(paths.list_images("trainData2/")))
    x_train = np.zeros((len(imagePaths)*2, 28, 28, 3))

    j = 0

    for i in range(len(imagePaths)):

        # Positive Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite("croppped.png", arr[58:88, 702:1215])

        arr = arr[58:88, 702:1215]

        card = int(imagePaths[i][imagePaths[i].find('/')+1])

        if (card == 0):
            arr = arr[0:30, 50:104]

        elif (card == 1):
            arr = arr[0:30, 109:163]

        elif (card == 2):
            arr = arr[0:30, 168:222]

        elif (card == 3):
            arr = arr[0:30, 227:281]

        elif (card == 4):
            arr = arr[0:30, 286:340]

        elif (card == 5):
            arr = arr[0:30, 345:399]

        elif (card == 6):
            arr = arr[0:30, 404:459]

        elif (card == 7):
            arr = arr[0:30, 464:518]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j] = img

        # Negative Label

        img = cv2.imread(imagePaths[i])
        arr = img_to_array(img)
        #cv2.imwrite("croppped.png", arr[58:88, 702:1215])

        arr = arr[58:88, 702:1215]

        card = int(imagePaths[i][imagePaths[i].find('/')+1])
        nonPlayedCards = np.arange(8)
        nonPlayedCards = nonPlayedCards.tolist()
        nonPlayedCards.remove(card)

        cardNotPlayed = randint(0, 6)

        if (cardNotPlayed == 0):
            arr = arr[0:30, 50:104]

        elif (cardNotPlayed == 1):
            arr = arr[0:30, 109:163]

        elif (cardNotPlayed == 2):
            arr = arr[0:30, 168:222]

        elif (cardNotPlayed == 3):
            arr = arr[0:30, 227:281]

        elif (cardNotPlayed == 4):
            arr = arr[0:30, 286:340]

        elif (cardNotPlayed == 5):
            arr = arr[0:30, 345:399]

        elif (cardNotPlayed == 6):
            arr = arr[0:30, 404:459]

        elif (cardNotPlayed == 7):
            arr = arr[0:30, 464:518]


        img = arr
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        x_train[j+1] = img

        j += 2

    y_train = np.zeros(len(x_train))

    for i in range(len(y_train)):
        y_train[i] = (i+1)%2

    return x_train, y_train

def loadTestingImages2():

    img = cv2.imread("testCNN.png")
    arr = img_to_array(img)
    cv2.imwrite("croppped.png", arr[88:118, 702:1215])

    arr = arr[88:118, 702:1215]

    cv2.imwrite("testData2/output1.png", arr[0:30, 50:104])

    cv2.imwrite("testData2/output2.png", arr[0:30, 109:163])

    cv2.imwrite("testData2/output3.png", arr[0:30, 168:222])

    cv2.imwrite("testData2/output4.png", arr[0:30, 227:281])

    cv2.imwrite("testData2/output5.png", arr[0:30, 286:340])

    cv2.imwrite("testData2/output6.png", arr[0:30, 345:399])

    cv2.imwrite("testData2/output7.png", arr[0:30, 404:459])

    cv2.imwrite("testData2/output8.png", arr[0:30, 464:518])
