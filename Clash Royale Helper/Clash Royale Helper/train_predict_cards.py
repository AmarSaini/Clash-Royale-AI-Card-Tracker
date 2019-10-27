import numpy as np

# Training the data
from keras.utils import to_categorical
from LeNetClass import LeNet
# Used for aug data gen
from keras.preprocessing.image import ImageDataGenerator
# Used for training
from keras.optimizers import Adam

# Setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths
# Used for predictions
from keras.models import load_model

# Used for live predictions
import time
from PIL import ImageGrab

# Used for GUI
import tkinter
from PIL import ImageTk
from PIL import Image

# Use other files
from load_train_test_1 import loadTrainingImages1 
from load_train_test_1 import loadTestingImages1 

def trainModel1():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImages1()
    # x_test, y_test = loadTestingImages()
    print(x_train.shape)
    print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255
    #x_test /= 255

    y_train = to_categorical(y_train, num_classes=96)
    #y_test = to_categorical(y_test, num_classes=96)


    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = LeNet.build(width=32, height=32, depth=3, classes=96)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS), 
                            validation_data=(x_train, y_train), steps_per_epoch=len(x_train) // BS,
                            epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save("testNet.model")


def modelPredicts1():

    loadTestingImages1()

    imageNames = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imageNames)):
        imageNames[i] = imageNames[i][imageNames[i].find('/')+1:-4]

    print("[INFO] loading network...")
    model = load_model("testNet.model")

    for i in range(8):
        img = cv2.imread("testData/output" + str(i+1) + ".png")
        orig = img.copy()

        img = cv2.resize(img, (32, 32))
        img = img.astype("float")/255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)


        output = model.predict(img)[0]
        label = output.argmax()

        print(output)
        print(label)

        label = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)

        print(label)

        orig = cv2.resize(orig, (400, 400))
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(0)


def liveModelPredicts1():

    imagePaths = sorted(list(paths.list_images("trainData/")))
    imageNames = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imageNames)):
        imageNames[i] = imageNames[i][imageNames[i].find('/')+1:-4]

    print("[INFO] loading network...")
    model = load_model("testNet.model")

    opponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']
    tempOpponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']

    root = tkinter.Tk()
    myFrame = tkinter.LabelFrame(root, text="Opponent's Cards", labelanchor="n")
    myFrame.pack()

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    while (True):

        if (time.time()-startTime > 1):

            im = ImageGrab.grab()
            im.save("testCNN.png")
            loadTestingImages1()

            for i in range(8):

                if (opponentCards[i] != "MysteryCard"):
                    continue

                img = cv2.imread("testData/output" + str(i+1) + ".png")
                img = cv2.resize(img, (32, 32))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model.predict(img)[0]
                label = output.argmax()

                if (imageNames[label] == "MysteryCard"):
                    continue

                elif (tempOpponentCards[i] == imageNames[label]):
                    opponentCards[i] = imageNames[label]

                    img = Image.open(imagePaths[label])
                    img.thumbnail((128, 128), Image.ANTIALIAS)
                    img = ImageTk.PhotoImage(img)
                    panel = tkinter.Label(myFrame, image = img, borderwidth=10)
                    panel.image = img
                    panel.grid(row=0, column=i)
                    root.update()

                else:
                    tempOpponentCards[i] = imageNames[label]

                labelString = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)

                print(labelString)

            print("--------Opponent's Deck--------")
            print(opponentCards)
            print()
            print()

            startTime = time.time()

# --- CNN 1 ---
#trainModel1()
modelPredicts1()
#liveModelPredicts1()