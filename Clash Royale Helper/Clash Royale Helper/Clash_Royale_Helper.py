import numpy as np

# Used for setting up data
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from imutils import paths

# Used for build
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# Used for aug data gen
from keras.preprocessing.image import ImageDataGenerator

# Used for training
from keras.optimizers import Adam

# Used for predictions
from keras.models import load_model

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

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if (K.image_data_format() == "channels_first"):
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

# CNN 1

def loadTrainingImages1():
    x_train = np.zeros((87, 32, 32, 3))

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
    cv2.imwrite("croppped.png", arr[58:169, 702:1215])

    arr = arr[58:169, 702:1215]

    cv2.imwrite("testData/output1.png", arr[30:97, 16:70])

    cv2.imwrite("testData/output2.png", arr[30:97, 77:131])

    cv2.imwrite("testData/output3.png", arr[30:97, 138:192])

    cv2.imwrite("testData/output4.png", arr[30:97, 199:253])

    cv2.imwrite("testData/output5.png", arr[30:97, 260:314])

    cv2.imwrite("testData/output6.png", arr[30:97, 321:375])

    cv2.imwrite("testData/output7.png", arr[30:97, 382:436])

    cv2.imwrite("testData/output8.png", arr[30:97, 443:497])

def trainModel1():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImages1()
    #x_test, y_test = loadTestingImages()
    print(x_train.shape)
    print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255
    #x_test /= 255

    y_train = to_categorical(y_train, num_classes=87)
    #y_test = to_categorical(y_test, num_classes=87)


    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = LeNet.build(width=32, height=32, depth=3, classes=87)
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

# CNN 2

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
            arr = arr[0:30, 16:70]

        elif (card == 1):
            arr = arr[0:30, 77:131]

        elif (card == 2):
            arr = arr[0:30, 138:192]

        elif (card == 3):
            arr = arr[0:30, 199:253]

        elif (card == 4):
            arr = arr[0:30, 260:314]

        elif (card == 5):
            arr = arr[0:30, 321:375]

        elif (card == 6):
            arr = arr[0:30, 382:436]

        elif (card == 7):
            arr = arr[0:30, 443:497]


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
            arr = arr[0:30, 16:70]

        elif (cardNotPlayed == 1):
            arr = arr[0:30, 77:131]

        elif (cardNotPlayed == 2):
            arr = arr[0:30, 138:192]

        elif (cardNotPlayed == 3):
            arr = arr[0:30, 199:253]

        elif (cardNotPlayed == 4):
            arr = arr[0:30, 260:314]

        elif (cardNotPlayed == 5):
            arr = arr[0:30, 321:375]

        elif (cardNotPlayed == 6):
            arr = arr[0:30, 382:436]

        elif (cardNotPlayed == 7):
            arr = arr[0:30, 443:497]


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
    cv2.imwrite("croppped.png", arr[58:88, 702:1215])

    arr = arr[58:88, 702:1215]

    cv2.imwrite("testData2/output1.png", arr[0:30, 16:70])

    cv2.imwrite("testData2/output2.png", arr[0:30, 77:131])

    cv2.imwrite("testData2/output3.png", arr[0:30, 138:192])

    cv2.imwrite("testData2/output4.png", arr[0:30, 199:253])

    cv2.imwrite("testData2/output5.png", arr[0:30, 260:314])

    cv2.imwrite("testData2/output6.png", arr[0:30, 321:375])

    cv2.imwrite("testData2/output7.png", arr[0:30, 382:436])

    cv2.imwrite("testData2/output8.png", arr[0:30, 443:497])

def trainModel2():
    EPOCHS = 150
    INIT_LR = 1e-3
    BS = 8

    print("[INFO] Loading Images")
    x_train, y_train = loadTrainingImages2()
    print(x_train.shape)
    print(y_train.shape)
    print("[INFO] Images have been loaded.")

    x_train /= 255

    y_train = to_categorical(y_train, num_classes=2)

    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS), 
                            validation_data=(x_train, y_train), steps_per_epoch=len(x_train) // BS,
                            epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save("testNet2.model")

def modelPredicts2():

    loadTestingImages2()

    print("[INFO] loading network...")
    model = load_model("testNet2.model")

    for i in range(8):
        img = cv2.imread("testData2/output" + str(i+1) + ".png")
        orig = img.copy()

        img = cv2.resize(img, (28, 28))
        img = img.astype("float")/255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)


        output = model.predict(img)[0]
        label = output.argmax()
        msg = "Not Placed"

        if (label == 1):
            msg = "Placed"

        print(output)
        print(label)

        label = "Card " + str(i) + " - {}: {:.2f}%".format(msg, output[label] * 100)

        print(label)

        orig = cv2.resize(orig, (400, 400))
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(0)

def liveModelPredicts2():

    print("[INFO] loading network...")
    model = load_model("testNet2.model")

    opponentHand = ['Card 1', 'Card 2', 'Card 3', 'Card 4', 'Card 5', 'Card 6', 'Card 7', 'Card 8']

    print("[INFO] Type anything and press enter to begin...")
    input()

    startTime = time.time()

    while (True):

        if (time.time()-startTime > 1):

            im = ImageGrab.grab()
            im.save("testCNN.png")
            loadTestingImages2()

            for i in range(8):
                img = cv2.imread("testData2/output" + str(i+1) + ".png")
                img = cv2.resize(img, (28, 28))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model.predict(img)[0]
                label = output.argmax()
                msg = "Not Placed"

                if (label == 1):
                    msg = "Placed"
                    opponentHand.remove("Card " + str(i+1))
                    opponentHand.append("Card " + str(i+1))

                labelString = "Card " + str(i+1) + " - {}: {:.2f}%".format(msg, output[label] * 100)

                print(labelString)

            print("--------Opponent's Hand--------")
            print(opponentHand)
            print()
            print()

            startTime = time.time()



def liveBothModelPredicts():

    imagePaths = sorted(list(paths.list_images("trainData/")))
    imageNames = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imageNames)):
        imageNames[i] = imageNames[i][imageNames[i].find('/')+1:-4]

    cardCollection = loadCardCollection()

    print("[INFO] loading both networks...")
    model1 = load_model("testNet.model")
    model2 = load_model("testNet2.model")

    opponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']
    tempOpponentCards = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']

    continuousClassificationCount = [0, 0, 0, 0, 0, 0, 0, 0]
    requiredContinuousClassificationCount = 3

    opponentHand = ['MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard', 'MysteryCard']

    # Cards that are placed before getting classified
    pending = []
    tempPending = []

    pendingElixir = 0

    root = tkinter.Tk()
    elixerFrame = tkinter.LabelFrame(root, text="Opponent's Elixer", labelanchor="n")
    elixerFrame.pack()

    myFrame = tkinter.LabelFrame(root, text="Opponent's Cards in Hand", labelanchor="n")
    myFrame.pack()

    myFrame2 = tkinter.LabelFrame(root, text="Opponent's Upcoming Cards", labelanchor="n")
    myFrame2.pack()

    #myFrame3 = tkinter.LabelFrame(root, text="Opponent's Deck", labelanchor="n")
    #myFrame3.pack()

    panel = tkinter.Label(elixerFrame, text='L')
    panel.grid(row=0, column=0)
    root.update()

    for i in range(4):
        img = Image.open("trainData/MysteryCard.png")
        img.thumbnail((128, 128), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame, image = img, borderwidth=10, bg='green')
        panel.image = img
        panel.grid(row=0, column=i)
        root.update()

    for i in range(4):
        img = Image.open("trainData/MysteryCard.png")
        img.thumbnail((128, 128), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = tkinter.Label(myFrame2, image = img, borderwidth=10, bg='orange')
        panel.image = img
        panel.grid(row=0, column=i)
        root.update()

    print("[INFO] Enter the starting elixer to begin..")
    elixir = int(input())

    elixirRatio = 1/2.8

    startTime = time.time()
    trueStartTime = time.time()
    snapshotTime = 0.4

    while (True):

        elapsedTime = time.time()-startTime

        if (elapsedTime > 120):
            elixirRatio = 1/1.4

        if (elapsedTime > snapshotTime):

            startTime = time.time()

            elixir += elixirRatio * elapsedTime
            if (elixir > 10):
                elixir = 10

            panel = tkinter.Label(elixerFrame, text=format(elixir, '.1f'))
            panel.grid(row=0, column=0)
            root.update()

            im = ImageGrab.grab()
            im.save("testCNN.png")
            loadTestingImages1()
            loadTestingImages2()

            for i in range(8):

                if (opponentCards[i] != "MysteryCard"):
                    continue

                img = cv2.imread("testData/output" + str(i+1) + ".png")
                img = cv2.resize(img, (32, 32))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model1.predict(img)[0]
                label = output.argmax()

                if (imageNames[label] == "MysteryCard"):
                    continue

                elif (tempOpponentCards[i] == imageNames[label]):
                    if (continuousClassificationCount[i] == requiredContinuousClassificationCount):
                        opponentCards[i] = imageNames[label]
                    else:
                        continuousClassificationCount[i] += 1

                    #img = Image.open(imagePaths[label])
                    #img.thumbnail((128, 128), Image.ANTIALIAS)
                    #img = ImageTk.PhotoImage(img)
                    #panel = tkinter.Label(myFrame3, image = img, borderwidth=10)
                    #panel.image = img
                    #panel.grid(row=0, column=i)
                    #root.update()

                else:
                    tempOpponentCards[i] = imageNames[label]
                    continuousClassificationCount[i] = 0

                labelString = "{}: {:.2f}%".format(imageNames[label], output[label] * 100)

                print(labelString)

            # Move all pending cards to the back

            for i in range(len(pending)):

                if (opponentCards[pending[i]] == "MysteryCard"):
                    tempPending.append(pending[i])
                    continue

                else:
                    opponentHand.pop(0)
                    opponentHand.append(opponentCards[pending[i]])

                    elixir -= cardCollection[opponentCards[pending[i]]]

                if (i == len(pending)-1 and len(tempPending) == 0):
                    elixir += elixirRatio * (time.time() - pendingElixir)
                    if (elixir > 10):
                        elixir = 10
                    pendingElixir = 0
                    
            pending = tempPending
            tempPending = []

            for i in range(8):
                img = cv2.imread("testData2/output" + str(i+1) + ".png")
                img = cv2.resize(img, (28, 28))
                img = img.astype("float")/255.0
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)

                output = model2.predict(img)[0]
                label = output.argmax()
                msg = "Not Placed"

                if (label == 1 or (label == 0 and output[label] < .80)):
                    msg = "Placed"
                    if (opponentCards[i] == "MysteryCard"):
                        if (i not in pending):
                            pending.append(i)
                            if (pendingElixir == 0):
                                pendingElixir = time.time()

                    elif (opponentHand.index(opponentCards[i]) < 4):
                        opponentHand.remove(opponentCards[i])
                        opponentHand.append(opponentCards[i])
                        elixir -= cardCollection[opponentCards[i]]

                labelString = "Card " + str(i+1) + " - {}: {:.2f}%".format(msg, output[label] * 100)

                print(labelString)

            for i in range(4):
                img = Image.open("trainData/" + opponentHand[i] + ".png")
                img.thumbnail((128, 128), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                panel = tkinter.Label(myFrame, image = img, borderwidth=10, bg='green')
                panel.image = img
                panel.grid(row=0, column=i)
                root.update()

            for i in range(4):
                img = Image.open("trainData/" + opponentHand[i+4] + ".png")
                img.thumbnail((128, 128), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                panel = tkinter.Label(myFrame2, image = img, borderwidth=10, bg='orange')
                panel.image = img
                panel.grid(row=0, column=i)
                root.update()
            
            print("--------Opponent's Deck--------")
            print(opponentCards)
            print("--------Opponent's Hand--------")
            print(opponentHand)
            print("--------Pending--------")
            print(pending)
            print()
            print()


def createCardCollection():
    imageNames = sorted(list(paths.list_images("trainData/")))

    for i in range(len(imageNames)):
        imageNames[i] = imageNames[i][imageNames[i].find('/')+1:-4]

    cardCollection = dict()

    for x in imageNames:
        print(x)
        cardCollection[x] = int(input())

    with open('cardCollection.txt', 'w') as f:
        for key, value in cardCollection.items():
            f.write('%s:%s\n' % (key, value))

def loadCardCollection():
    data = dict()
    with open('cardCollection.txt') as raw_data:
        for item in raw_data:
            key,value = item.split(':', 1)
            data[key]=int(value[0:value.find('/')])

    return data

# --- CNN 1 ---

#trainModel1()
#modelPredicts1()
#liveModelPredicts1()


# --- CNN 2 ---

#generateTrainingImages2()
#labelTrainingData2()
#trainModel2()
#modelPredicts2()
#liveModelPredicts2()

liveBothModelPredicts()
#createCardCollection()
#print(loadCardCollection())

def testingGUI():

    root = tkinter.Tk()

    myFrame = tkinter.LabelFrame(root, text="Opponent's Cards", labelanchor="n")
    myFrame.pack()

    for r in range(1):
        for c in range(4):
            img = Image.open("trainData/GoblinHutCard.png")
            img.thumbnail((128, 128), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            panel = tkinter.Label(myFrame, image = img, borderwidth=10)
            panel.image = img
            panel.grid(row=r, column=c)
            root.update()

    st = time.time()

    while(True):
        if(time.time() - st > 1):
            img = Image.open("trainData/TheLogCard.png")
            img.thumbnail((128, 128), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            panel = tkinter.Label(myFrame, image = img, borderwidth=10)
            panel.image = img
            panel.grid(row=0, column=2)
            root.update()

            st = time.time()