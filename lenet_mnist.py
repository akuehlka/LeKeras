# lenet_mnist.py
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s","--save-model",type=int,default=1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l","--load-model",type=int,default=1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w","--weights",type=str,
                help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab MNIST dataset
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST original",data_home="./mnist_dataset")

# reshape to 28x28 images
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
# add 1 dimension, because that's what keras expect
data = data[:, np.newaxis, :, :]
# 2/3 training/testing split
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.33
)

# convert to categorical labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# only train and evaluate if we are not loading a pre-existent model
if args["load_model"] < 1:
    print("[INFO] Training...")
    model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)
    # show accuracy on the testing set
    print("[INFO] Evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # resize image from 28x28 to 96x96
    image = (testData[i][0]*255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5,20),
                cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)

    # show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)