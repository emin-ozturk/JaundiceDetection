import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import sklearn.metrics as metrics
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

def loadImage():
  !unzip goz.zip
  print("Görüntüler yüklendi")
  return list(paths.list_images("goz"))

def getImage(dataPaths):
  images = []
  for path in dataPaths:
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    images.append(image)
  images = np.array(images)
  print("Görüntüler verileri alındı")
  return images

def getLabel(dataPaths):
  labels = []
  lb = LabelBinarizer()
  for path in dataPaths:
    labels.append(path.split(os.path.sep)[-2])
  labels = np.array(labels)
  labels = lb.fit_transform(labels)
  labels = to_categorical(labels)
  print("Etiket verileri yüklendi")
  return labels, lb

def getImageDataGenerator():
  train = ImageDataGenerator()
  mean = np.array([123.68, 116.779, 103.939], dtype="float32")
  train.mean = mean
  return train

def createModel():
  baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
  headModel = baseModel.output
  headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
  headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
  headModel = tf.keras.layers.Dense(128, activation="relu")(headModel)
  headModel = tf.keras.layers.Dropout(0.5)(headModel)
  headModel = tf.keras.layers.Dense(2, activation="softmax")(headModel)

  model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
  for layer in baseModel.layers:
    layer.trainable = False
  model.compile(loss="binary_crossentropy", optimizer="adam",	metrics=["accuracy"])
  print("Model oluşturuldu")
  return model

def fitModel(train, trainX, trainY, batch_size, epochs):
  print("Eğitim başladı...")
  result =  model.fit(train.flow(trainX, trainY, batch_size=batch_size), epochs = epochs)
  return model, result

def testModel(model, testX, testY, batch_size, lb):
  print("Test ediliyor")
  pred = model.predict(x = testX.astype("float32"), batch_size = batch_size)
  print(classification_report(testY.argmax(axis=1),	pred.argmax(axis=1), target_names = lb.classes_))
  mae = metrics.mean_absolute_error(Y_test, pred)
  rmse = np.sqrt(metrics.mean_squared_error(Y_test, pred))
  print("Mean Absolute Error: ", mae)
  print("Root Mean Square Error: ", rmse)

def saveModel(model):
  tfjs.converters.save_keras_model(model, "model")
  print("Model .json uzantılı kaydedildi")

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  model = converter.convert()
  with open("model.tflite", 'wb') as f:
    f.write(model)
  print("Model .tflite uzantılı kaydedildi")

if __name__ == "__main__":
  epochs = 600
  batch_size = 20
  dataPaths = loadImage()
  images = getImage(dataPaths)
  labels, lb = getLabel(dataPaths)
  X_train, X_test, Y_train, Y_test = train_test_split(images, labels,	test_size=0.2, random_state=109)
  train = getImageDataGenerator()
  model = createModel()
  model, result = fitModel(train, X_train, Y_train, batch_size, epochs)
  testModel(model, X_test, Y_test, batch_size, lb)
  saveModel(model)