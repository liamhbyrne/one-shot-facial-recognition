from typing import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from dataPrep import Prepper

class SiamModel:
    def __init__(self):
        self._baseModel : Sequential


    def buildBaseModel(self, inp_shape : Tuple):
        self._baseModel = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(6, (3, 3), activation='relu', input_shape=inp_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            # The second convolution
            tf.keras.layers.Conv2D(12, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.25),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(50, activation='relu')
        ])
        self._baseModel.summary()

    def fitModel(self, X_train, X_test, Y_train, Y_test):
        image_dimensions = (*X_train.shape[-2:], 1)

        first_image = tf.keras.Input(shape=image_dimensions)
        second_image = tf.keras.Input(shape=image_dimensions)

        self.buildBaseModel(inp_shape=image_dimensions)

        first_feature_vector = self._baseModel(first_image)
        second_feature_vector = self._baseModel(second_image)
        distance_metric = tf.keras.layers.Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([first_feature_vector, second_feature_vector])
        model = tf.keras.Model(inputs=[first_image, second_image], outputs=distance_metric)
        model.compile(loss=self.contrastive_loss, optimizer='adam', metrics=['acc'])
        img_1 = X_train[:, 0]
        img2  = X_train[:, 1]
        model.fit([img_1, img2], Y_train, validation_split=.25,
                  batch_size=128, verbose=1, epochs=13)
        print("PRED SHAPE", X_test[:, 0].shape)
        pred = model.predict([X_test[:, 0], X_test[:, 1]])
        print("accuracy metric ::", Y_test[pred.ravel() < 0.5].mean())
        self._finalModel = model

    def predictIMGs(self, arr1, arr2):
        print("DISTANCE ::=", self._finalModel.predict([arr1, arr2]))

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    @staticmethod
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

dataCollector = Prepper(PATH_TO_KNOWN_FACES)
dataCollector.findRealPairs()
dataCollector.findFakePairs()
X_train, X_test, Y_train, Y_test = dataCollector.aggregateDataset(1800, 0.3)
siam = SiamModel()
siam.fitModel(X_train, X_test, Y_train, Y_test)

im1 = np.zeros([1, 112,92])
im1[0] = dataCollector.convertPGMtoNumpy(PATH_TO_KNOWN_FACE_1)
im2 = np.zeros([1, 112,92])
im2[0] = dataCollector.convertPGMtoNumpy(PATH_TO_KNOWN_FACE_1)
siam.predictIMGs(im1/255, im2/255)
