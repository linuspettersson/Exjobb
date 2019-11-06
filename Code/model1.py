#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import subprocess

from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

#subprocess.call(["load_ext", "tensorboard"])

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Eager execution: {}".format(tf.executing_eagerly()))

#Get MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#train_labels = train_labels[:1000]
#test_labels = test_labels[:1000]

train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()

# Define the Keras TensorBoard callback
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_images, train_labels, epochs=1, callbacks=[tensorboard_callback])

subprocess.call(["mkdir", "-p", "saved_model"])
model.save("saved_model/my_model")

#model.evaluate(test_images, test_labels, verbose=1)

# Display the model's architecture
model.summary()

subprocess.call(["tensorboard", "--logdir", "logs/fit"])
  


