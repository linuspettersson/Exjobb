from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import subprocess

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

#Get MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

test_images = test_images.reshape(-1, 28 * 28) / 255.0

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()

i = 0
for i in range(5):
	new_model.evaluate(test_images, test_labels, verbose=2)
	
