#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import subprocess

from datetime import datetime

from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

converter = trt.TrtGraphConverter(input_saved_model_dir = './saved_model')
converter.convert()
saved_model_dir_trt = "./saved_model/trtmodel.trt"
converter.save(saved_model_dir_trt)

#with tf.Session() as sess:
#	tf.saved_model.loader.load(
#		sess, [tf.saved_model.tag_constants.SERVING], output)
#	output = sess.run([output_tensor], feed_dict={input_tensor: input_data})
