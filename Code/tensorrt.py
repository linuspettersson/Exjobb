#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import os
#import matplotlib.pyplot as plt
import subprocess

from datetime import datetime

from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
from tensorflow import keras
#import tensorflow_datasets as tfds

#converter = trt.TrtGraphConverter(input_graph_def = './chkpts/training_graph.pb', nodes_blacklist = ['logits', 'classes'])
#frozen_graph = converter.convert()
#saved_model_dir_trt = "./saved_model/trtmodel.trt"
#converter.save(saved_model_dir_trt)

with tf.Session() as sess:
    # First deserialize your frozen graph:
    with tf.gfile.GFile("./chkpts/training_graph.pb", "rb") as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    # Now you can create a TensorRT inference graph from your
    # frozen graph:
    converter = trt.TrtGraphConverter(
	    input_graph_def=frozen_graph,
	    nodes_blacklist=['logits', 'classes']) #output nodes
    trt_graph = converter.convert()
    # Import the TensorRT graph into a new graph and run:
    output_node = tf.import_graph_def(
        trt_graph,
        return_elements=['logits', 'classes'])
    sess.run(output_node)

print("exit")
