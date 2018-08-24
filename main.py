#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:13:16 2018

@author: xrj
"""

#import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug
from nets.mobilenet import mobilenet_v2
#from nets.mobilenet import mobilenet
#import matplotlib.pyplot as plt

import blocks

slim = tf.contrib.slim
sess = tf.InteractiveSession()



#bulid
transpose_layers = []
#gen_input = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
def G(input_tensor, reuse=False):
    with tf.variable_scope('G', reuse=reuse):
        net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor,num_classes=0)
        net = blocks.layer1x1(net, is_training=True)
        net = blocks.gen_transpose(net, mbnet2)
    return net
    
#den_input = tf.placeholder(shape=(None,224,224,1),dtype=tf.float32)
def D(input_tensor, reuse=False):
    with tf.variable_scope('D'):
        net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor,num_classes=0)
        net = blocks.layer1x1(net, is_training=True)
    return net

#Dloss = 
#Gloss = 
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./mobilenetv2-test', sess.graph)