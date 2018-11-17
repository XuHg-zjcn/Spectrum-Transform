#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   Copyright 2018 Xu Ruijun

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
batch_size = 10
random_dim = 5
#bulid
#gen_input = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)
def G(input_tensor):
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor, num_classes=0)
        net = blocks.layer1x1(net, 1280, is_training=True, 
                              activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        net = blocks.gen_transpose(net, mbnet2)
    return net
    

#den_input = tf.placeholder(shape=(None,224,224,1),dtype=tf.float32)
def D(input_tensor1, input_tensor2):
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
#        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
#            net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor, num_classes=0)
        _, endpoints1 = blocks.den_net(input_tensor1)
        _, endpoints2 = blocks.den_net(input_tensor1)
        net = blocks.den_out(endpoints1, endpoints2)
    return net

# get the variables list is empty
# copy from https://github.com/bojone/gan/blob/master/mnist_gangp.py
real_IR = tf.placeholder(tf.float32, shape=[1, 224, 224, 1])
Vis = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
gen_IR = G(Vis)

eps = tf.random_uniform([1, 224, 224, 1], minval=0., maxval=1.)
X_inter = tf.add(eps*real_IR, (1. - eps)*gen_IR, name='X_inter')
grad = tf.gradients(D(X_inter, real_IR), [X_inter])[0]

grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=(1, 2, 3)))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.), name='grad_pen')

D_gen_IR = D(gen_IR, real_IR)
D_gen_IR = tf.identity(D_gen_IR, name='D_gen_IR')
D_loss = grad_pen - D_gen_IR
G_loss = D_gen_IR
#loss = D_loss + G_loss

D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
G_mbnet2_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G/MobilenetV2/')

G_solver = tf.train.AdamOptimizer(1e-4, 0.5, name='Adam_G').minimize(G_loss, var_list=G_variables)
D_solver = tf.train.AdamOptimizer(1e-4, 0.5, name='Adam_D').minimize(D_loss, var_list=D_variables)
#op = tf.train.AdamOptimizer(1e-4).minimize(loss)

#saver = tf.train.Saver(G_mbnet2_varlist)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./mobilenetv2-test', sess.graph)
checkpoint = './mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
#saver.restore(sess, checkpoint)
