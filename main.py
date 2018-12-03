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
import matplotlib.pyplot as plt

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
        tf.summary.histogram('mobilenet_v2',net)
        #net = blocks.layer1x1(net, 1280, is_training=True, 
        #                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        tf.summary.histogram('layer1x1',net)
        net = blocks.gen_transpose(net, mbnet2)
    return net
    

#den_input = tf.placeholder(shape=(None,224,224,1),dtype=tf.float32)
def D(input_tensor1, input_tensor2):
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
#        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
#            net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor, num_classes=0)
        net, endpoints1 = blocks.den_net(input_tensor1)
        tf.summary.histogram('den_net_gen_IR',net)
        net, endpoints2 = blocks.den_net(input_tensor2)
        tf.summary.histogram('den_net_real_IR',net)
        net = blocks.den_out(endpoints1, endpoints2)
    return net

# get the variables list is empty
# copy from https://github.com/bojone/gan/blob/master/mnist_gangp.py
real_IR = tf.placeholder(tf.float32, shape=[1, 224, 224, 1])
Vis = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
weight_intitializer = tf.truncated_normal_initializer(stddev=1.0)
with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected, slim.separable_conv2d],
                    weights_initializer=weight_intitializer):
    gen_IR = G(Vis)
    tf.summary.histogram('gen_IR',gen_IR)
    tf.summary.histogram('Vis',Vis)
    eps = tf.random_uniform([1, 224, 224, 1], minval=0., maxval=1.)
    X_inter = tf.add(eps*real_IR, (1. - eps)*gen_IR, name='X_inter')
    grad = tf.gradients(D(X_inter, real_IR), [X_inter])[0]
    
    tf.summary.histogram('grad',grad)
    grad_norm = tf.reduce_sum((grad)**2, axis=(1, 2, 3))
    grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.), name='grad_pen')
    print(grad_norm.shape)
    tf.summary.histogram('grad_norm',grad_norm)
    tf.summary.scalar('grad_pen',grad_pen)
    
    tf.summary.histogram('real_IR',real_IR)
    D_gen_IR = D(gen_IR, real_IR)
    D_gen_IR = tf.identity(D_gen_IR, name='D_gen_IR')
    D_loss = grad_pen - D_gen_IR
    G_loss = D_gen_IR
#loss = D_loss + G_loss
tf.summary.scalar('G_loss(D_gen_IR)',G_loss)
tf.summary.scalar('D_loss',D_loss)

D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
G_mbnet2_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G/MobilenetV2/')

G_solver = tf.train.AdamOptimizer(1e-3, name='Adam_G').minimize(G_loss, var_list=G_variables)
D_solver = tf.train.AdamOptimizer(1e-3, name='Adam_D').minimize(D_loss, var_list=D_variables)
#op = tf.train.AdamOptimizer(1e-4).minimize(loss)
tfrecords_filename = './TFRecords1/'
filename_queue = tf.train.string_input_producer([tfrecords_filename],)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
           features={'Vis_image': tf.FixedLenFeature([], tf.string),
                     'IR_image' : tf.FixedLenFeature([], tf.string),})
Vis_image = tf.decode_raw(features['Vis_image'],tf.uint8)
Vis_image = tf.reshape(Vis_image, [240, 320, 3])

IR_image = tf.decode_raw(features['IR_image'],tf.uint16)
IR_image = tf.reshape(IR_image, [240, 320, 1])
#print(Vis_image.shape)
#print(IR_image.shape)
Vis_image = tf.random_crop(Vis_image, (224, 224, 3), seed=12345678)
IR_image = tf.random_crop(IR_image, (224, 224, 1), seed=12345678)
Vis_image, IR_image = tf.train.shuffle_batch([Vis_image, IR_image], num_threads=2, batch_size=1, capacity=2000,
                                                min_after_dequeue=1000)
    
saver = tf.train.Saver(max_to_keep=20)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./mobilenetv2-test', sess.graph)
#checkpoint = './mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
checkpoint = './checkpoint/checkpoint.ckpt'
#saver.restore(sess, checkpoint)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        print(i)
        IR_i, Vis_i = sess.run([IR_image, Vis_image])
        '''plt.subplot(211)
        plt.imshow(IR_i.reshape([224,224]))
        plt.subplot(212)
        plt.imshow(Vis_i.reshape([224,224,3]))
        plt.show()'''
        print("sess.runOK")
        IR_i = IR_i/30000.0
        Vis_i = Vis_i/255.0
        if(i%1 == 0):
            summary_str=sess.run(merged_summary_op, feed_dict={real_IR:IR_i, Vis:Vis_i})
            summary_writer.add_summary(summary_str, i)
            print("summaryOK")
        if(i%5 == 0):
            saver.save(sess, 'checkpoint/net.ckpt', global_step=i)
            print("SaveOK")
        sess.run(G_solver, {real_IR:IR_i, Vis:Vis_i})
        sess.run(D_solver, {real_IR:IR_i, Vis:Vis_i})
        print("solverOK")
