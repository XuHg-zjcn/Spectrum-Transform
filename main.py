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
import mobilenet_v2
#from nets.mobilenet import mobilenet
import matplotlib.pyplot as plt

import blocks
#import importlib
#importlib.reload(blocks)
import time

slim = tf.contrib.slim
#sess = tf.InteractiveSession()
batch_size = 12
random_dim = 5
tf.reset_default_graph()
#bulid
#gen_input = tf.placeholder(shape=(None,224,224,3),dtype=tf.float32)

global G_mbnet2
global G_mnout
global G_1x1
def G(input_tensor):
    net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor, num_classes=0)
    global G_mbnet2
    global G_mnout
    global G_1x1
    G_mbnet2 = mbnet2
    G_mnout = net
    net = slim.conv2d(net, 1280, [4,4], stride=1, padding='VALID', 
                      activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm)
    net = blocks.layer1x1(net, 1280, 
                          activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm)
    G_1x1 = net
    tf.summary.histogram('layer1x1',net)
    net = blocks.gen_transpose(net, mbnet2, input_tensor)
    return net
    

#den_input = tf.placeholder(shape=(None,224,224,1),dtype=tf.float32)
global endpoints1
global endpoints2
def D(input_tensor1, input_tensor2, is_summary):
    global endpoints1
    global endpoints2
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
#        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
#            net, mbnet2 = mobilenet_v2.mobilenet_v2_050(input_tensor, num_classes=0)
        net, endpoints1 = blocks.den_net(input_tensor1)
        if is_summary:
            tf.summary.histogram('den_net_gen_IR',net)
        net, endpoints2 = blocks.den_net(input_tensor2)
        if is_summary:
            tf.summary.histogram('den_net_real_IR',net)
        net, sjlist = blocks.den_out(endpoints1, endpoints2, blocks.G_tran)
    return net, sjlist

def parser(record):
    features = tf.parse_single_example(record,
        features={'Vis_image': tf.FixedLenFeature([], tf.string),
                  'IR_image' : tf.FixedLenFeature([], tf.string),})
    Vis_image = tf.decode_raw(features['Vis_image'],tf.uint8)
    Vis_image = tf.reshape(Vis_image, [240, 320, 3])
    #Vis_image = tf.random_crop(Vis_image, (224, 224, 3), seed=12345679)
    #Vis_image = tf.cast(Vis_image, tf.float32) 
    #Vis_image = Vis_image/255.0
    IR_image = tf.decode_raw(features['IR_image'],tf.uint16)
    IR_image = tf.reshape(IR_image, [240, 320, 1])
    #IR_image = tf.random_crop(IR_image, (224, 224, 1), seed=12345679)
    #IR_image = tf.cast(IR_image, tf.float32)
    #IR_image = IR_image/30000.0
    return Vis_image, IR_image
  
def parser2(Vis_image, IR_image):
    Vis_image = tf.random_crop(Vis_image, (224, 224, 3), seed=12345679)
    Vis_image = tf.cast(Vis_image, tf.float32) 
    Vis_image = Vis_image/255.0
    IR_image = tf.random_crop(IR_image, (224, 224, 1), seed=12345679)
    IR_image = tf.cast(IR_image, tf.float32)
    IR_image = IR_image/30000.0
    return Vis_image, IR_image

tf.reset_default_graph()
tfrecords_filename = ["/gdrive/My Drive/Spectrum-Transform/colab_data/TFRecords2", 
 "/gdrive/My Drive/Spectrum-Transform/colab_data/TFRecords3"]
dataset = tf.data.TFRecordDataset(tfrecords_filename)

dataset = dataset.map(parser).shuffle(buffer_size=2000).repeat().map(parser2).batch(batch_size)
iterator = dataset.make_initializable_iterator()
# get the variables list is empty
# copy from https://github.com/bojone/gan/blob/master/mnist_gangp.py
#real_IR = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
#Vis = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
Vis, real_IR = iterator.get_next()
weight_intitializer = tf.truncated_normal_initializer(stddev=0.05)

with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected, slim.separable_conv2d],
                    weights_initializer=weight_intitializer):
    gen_IR = G(Vis)
    #print(gen_IR.shape)
    tf.summary.histogram('gen_IR',gen_IR)
    tf.summary.histogram('Vis',Vis)
    tf.summary.histogram('real_IR',real_IR)
    eps = tf.random_uniform([batch_size, 1, 1, 1], minval=0., maxval=1.)
    X_inter = tf.add(eps*real_IR, (1. - eps)*gen_IR, name='X_inter')
    Dxr, _ = D(X_inter, real_IR, False)
    grad = tf.gradients(Dxr, [X_inter])[0]
    
    tf.summary.histogram('grad',grad)
    grad_norm = tf.reduce_mean(tf.nn.relu(tf.square(grad) - 1.),axis=[0,1,2,3])
    grad_mean, grad_std = tf.nn.moments(grad, axes=[0,1,2,3])
    #print(grad_std.shape)
    grad_pen = 10 * grad_norm
    tf.summary.scalar('grad_mean',grad_mean)
    tf.summary.scalar('grad_std',grad_std)
    
    D_gen_IR, sjlist = D(gen_IR, real_IR, True)
    for ni, i in enumerate(sjlist):
        tf.summary.histogram('D%d'%ni,i)
        
    D_gen_IR = tf.identity(D_gen_IR, name='D_gen_IR')
    D_loss = grad_pen - D_gen_IR
    G_loss = D_gen_IR
#loss = D_loss + G_loss
tf.summary.histogram('mbnet_layer2',G_mbnet2['layer_2'])
tf.summary.histogram('mbnet_layer4',G_mbnet2['layer_4'])
tf.summary.histogram('mbnet_layer7',G_mbnet2['layer_7'])
tf.summary.histogram('mbnet_layer11',G_mbnet2['layer_11'])
tf.summary.histogram('mbnet_layer14',G_mbnet2['layer_14'])
tf.summary.histogram('mbnet_layer18',G_mbnet2['layer_18'])
tf.summary.histogram('mobilenet_v2_out',G_mnout)
for ni, i in enumerate(blocks.G_tran):
    tf.summary.histogram('transpose%d'%ni,i)
tf.summary.scalar('G_loss_D_IRs',G_loss)
tf.summary.scalar('D_loss',D_loss)

D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')
G_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transpose')
#G_mbnet2_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G/MobilenetV2/')

G_solver = tf.train.AdamOptimizer(1e-4, name='Adam_G').minimize(G_loss, var_list=G_variables)
D_solver = tf.train.AdamOptimizer(1e-4, name='Adam_D').minimize(D_loss, var_list=D_variables)
#op = tf.train.AdamOptimizer(1e-4).minimize(loss)
'''Gg = G_Op.compute_gradients(G_loss, G_variables)
Gg2 = []
for ni,i in enumerate(Gg):
    Gg2.append((Gg[ni][1],Gg[ni][1]))
G_solver = G_Op.apply_gradients(Gg2)'''


#Vis_image, IR_image = tf.train.shuffle_batch([Vis_image, IR_image], num_threads=2, batch_size=2, capacity=2000,
#                                                min_after_dequeue=1000)
    
saver = tf.train.Saver(max_to_keep=20)
merged_summary_op = tf.summary.merge_all()
#checkpoint = './mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
#checkpoint = './checkpoint/checkpoint.ckpt'
#saver.restore(sess, checkpoint)
Pretrained_model_dir = './mobilenet_v2_0.5_224/mobilenet_v2_0.5_224.ckpt'
inception_except_logits = slim.get_variables_to_restore(include=['MobilenetV2'], exclude=['MobilenetV2/Logits'])
init_fn = slim.assign_from_checkpoint_fn(Pretrained_model_dir, inception_except_logits, ignore_missing_vars=True)
#sess.graph.finalize()

t0=time.time()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/gdrive/My Drive/Spectrum-Transform/colab_data/summary3', sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(iterator.initializer)
    sess.run(init_op)
    #init_fn(sess)
    saver.restore(sess, '/gdrive/My Drive/Spectrum-Transform/colab_data/checkpoint/net21.ckpt-13000')
    #coord=tf.train.Coordinator()
    #threads= tf.train.start_queue_runners(coord=coord, sess=sess)
    #IR_i, Vis_i= sess.run([IR_image, Vis_image])
    wmg=True
    for i in range(13000, 20001):
        #IR_i2 = IR_i
        #Vis_i2 = Vis_i
        #IR_i, Vis_i = sess.run([IR_image, Vis_image])
        #print("sess.runOK")
        if i%50 == 0:
            t=time.time()
            print(i,(t-t0)/50)
            t0=t
            summary_str, G_l, D_l, gm, gs = sess.run([merged_summary_op, G_loss, D_loss, grad_mean, grad_std])
            summary_writer.add_summary(summary_str, i)
            #print("summaryOK")
            print('loss',G_l, D_l, gm, gs)
          
        if i%500 == 0 and i != 13000:
            saver.save(sess, '/gdrive/My Drive/Spectrum-Transform/colab_data/checkpoint/net21.ckpt', 
                       global_step=i, write_meta_graph=wmg)
            print("SaveOK")
            IR_i, Vis_i, gIR = sess.run([real_IR, Vis, gen_IR])
            plt.subplot(131)
            plt.imshow(IR_i.reshape([224*batch_size,224])[:224*3,])
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(Vis_i.reshape([224*batch_size,224,3])[:224*3,:,])
            plt.subplot(133)
            plt.imshow(gIR.reshape([224*batch_size,224])[:224*3,])
            plt.colorbar()
            plt.show()
        sess.run([G_solver, D_solver])
        wmg=False
        #print("solverOK")
        
