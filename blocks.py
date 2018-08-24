#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:56:49 2018

@author: xrj
"""
import tensorflow as tf
#from nets.mobilenet import mobilenet_v2
slim = tf.contrib.slim

def get_layer_num():
    global layer_num
    layer_num +=1
    return layer_num

def layer1x1(input_tensor, is_training):
    with tf.variable_scope('1x1'):
        net = slim.dropout(input_tensor, scope='Dropout', is_training=is_training)
        net = slim.conv2d(
            net,
            1280, [1, 1],
            #activation_fn=None,
            #normalizer_fn=None,
            biases_initializer=tf.zeros_initializer(),
            scope='Conv2d_1c_1x1')
    return net

#blocks
def up_block(input_tensor, from_mbnet2, num_outputs, 
             stride=None, lmul=None, fmul=None, **kwargs):
    num_input = num_outputs * 2
    if lmul is None:
        lmul = 1
    if fmul is None:
        fmul = 1
    if stride is None:
        stride = 2
    num_input = input_tensor.get_shape().as_list()[3]
    num_from  = from_mbnet2.get_shape().as_list()[3]
    with tf.variable_scope('up_block_%d'%get_layer_num()):
        net = tf.identity(input_tensor)
        if lmul != 0:
            net = convt1x1(net, num_input*lmul, scope='expand')
            net = convt3x3(net, num_input*lmul, stride, scope='3x3', **kwargs)
        if fmul != 0:
            pre = convt1x1(from_mbnet2, num_from*fmul, scope='pretreat')
        net = tf.concat([net, pre], 3)
        net = convt1x1(net, num_outputs, scope='output', activation_fn=None)
        net = tf.identity(net)
    return net

def norm_block(input_tensor, num_outputs, mul=None):
    if mul is None:
        mul = 6
    num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('norm_block_%d'%get_layer_num()):
        input_tensor = tf.identity(input_tensor)
        net = convt1x1(input_tensor, num_input*mul, scope='expand')
        net = convsep3(net, None         , scope='3x3')
        net = convt1x1(net, num_outputs  , scope='output', activation_fn=None)
        if(num_input == num_outputs): #check add
            net += input_tensor
        net = tf.identity(net)
    return net

def only_up(input_tensor, num_outputs):
    #num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('up_without_from_%d'%get_layer_num()):
        input_tensor = tf.identity(input_tensor)
        net = convt1x1(input_tensor, num_outputs, scope='expand')
        net = convsep3(net, None         , scope='3x3')
        net = tf.identity(net)
    return net
                                   
def convt1x1(input_tensor, num_outputs, **kwargs):
    return slim.conv2d(input_tensor, num_outputs, [1,1], 
                          stride=1, **kwargs);
                       
def convt3x3(input_tensor, num_outputs, stride, scope='3x3', **kwargs):
    global channal_num
    with tf.variable_scope(scope):
        input_tensor = tf.identity(input_tensor)
        return slim.conv2d_transpose(input_tensor, num_outputs, [3,3], 
                              stride=stride, **kwargs);

def convsep3(input_tensor, num_outputs, **kwargs):
    global channal_num
    return slim.separable_conv2d(input_tensor, num_outputs, [3,3], 
                                 depth_multiplier=1, stride=1, **kwargs);

def gen_transpose(input_tensor, mbnet2):
    with tf.variable_scope('transpose'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.relu6, 
                            normalizer_fn=slim.batch_norm,
                            trainable=True):
            net = tf.identity(input_tensor)
            net = tf.tile(net, [1,7,7,1])
            net =   up_block(net, mbnet2['layer_18'], 160, 
                                 stride=2, fmul=5, lmul=0, padding='VALID')  # 18
            net = norm_block(net, 80)                                # 17
            net = norm_block(net, 80)                                                             
            net = norm_block(net, 80)                                # 15
            net =   up_block(net, mbnet2['layer_14'], 48, 
                                 stride=2, fmul=2, lmul=2)                   # 14
            net = norm_block(net, 48)                                # 13
            net = norm_block(net, 48)                                # 12
            net =   up_block(net, mbnet2['layer_11'], 32, 
                                 stride=1, fmul=2, lmul=2)                   # 11
            net = norm_block(net, 32)                                # 10
            net = norm_block(net, 32)                                # 9
            net = norm_block(net, 32)                                # 8
            net =   up_block(net, mbnet2['layer_7'], 16, 
                                 stride=2, fmul=3, lmul=3)                   # 7
            net = norm_block(net, 16)                                # 6
            net = norm_block(net, 16)                                # 5
            net =   up_block(net, mbnet2['layer_4'], 16, 
                                 stride=2, fmul=3, lmul=3)                   # 4
            net = norm_block(net, 16)                                # 3
            net =   up_block(net, mbnet2['layer_2'], 8, 
                                 stride=2, fmul=3, lmul=3)                   # 2
            net =    only_up(net, 16)                                # 1
            net = slim.conv2d_transpose(net, 1, [3, 3], 
                                            stride=2, padding='SAME', scope='irout_3', 
                                            activation_fn=slim.relu, normalizer_fn=None)
            net = tf.identity(net)
    return net