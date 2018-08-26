#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:56:49 2018

@author: xrj
"""
import copy
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
slim = tf.contrib.slim
from nets.mobilenet import conv_blocks as ops

D_V2_DEF = copy.deepcopy(mobilenet_v2.V2_DEF)
D_V2_DEF['defaults'][(ops.expanded_conv,)].pop('normalizer_fn')

class layer_num(object):
    
    def __init__(self):
        self.__num = 0
    
    @property
    def get(self):
        self.__num += 1
        return self.__num
    
    def clear(self):
        self.__num = 0

def layer1x1(input_tensor, num_outputs, is_training, 
             activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    with tf.variable_scope('1x1'):
        net = slim.dropout(input_tensor, scope='Dropout', is_training=is_training)
        net = slim.conv2d(
            net, num_outputs, [1, 1], stride=1, padding='VALID', 
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            biases_initializer=tf.zeros_initializer(),
            scope='Conv2d_1c_1x1')
    return net

#blocks
def up_block(input_tensor, from_mbnet2, num_outputs, counter,
             stride=None, lmul=None, fmul=None, **kwargs):
    if lmul is None:
        lmul = 1
    if fmul is None:
        fmul = 1
    if stride is None:
        stride = 2
    num_input = input_tensor.get_shape().as_list()[3]
    num_from  = from_mbnet2.get_shape().as_list()[3]
    with tf.variable_scope('up_block_%d'%counter.get):
        net = tf.identity(input_tensor, name='input')
        if lmul != 0:
            net = convt1x1(net, num_input*lmul, scope='expand')
            #tf.identity(net, name='expand_out')
            net = convt3x3(net, num_input*lmul, stride, scope='3x3', **kwargs)
            #tf.identity(net, name='3x3_out')
        if fmul != 0:
            pre = convt1x1(from_mbnet2, num_from*fmul, scope='preprocess')
            #tf.identity(net, name='preprocess_out')
        net = tf.concat([net, pre], 3)
        net = convt1x1(net, num_outputs, scope='down', activation_fn=None)
        net = tf.identity(net, name='output')
    return net

def norm_block(input_tensor, num_outputs, counter, mul=None):
    if mul is None:
        mul = 6
    num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('norm_block_%d'%counter.get):
        net = tf.identity(input_tensor, name='input')
        net = convt1x1(net, num_input*mul, scope='expand')
        #net = tf.identity(net, name='expand_out')
        net = convsep3(net, None, scope='3x3')
        #net = tf.identity(net, name='3x3_out')
        net = convt1x1(net, num_outputs, scope='down', activation_fn=None)
        if(num_input == num_outputs): #check add
            #net = tf.identity(net, name='down_out')
            net += input_tensor
        net = tf.identity(net, name='output')
    return net

def only_up(input_tensor, num_outputs, counter):
    global numer
    #num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('up_without_from_%d'%counter.get):
        net = tf.identity(input_tensor, name='input')
        net = convt1x1(net, num_outputs, scope='expand')
        #net = tf.identity(input_tensor, name='expand_out')
        net = convsep3(net, None, scope='3x3')
        net = tf.identity(net, name='output')
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
    counter = layer_num()
    with tf.variable_scope('transpose'):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.relu6, 
                            normalizer_fn=slim.batch_norm,
                            trainable=True):
            net =tf.identity(input_tensor, name='intput')
            net =    tf.tile(net, [1,7,7,1])
            net =   up_block(net, mbnet2['layer_18'], 160, counter,
                                 stride=2, fmul=5, lmul=0, padding='VALID')  # 18
            net = norm_block(net, 80, counter)                               # 17
            net = norm_block(net, 80, counter)                               # 16
            net = norm_block(net, 80, counter)                               # 15
            net =   up_block(net, mbnet2['layer_14'], 48, counter, 
                                 stride=2, fmul=2, lmul=2)                   # 14
            net = norm_block(net, 48, counter)                               # 13
            net = norm_block(net, 48, counter)                               # 12
            net =   up_block(net, mbnet2['layer_11'], 32, counter, 
                                 stride=1, fmul=2, lmul=2)                   # 11
            net = norm_block(net, 32, counter)                               # 10
            net = norm_block(net, 32, counter)                               # 9
            net = norm_block(net, 32, counter)                               # 8
            net =   up_block(net, mbnet2['layer_7'], 16, counter, 
                                 stride=2, fmul=3, lmul=3)                   # 7
            net = norm_block(net, 16, counter)                               # 6
            net = norm_block(net, 16, counter)                               # 5
            net =   up_block(net, mbnet2['layer_4'], 16, counter, 
                                 stride=2, fmul=3, lmul=3)                   # 4
            net = norm_block(net, 16, counter)                               # 3
            net =   up_block(net, mbnet2['layer_2'], 8, counter, 
                                 stride=2, fmul=3, lmul=3)                   # 2
            net =    only_up(net, 16, counter)                               # 1
            net = slim.conv2d_transpose(net, 1, [3, 3], 
                                            stride=2, padding='SAME', scope='irout_3')
            net = tf.identity(net, name='output')
    return net

def den_block(input_tensor, num_output, counter, stride=1, mul=6, exp=None):
    num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('den_block%i'%counter.get):
        input_tensor = tf.identity(input_tensor, name='input')
        if exp == False:
            net = slim.conv2d(input_tensor, num_input*mul, [1,1], stride=1, scope='expand')
            net = tf.identity(net, name='expand_out')
        else:
            net = input_tensor
        net = slim.separable_conv2d(net, num_input*mul, [3,3], 
                                 depth_multiplier=1, stride=1, scope='3x3')
        net = tf.identity(net, name='3x3_out')
        net = slim.conv2d(net, num_output, [1,1], stride=stride, scope='down')
        if num_input == num_output and stride==1:
            net = tf.identity(net, name='down_out')
            net += input_tensor
        net = tf.identity(input_tensor, name='output')
        return net
        
def den_net(input_tensor):
    counter = layer_num()
    endpoints = []
    with tf.variable_scope('den_net'):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.relu6, 
                            normalizer_fn=None,
                            trainable=True):
            net = tf.identity(input_tensor, name='input')
            net=slim.conv2d(net, 16, [3,3], stride=2)
            net = den_block(net,  8, counter, stride=1, mul=1, exp=False)
            endpoints.append(net)
            net = den_block(net, 16, counter, stride=2, mul=6)
            net = den_block(net, 16, counter, stride=1, mul=6)
            net = den_block(net, 16, counter, stride=2, mul=6)
            endpoints.append(net)
            net = den_block(net, 16, counter, stride=1, mul=6)
            net = den_block(net, 16, counter, stride=1, mul=6)
            net = den_block(net, 32, counter, stride=2, mul=6)
            endpoints.append(net)
            net = den_block(net, 32, counter, stride=1, mul=6)
            net = den_block(net, 32, counter, stride=1, mul=6)
            net = den_block(net, 32, counter, stride=1, mul=6)
            endpoints.append(net)
            net = den_block(net, 48, counter, stride=1, mul=6)
            net = den_block(net, 48, counter, stride=1, mul=6)
            net = den_block(net, 48, counter, stride=1, mul=6)
            endpoints.append(net)
            net = den_block(net, 80, counter, stride=2, mul=6)
            net = den_block(net, 80, counter, stride=1, mul=6)
            net = den_block(net, 80, counter, stride=1, mul=6)
            endpoints.append(net)
            net = den_block(net,160, counter, stride=1, mul=6)
            net=slim.conv2d(net,1280, [1,1], stride=1)
            net=tf.nn.avg_pool(net, [1, 7, 7, 1], [1, 1, 1, 1], padding='VALID')
            endpoints.append(net)
            net = tf.identity(input_tensor, name='ontput')
    return net, endpoints

def den_out(input_list1,input_list2):
    counterA = layer_num()
    assert len(input_list1) == len(input_list2)
    inlen = len(input_list1)
    lo = 0
    for i in range(inlen):
        with tf.variable_scope('in_%d'%counterA.get):
            counterN = layer_num()
            net = tf.abs(input_list1[i] - input_list2[i])
            dchs = net.get_shape().as_list()[3]
            if(dchs>7):
                net = den_block(net, dchs, counterN, stride=2, mul=6)
            net = den_block(net, dchs, counterN, stride=1, mul=6)
            net = den_block(net, dchs, counterN, stride=1, mul=6)
            net = den_block(net, dchs*2, counterN, stride=1, mul=6)
            assert net.get_shape().as_list()[1] == net.get_shape().as_list()[2]
            patch = net.get_shape().as_list()[1]
            net = slim.avg_pool2d(net, [patch, patch], stride=1, padding='VALID')
            net = tf.identity(net, name='output')
            net = layer1x1(net, 1, is_training=True, 
                           activation_fn=tf.nn.relu, normalizer_fn=None)
        lo += net
    return lo