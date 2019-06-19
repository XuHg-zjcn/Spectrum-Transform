#@title
%%writefile blocks.py

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
import copy
import tensorflow as tf
import mobilenet_v2
slim = tf.contrib.slim
import conv_blocks as ops
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops import array_ops

#D_V2_DEF = copy.deepcopy(mobilenet_v2.V2_DEF)
#D_V2_DEF['defaults'][(ops.expanded_conv,)].pop('normalizer_fn')

'''@ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
def _DepthwiseConv2DNativeBackpropInputGrad(op, grad):
  """The derivatives for depth-wise deconvolution.
  Args:
    op: the depth-wise deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output
  Returns:
    the gradients w.r.t. the input and the filter
  """
  return [None,
          nn_ops.depthwise_conv2d_native_backprop_filter(
              grad,
              array_ops.shape(op.inputs[1]),
              op.inputs[2],
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format")),
          nn_ops.depthwise_conv2d_native(
              grad,
              op.inputs[1],
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format"))]'''

class layer_num(object):
    
    def __init__(self):
        self.__num = 0
    
    @property
    def get(self):
        self.__num += 1
        return self.__num
    
    def clear(self):
        self.__num = 0

is_training = False#tf.placeholder(shape=(), dtype=tf.bool)
def layer1x1(input_tensor, num_outputs, 
             activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.batch_norm):
    global is_training
    with tf.variable_scope('1x1'):
        net = slim.dropout(input_tensor, 0.5, \
                           scope='Dropout', is_training=True)
        net = slim.conv2d(
            net, num_outputs, [1, 1], stride=1, padding='VALID', 
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            biases_initializer=tf.zeros_initializer(),
            scope='Conv2d_1c_1x1')
    return net

#blocks
from_mbnet2s = []
from_mbnet2s_dropout = []
keep_probs = [0.3,0.2,0.1,0.05,0.03,0.02,0.02]
def up_block(input_tensor, from_mbnet2, num_outputs, counter, 
             stride=None, lmul=None, fmul=None, **kwargs):
    global is_training
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
        from_mbnet2 = tf.identity(from_mbnet2, name='from_mbnet2')
        l=len(from_mbnet2s)
        from_mbnet2s.append(from_mbnet2)
        #from_mbnet2 = slim.dropout(from_mbnet2, keep_probs[l], 
        #                           scope='Dropout',is_training=is_training)
        from_mbnet2s_dropout.append(from_mbnet2)
        if lmul != 0:
            net = convt1x1(net, num_input*lmul, scope='expand')
            #tf.identity(net, name='expand_out')
            net = convt3x3(net, num_input*lmul, stride, scope='3x3', **kwargs)
            #tf.identity(net, name='3x3_out')
        if fmul != 0:
            pre = convt1x1(from_mbnet2, num_from*fmul, scope='preprocess')
            #tf.identity(net, name='preprocess_out')
        if net.shape[1] == 8 and pre.shape[1] == 7:
            net = net[:,:7,:7,:]
        net = tf.concat([net, pre], 3)
        net = convt1x1(net, num_outputs, scope='down', activation_fn=None)
        net = tf.identity(net, name='output')
    return net

def norm_block(input_tensor, num_outputs, counter, 
               mul=None, connect=None):
    if mul is None:
        mul = 6
    if connect is None:
        connect = True
    num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('norm_block_%d'%counter.get):
        net = tf.identity(input_tensor, name='input')
        net = convt1x1(net, num_input*mul, scope='expand')
        #net = tf.identity(net, name='expand_out')
        net = convsep3(net, None, scope='3x3')
        #net = tf.identity(net, name='3x3_out')
        net = convt1x1(net, num_outputs, scope='down', activation_fn=None)
        if num_input == num_outputs and connect: #check add
            #net = tf.identity(net, name='down_out')
            net += input_tensor
        net = tf.identity(net, name='output')
    return net

def only_up(input_tensor, num_outputs, counter):
    #global numer
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
                          stride=1, **kwargs)
                       
def convt3x3(input_tensor, num_outputs, stride, scope='3x3', **kwargs):
    global channal_num
    with tf.variable_scope(scope):
        input_tensor = tf.identity(input_tensor)
        return slim.conv2d_transpose(input_tensor, num_outputs, [3,3], 
                              stride=stride, **kwargs)

def convsep3(input_tensor, num_outputs, **kwargs):
    global channal_num
    return slim.separable_conv2d(input_tensor, num_outputs, [3,3], 
                                 depth_multiplier=1, stride=1, **kwargs)
G_tran=[]
def gen_transpose(input_tensor, mbnet2, Vis_RGB):
    counter = layer_num()
    with tf.variable_scope('transpose'):
        with slim.arg_scope([slim.conv2d, \
                             slim.conv2d_transpose, \
                             slim.separable_conv2d], \
                            padding='SAME', \
                            activation_fn=tf.nn.leaky_relu, \
                            normalizer_fn=slim.batch_norm, \
                            trainable=True):
            net =tf.identity(input_tensor, name='intput')
            G_tran.append(net)#1
            net = slim.conv2d_transpose(net, 160, [4,4], stride=1, padding='VALID')
            net =   up_block(net, mbnet2['layer_20'], 160, counter,
                             stride=1, fmul=1, lmul=0)
            net = norm_block(net, 160, counter)
            net = norm_block(net, 160, counter)
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_17'], 80, counter,
                                 stride=2, fmul=2, lmul=2, padding='SAME') # 17
            net = norm_block(net, 80, counter)                              # 16
            net = norm_block(net, 80, counter)                              # 15
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_14'], 48, counter, 
                                 stride=2, fmul=2, lmul=2)                  # 14
            net = norm_block(net, 48, counter)                              # 13
            net = norm_block(net, 48, counter)                              # 12
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_11'], 32, counter, 
                                 stride=1, fmul=2, lmul=2)                  # 11
            net = norm_block(net, 32, counter)                              # 10
            net = norm_block(net, 32, counter)                              # 9
            net = norm_block(net, 32, counter)                              # 8
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_7'], 16, counter, 
                                 stride=2, fmul=3, lmul=3)                  # 7
            net = norm_block(net, 16, counter)                              # 6
            net = norm_block(net, 16, counter)                              # 5
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_4'], 16, counter, 
                                 stride=2, fmul=3, lmul=3)                  # 4
            net = norm_block(net, 16, counter)                              # 3
            G_tran.append(net)
            net =   up_block(net, mbnet2['layer_2'], 16, counter, 
                                 stride=2, fmul=3, lmul=3)                  # 2
            G_tran.append(net)
            net = norm_block(net, 16, counter)                              # 1
            net = norm_block(net, 16, counter)
            G_tran.append(net)
            net =   up_block(net, Vis_RGB, 16, counter, 
                                 stride=2, fmul=3, lmul=3)
            G_tran.append(net)
            net = norm_block(net, 8, counter, connect = False)
            G_tran.append(net)
            net = norm_block(net, 8, counter)
            with slim.arg_scope([slim.conv2d, \
                             slim.conv2d_transpose, \
                             slim.separable_conv2d], \
                            padding='SAME', \
                            activation_fn=tf.nn.leaky_relu, \
                            normalizer_fn=None, \
                            trainable=True):
                net = norm_block(net, 1, counter)
            G_tran.append(net)
            net = tf.identity(net, name='output')
    return net

def den_block(input_tensor, num_output, counter, stride=1, mul=6, exp=None):
    num_input = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope('den_block%i'%counter.get):
        input_tensor = tf.identity(input_tensor, name='input')
        if exp == False:
            net = slim.conv2d(input_tensor, num_input*mul, [1,1], \
                              stride=1, scope='expand')
            net = tf.identity(net, name='expand_out')
        else:
            net = input_tensor
        net = slim.conv2d(net, num_input*mul, [3,3], 
                                    stride=1, scope='3x3')
        #net = slim.separable_conv2d(net, num_input*mul, [3,3], 
        #                  depth_multiplier=1, stride=1, scope='3x3')
        net = tf.identity(net, name='3x3_out')
        net = slim.conv2d(net, num_output, [1,1], stride=stride, scope='down')
        if num_input == num_output and stride==1:
            net = tf.identity(net, name='down_out')
            net += input_tensor
        net = tf.identity(net, name='output')
        return net
        
def den_net(input_tensor):
    counter = layer_num()
    endpoints = []
    with tf.variable_scope('den_net'):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.leaky_relu, 
                            normalizer_fn=slim.layer_norm,
                            trainable=True):
            net = tf.identity(input_tensor, name='input')
            endpoints.append(net)#224
            net=slim.conv2d(net, 16, [3, 3], stride=2)
            net = den_block(net,  8, counter, stride=1, mul=1, exp=False)
            endpoints.append(net)#112
            net = den_block(net, 16, counter, stride=2, mul=6)
            net = den_block(net, 16, counter, stride=1, mul=6)
            endpoints.append(net)#56
            net = den_block(net, 16, counter, stride=2, mul=6)
            net = den_block(net, 16, counter, stride=1, mul=6)
            net = den_block(net, 16, counter, stride=1, mul=6)
            endpoints.append(net)#28
            net = den_block(net, 32, counter, stride=2, mul=6)
            net = den_block(net, 32, counter, stride=1, mul=6)
            net = den_block(net, 32, counter, stride=1, mul=6)
            net = den_block(net, 32, counter, stride=1, mul=6)
            endpoints.append(net)#14
            net = den_block(net, 48, counter, stride=1, mul=6)
            net = den_block(net, 48, counter, stride=1, mul=6)
            net = den_block(net, 48, counter, stride=1, mul=6)
            endpoints.append(net)#14
            net = den_block(net, 80, counter, stride=2, mul=6)
            net = den_block(net, 80, counter, stride=1, mul=6)
            net = den_block(net, 80, counter, stride=1, mul=6)
            endpoints.append(net)#7
            net = den_block(net,160, counter, stride=2, mul=6)
            net = den_block(net,160, counter, stride=1, mul=6)
            net = den_block(net,160, counter, stride=1, mul=6)
            endpoints.append(net)#4
            net = slim.conv2d(net, 640, [4,4], stride=1, padding='VALID')
            endpoints.append(net)
            net = tf.identity(input_tensor, name='ontput')
    return net, endpoints
net224=[]
def den_outd(net, i):
    global net224
    counterN = layer_num()
    dchs = net.get_shape().as_list()[3]
    Nwide = net.get_shape().as_list()[1]
    with tf.variable_scope('in_%d'%i, reuse=tf.AUTO_REUSE):
        if Nwide >= 4:
            if Nwide == 224:
                print(224)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.leaky_relu, 
                            normalizer_fn=None,
                            trainable=True):
                    net = den_block(net, dchs*2, counterN, stride=1, mul=6)
                    net224.append(net)
            else:
                net = den_block(net, dchs*2, counterN, stride=1, mul=6)
        #Nwide = net.get_shape().as_list()[1]
        #print(nM)
        assert net.get_shape().as_list()[1] == net.get_shape().as_list()[2]
        #if Nwide > 7:
        #    net = slim.max_pool2d(net, [3, 3], \
        #                          stride=2, padding='SAME')
    return net
    
def den_out(input_list1, input_list2, G_tranlist):
    assert len(input_list1) == len(input_list2)
    inlen = len(input_list1)
    lo=0
    sjlist = []
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.leaky_relu, 
                            normalizer_fn=slim.layer_norm,
                            trainable=True):
        for i in range(inlen):
            #with tf.variable_scope('in_%d'%counterA.get):
            #dchs = input_list1[i].get_shape().as_list()[3]
            if i >= 2:
                G_tran = G_tranlist[-i-4]
            elif i == 1:
                G_tran = G_tranlist[-4]
            elif i == 0:
                G_tran = G_tranlist[-1]
            else:
                raise ValueError
            net1 = input_list1[i]
            net1 = tf.concat([net1, G_tran], axis=3)
            net1 = den_outd(net1, i)
            #print(net1.shape)
            #net1 = tf.identity(net1, name='output1')

            net2 = input_list2[i]
            net2 = tf.concat([net2, G_tran], axis=3)
            net2 = den_outd(net2, i)
            #net2 = tf.identity(net2, name='output2')
            sjlist.append(net1)
            sjlist.append(net2)
            l = tf.reduce_mean(tf.square(net2 - net1), axis=[1,2,3])
            l = tf.reduce_mean(l)
            lo += l
    return lo, sjlist
