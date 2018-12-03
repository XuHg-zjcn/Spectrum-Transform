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

import tensorflow as tf
import os
from PIL import Image
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

f = open('outr.txt')
ir_base="../../TT/hres_tir"
vis_base="../../TT/hres_vis"

writer = tf.python_io.TFRecordWriter('./TFRecords1/')
Np=0
Nnp=0
Nerr=0
for i in f:
    i=i[:-1]
    i=i.split()
    li=len(i)
    if li == 4:
        Np += 1
        yshiftNpix=eval(i[1])
        xshiftNpix=eval(i[2])
        '''img=tf.gfile.FastGFile(Visimg_dir+i[0]+'.jpg')
        img=tf.image.decode_jpeg(img)
        img=tf.image.convert_image_dtype(img,dtype=tf.float32)
        img=tf.image.resize_bicubic(img, (320,427))
        img=tf.image.crop_and_resize(img,)'''
        vis_name = os.path.join(vis_base, str(i[0])+'.jpg')
        ir_name = os.path.join(ir_base, str(i[0])+'.png')
        vis_img = Image.open(vis_name)
        vis_img = vis_img.resize((427,320), Image.ANTIALIAS)
        vis_img = np.array(vis_img)#.astype(np.float32)
        vis_img = vis_img[40+yshiftNpix:
                          40+yshiftNpix+240, 
                          53+xshiftNpix:
                          53+xshiftNpix+320]
        print(vis_img.shape)
        vis_img = vis_img.tobytes()
        ir_img = np.array(Image.open(ir_name)).astype(np.uint16)
        print(ir_img.shape)
        ir_img = ir_img.tobytes()
        example=tf.train.Example(features=tf.train.Features(feature={
                 'Vis_image':_bytes_feature(vis_img),
                 'IR_image':_bytes_feature(ir_img)}))
        writer.write(example.SerializeToString())
    elif li == 5:
        Nnp += 1
    else:
        Nerr += 1
    print(i)
writer.close()
print(Np, Nnp, Nerr)
