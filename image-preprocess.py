#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 05:37:53 2018

@author: xrj
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

vis_base='./hres_vis'
ir_base='./hres_tir'
l25m = []
for i in os.listdir('./hres_vis'):
    n = i[:-4]
    print(n)
    vis_name = os.path.join(vis_base,n+'.jpg')
    ir_name = os.path.join(ir_base,n+'.png')
    vis_img = Image.open(vis_name)
    vis_img = vis_img.resize((427,320), Image.ANTIALIAS)
    vis_img = np.array(vis_img).astype(np.float32)
    ir_img = np.array(Image.open(ir_name)).astype(np.float32)
    
    vis_G = np.gradient(vis_img, axis=(0,1))
    ir_G = np.gradient(ir_img, axis=(0,1))

    vis_dy = vis_G[0].mean(axis=2)
    vis_dx = vis_G[1].mean(axis=2)
    vis_r = np.sqrt(vis_dx*vis_dx + vis_dy*vis_dy)
    ir_dy = ir_G[0]
    ir_dx = ir_G[1]
    ir_r = np.sqrt(ir_dx*ir_dx + ir_dy*ir_dy)

    vis_g = gaussian_filter(vis_r, 1.5)
    ir_g = gaussian_filter(ir_r, 1.5)
    l25 = []
    for j in range(25):
        l25.append((vis_g[35+j:35+j+240,53:53+320] * ir_g).mean())
    l25m.append(l25)
l25m = np.array(l25m)

nlst = []
for i in os.listdir('./hres_vis'):
     nlst.append(int(i[4:-4]))