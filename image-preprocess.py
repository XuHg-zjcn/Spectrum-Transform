#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 05:37:53 2018

@author: xrj
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", help="visible image dir")
parser.add_argument("-i", "--ir", help="IR image dir")
args = parser.parse_args()

vis_base=args.vis
ir_base=args.ir

nlst = []
for i in os.listdir(vis_base):
     nlst.append(int(i[4:-4]))

class align():
    def __init__(self):
        self.img_num = -1
        self.l25m = []
        self.l50mn = [] # num of l50m
        self.l50m = []
        self.largmax = []
        self.wrong = []
        
    def next_img(self):
        self.img_num +=1
        n = nlst[self.img_num]
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
        vis_gnorm = vis_g/np.percentile(vis_g,99)
        ir_gnorm = ir_g/np.percentile(ir_g,99)
        l25 = []
        for j in range(25):
            l25.append((vis_g[35+j:35+j+240, 53:53+320] * ir_g).mean())
        self.l25m.append(l25)
        lmax = np.array(l25).argmax()
        if lmax > 20:
            self.l50mn.append(self.img_num)
            for j in range(25):
                l25.append((vis_g[60+j:60+j+240, 53:53+320] * ir_g).mean())
            self.l50m.append(l25[25:])
            lmax = l25.argmax()
        self.largmax.append(lmax)
        
        vis_gnorm_crop = vis_gnorm[34+lmax:34+lmax+240, 53:53+320]
        grad_img = np.concatenate((vis_gnorm_crop,ir_gnorm,vis_gnorm_crop))
        grad_show.imshow(grad_img)
        
        vis_crop = vis_img[34+lmax:34+lmax+240, 53:53+320]/255.0
        vis_show.imshow(vis_crop)
        
        ir_show.imshow(ir_img)
        
        line.plot(l25)
        
        
aligner  = align()
vis_show = plt.subplot(221)
ir_show = plt.subplot(222)
grad_show = plt.subplot(223)
line = plt.subplot(224)
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
next_button = Button(axnext, 'next')
next_button.on_clicked(aligner.next_img)
plt.show()