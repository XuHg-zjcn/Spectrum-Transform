#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 05:37:53 2018

@author: xrj
"""
import math
import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import widgets

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", help="visible image dir")
parser.add_argument("-i", "--ir", help="IR image dir")
args = parser.parse_args()

vis_base=args.vis
ir_base=args.ir

lnpixel = 120*0.014/math.tan(math.radians(17))

nlst = []
for i in os.listdir(vis_base):
     nlst.append(int(i[4:-4]))

'''
l25m: the list of 25 point test of each image shape(nimage, 25)
l45mn: list 25 to 45 point test image number
l45m: shape(nimage, 20)
logtemp: [RAWlargmax, 'str', yshift, xshift]
'''
class align():
    def __init__(self):
        self.img_num = -1
        self.l25m = []
        self.l45mn = []
        self.l45m = []
        self.logs = []
        self.logtemp = []
        self.yshiftNpix = None
        
    def next_img(self):
        self.yshiftNpix = None
        self.logs.append(self.logtemp)
        self.logtemp = []
        self.img_num +=1
        n = nlst[self.img_num]
        
        vis_name = os.path.join(vis_base, 'FLIR'+str(n)+'.jpg')
        ir_name = os.path.join(ir_base, 'FLIR'+str(n)+'.png')
        vis_img = Image.open(vis_name)
        vis_img = vis_img.resize((427,320), Image.ANTIALIAS)
        self.vis_img = np.array(vis_img).astype(np.float32)
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
        ir_g = gaussian_filter(ir_r, 1)
        self.vis_gnorm = vis_g/np.percentile(vis_g,99)
        self.ir_gnorm = ir_g/np.percentile(ir_g,99)
        l25 = []
        for j in range(25):
            l25.append((vis_g[35+j:35+j+240, 53:53+320] * ir_g).mean())
        self.l25m.append(l25)
        lmax = np.array(l25).argmax()
        if lmax > 20:
            self.l45mn.append(self.img_num)
            for j in range(20):
                l25.append((vis_g[60+j:60+j+240, 53:53+320] * ir_g).mean())
            self.l45m.append(l25[25:])
            lmax = np.array(l25).argmax()
        self.logtemp.append(lmax)
        # subplot(223)
        vis_gnorm_crop = self.vis_gnorm[35+lmax:35+lmax+240, 53:53+320]
        grad_img = np.stack((vis_gnorm_crop,self.ir_gnorm,vis_gnorm_crop), axis=2)
        grad_show.clear()
        grad_show.imshow(grad_img)
        # subplot(221)
        vis_crop = self.vis_img[35+lmax:35+lmax+240, 53:53+320]/255.0
        vis_show.clear()
        vis_show.imshow(vis_crop)
        # subplot(222)
        ir_show.clear()
        ir_show.imshow(ir_img)
        # subplot(224)
        line.clear()
        line.plot(l25)
        plt.draw()
        show_l = lnpixel / (lmax - 5)
        namebox.set_val("FLIR%d"%n)
        yshift.set_val(lmax - 5)
        ym.set_val('%.2fm'%show_l)
    
    def pass_click(self, event):
        self.logtemp.append('pass')
        self.next_img()
    
    def not_pass_click(self, event):
        self.logtemp.append('not pass')
        self.next_img()
        
    def change_value_click(self, event):
        self.logtemp.append('change value')
        self.logtemp.append(self.yshiftNpix)
        self.next_img()
    
    def submit_y(self, text):
        try:
            self.yshiftNpix = int(text)
        except ValueError:
            print("text can't to int")
        else:
            n35 = self.yshiftNpix + 5
            
            vis_crop = self.vis_img[35+n35:35+n35+240, 53:53+320]/255.0
            vis_show.clear()
            vis_show.imshow(vis_crop)
            
            vis_gnorm_crop = self.vis_gnorm[35+n35:35+n35+240, 53:53+320]
            grad_img = np.stack((vis_gnorm_crop,self.ir_gnorm,vis_gnorm_crop), axis=2)
            grad_show.clear()
            grad_show.imshow(grad_img)
            
        
aligner  = align()
vis_show = plt.subplot(221)
ir_show = plt.subplot(222)
grad_show = plt.subplot(223)
line = plt.subplot(224)
pass_ax = plt.axes([0.81, 0.01, 0.09, 0.05])
pass_button = widgets.Button(pass_ax, 'pass')
pass_button.on_clicked(aligner.pass_click)

not_pass_ax = plt.axes([0.71, 0.01, 0.03, 0.05])
not_pass_button = widgets.Button(not_pass_ax, 'not pass')
not_pass_button.on_clicked(aligner.not_pass_click)

change_value_ax = plt.axes([0.71, 0.05, 0.03, 0.05])
change_value_button = widgets.Button(change_value_ax, 'change value')
change_value_button.on_clicked(aligner.change_value_click)

namebox_ax = plt.axes([0.01, 0.01, 0.06, 0.05])
namebox = widgets.TextBox(namebox_ax, 'file', 'FLIRxxxx')

yshiftbox_ax = plt.axes([0.08, 0.01, 0.02, 0.02])
yshift = widgets.TextBox(yshiftbox_ax, '', 'xx')
yshift_submit =  yshift.on_submit(aligner.submit_y)

ym_ax = plt.axes([0.11, 0.01, 0.04, 0.02])
ym = widgets.TextBox(ym_ax, '', 'x.xxm')

aligner.next_img()
plt.show()