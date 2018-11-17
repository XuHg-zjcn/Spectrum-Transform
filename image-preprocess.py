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
import math
import os
import time
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import widgets

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", help="visible image dir", 
                    default='../../TT/hres_vis')
parser.add_argument("-r", "--ir", help="IR image dir", 
                    default='../../TT/hres_tir')
parser.add_argument("-i", "--input", help="input npy file")
parser.add_argument("-o", "--out", help="output file")
parser.add_argument("-s", "--start", help="start of file name",
                    type=int)
parser.add_argument("-f", "--first", help="first image num in npy file",
                    type=int, default=0)
args = parser.parse_args()

vis_base=args.vis
ir_base=args.ir

lnpixel = 120*0.014/math.tan(math.radians(17))
nlst = np.load('nlst.npy')
scans = np.load(args.input)

'''
l25m: the list of 25 point test of each image shape(nimage, 25)
l45mn: list 25 to 45 point test image number
l45m: shape(nimage, 20)
logtemp: [RAWlargmax, 'str', yshift, xshift]
'''
class align():
    def __init__(self, filename, start_num, first_img):
        timestruct = time.localtime()
        strtime = time.strftime('%Y-%m-%d %H:%M:%S', timestruct)
        self.outfn = filename
        self.outf = open(self.outfn, 'a')
        self.outf.write('-----'+strtime+'-----\n')
        self.outf.close()
        self.img_num = start_num
        self.first_img = first_img
        self.logtemp = []
        self.yshiftNpix = None
        self.xshiftNpix = 0
    
    #print(num) is a speed test
    def next_img(self):
        self.yshiftNpix = None
        self.xshiftNpix = 0
        self.outf = open(self.outfn, 'a')
        self.outf.write('FLIR%d'%(nlst[self.img_num]))
        if len(self.logtemp) == 3:
            self.outf.write(' %2d %2d %8s\n'%tuple(self.logtemp))
        elif len(self.logtemp) == 5:
            self.outf.write(' %2d %2d %8s %2d %2d\n'%tuple(self.logtemp))
        elif len(self.logtemp) == 0:
            pass
        else:
            raise ValueError('len logtemp not 3 or 5')
        self.outf.close()
        self.logtemp = []
        self.img_num +=1
        n = nlst[self.img_num]
        self.scan = scans[self.img_num - self.first_img]
        #print(1)
        vis_name = os.path.join(vis_base, 'FLIR'+str(n)+'.jpg')
        ir_name = os.path.join(ir_base, 'FLIR'+str(n)+'.png')
        vis_img = Image.open(vis_name)
        vis_img = vis_img.resize((427,320), Image.ANTIALIAS)
        self.vis_img = np.array(vis_img).astype(np.float32)
        ir_img = np.array(Image.open(ir_name)).astype(np.float32)
        #print(2)
        vis_G = np.gradient(gaussian_filter(self.vis_img, 1), axis=(0,1))
        ir_G = np.gradient(ir_img, axis=(0,1))
        #print(3)
        vis_dy = vis_G[0]
        vis_dx = vis_G[1]
        vis_r = np.sqrt(vis_dx*vis_dx + vis_dy*vis_dy)
        vis_r = np.linalg.norm(vis_r, axis=2)
        ir_dy = ir_G[0]
        ir_dx = ir_G[1]
        ir_r = np.sqrt(ir_dx*ir_dx + ir_dy*ir_dy)
        self.vis_norm = vis_r/np.percentile(vis_r, 99)
        self.ir_norm = ir_r/np.percentile(ir_r, 99)
        #print(4)
        self.yshiftNpix, self.xshiftNpix = divmod(self.scan.argmax(), 40)
        self.yshiftNpix -= 20
        self.xshiftNpix -= 20
        #print(5)
        self.logtemp.append(self.yshiftNpix)
        self.logtemp.append(self.xshiftNpix)
        # subplot(223)
        vis_norm_crop = self.vis_norm[40+self.yshiftNpix:
                                      40+self.yshiftNpix+240, 
                                      53+self.xshiftNpix:
                                      53+self.xshiftNpix+320]
        grad_img = np.stack((vis_norm_crop,self.ir_norm,vis_norm_crop), axis=2)
        grad_show.clear()
        grad_show.imshow(grad_img)
        # subplot(221)
        vis_crop = self.vis_img[40+self.yshiftNpix:
                                40+self.yshiftNpix+240, 
                                53+self.xshiftNpix:
                                53+self.xshiftNpix+320]/255.0
        vis_show.clear()
        vis_show.imshow(vis_crop)
        # subplot(222)
        ir_show.clear()
        ir_show.imshow(ir_img)
        # subplot(224)
        scan_show.clear()
        scan_show.imshow(self.scan, 
                         extent=(-20.5,19.5,39.5,-20.5), cmap='gist_ncar')
        
        scan_show.scatter(self.xshiftNpix, self.yshiftNpix, color='red')
        #print(6)
        try:
            show_yl = lnpixel / (self.yshiftNpix)
        except ZeroDivisionError:
            show_yl = float('inf')
        
        try:
            show_xl = lnpixel / (self.xshiftNpix)
        except ZeroDivisionError:
            show_xl = float('inf')
        
        plt.draw()
        print('FLIR%d'%n)
        print('y %2d %.2f'%(self.yshiftNpix, show_yl))
        print('x %2d %.2f'%(self.xshiftNpix, show_xl))
    
    def pass_click(self, event):
        self.logtemp.append('pass')
        self.next_img()
    
    def not_pass_click(self, event):
        self.logtemp.append('not pass')
        self.next_img()
        
    def change_value_click(self, event):
        self.logtemp.append('change')
        #self.logtemp.append(int(input('yshift:')))
        #self.logtemp.append(int(input('xshift:')))
        self.next_img()
        
    def update_show(self):
        n35 = self.yshiftNpix + 5
        xsp = self.xshiftNpix
        
        vis_crop = self.vis_img[35+n35:35+n35+240, 53+xsp:53+xsp+320]/255.0
        vis_show.clear()
        vis_show.imshow(vis_crop)
        
        vis_norm_crop = self.vis_norm[35+n35:35+n35+240, 53+xsp:53+xsp+320]
        grad_img = np.stack((vis_norm_crop,self.ir_norm,vis_norm_crop), axis=2)
        grad_show.clear()
        grad_show.imshow(grad_img)

if args.start is None:
    start_num = -1
else:
    start_num = np.where(np.array(nlst) == args.start)[0][0]
aligner  = align(args.out, start_num, args.first)
vis_show = plt.subplot(221)
ir_show = plt.subplot(222)
grad_show = plt.subplot(223)
scan_show = plt.subplot(224)
pass_ax = plt.axes([0.81, 0.01, 0.09, 0.05])
pass_button = widgets.Button(pass_ax, 'pass')
pass_button.on_clicked(aligner.pass_click)

not_pass_ax = plt.axes([0.71, 0.01, 0.05, 0.03])
not_pass_button = widgets.Button(not_pass_ax, 'not pass')
not_pass_button.on_clicked(aligner.not_pass_click)

change_value_ax = plt.axes([0.71, 0.05, 0.05, 0.03])
change_value_button = widgets.Button(change_value_ax, 'change value')
change_value_button.on_clicked(aligner.change_value_click)
# TextBox.set_val is slowly, I no longer use it
aligner.next_img()
plt.show()
