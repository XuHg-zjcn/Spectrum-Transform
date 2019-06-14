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

import shutil
import os
f = open('out.txt')
fl = f.readline()
fl = f.readline()
while len(fl) != 0:
    a=fl.split()
    if len(a) == 4 and a[3] == 'pass':
        num=int(a[0][4:])
        print(num)
        vis_name = 'FLIR%04d.jpg'%num
        ir_name = 'FLIR%04d.png'%num
        vis_add = './hres_vis/{}'.format(vis_name)
        ir_add = './hres_tir/{}'.format(ir_name)
        os.mkdir('./good_imgs')
        os.mkdir('./good_imgs/good_vis')
        os.mkdir('./good_imgs/good_tir')
        shutil.copy(vis_add, './good_imgs/good_vis/')
        shutil.copy(ir_add, './good_imgs/good_tir/')
    fl = f.readline()
