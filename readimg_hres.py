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
import os
import imageio
import png
import numpy as np
from PIL import Image
root="./hres_img"
n=0
nlst=[]
for i in os.listdir(root):
  raw_add = os.path.join(root,i)
  ImSize=Image.open(raw_add).size
  if ImSize == (320,240):
    im2 = True
  elif ImSize == (640,480):
    im2 = False
  else:
    raise ValueError
  print(raw_add, im2)
  if i[-4:]=='.jpg' and i[:4]=='FLIR' and im2:
    n+=1
    os.system("exiftool -b -RawThermalImage %s > ./temp/tir.png"%raw_add)
    im=imageio.imread('./temp/tir.png')
    im=im*256+(im//256.0).astype(int)
    png.from_array(im, 'L;16').save('./hres_tir/%s.png'%(i[:-4]))
    os.system("exiftool -b -EmbeddedImage %s > ./hres_vis/%s.jpg"%(raw_add,i[:-4]))
    nlst.append(int(i[4:-4]))
print(nlst)
np.save('nlst.npy',nlst)
print('{} images'.format(n))
