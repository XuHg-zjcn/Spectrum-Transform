from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import os
vis_base='./hres_vis'
ir_base='./hres_tir'
nlst = np.load('nlst.npy')
scans = np.zeros([len(nlst), 40, 40])
for ni, n in enumerate(nlst):
	print(ni,'/',len(nlst))
	vis_name = os.path.join(vis_base, 'FLIR%04d.jpg'%n)
	ir_name = os.path.join(ir_base, 'FLIR%04d.png'%n)
	vis_img = Image.open(vis_name)
	vis_img = vis_img.resize((427,320), Image.ANTIALIAS)
	vis_img = np.array(vis_img).astype(np.float32)
	ir_img = np.array(Image.open(ir_name)).astype(np.float32)
	#print(2)
	vis_G = np.gradient(gaussian_filter(vis_img, 1), axis=(0,1))
	ir_G = np.gradient(ir_img, axis=(0,1))
	#print(3)
	vis_dy = vis_G[0]
	vis_dx = vis_G[1]
	vis_r = np.sqrt(vis_dx**2 + vis_dy**2)
	vis_r = np.linalg.norm(vis_r, axis=2)
	#print(vis_r.shape)
	ir_dy = ir_G[0]
	ir_dx = ir_G[1]
	ir_r = np.sqrt(ir_dx**2 + ir_dy**2)
	#print(ir_r.shape)
	for i in range(40):
		for j in range(40):
			scans[ni,i,j] = (ir_r*vis_r[20+i:20+i+240,33+j:33+j+320]).mean()
np.save('scans.npy',scans)
