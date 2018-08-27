import os
import imageio
import png
root="./hres_img"
n=0
for i in os.listdir(root):
  if(i[-4:]=='.jpg' and i[:4]=='FLIR' and int(i[4:-4])>9614):
    n+=1
    os.system("exiftool -b -RawThermalImage %s > ./temp/tir.png"%(os.path.join(root,i)))
    im=imageio.imread('./temp/tir.png')
    im=im*256+(im//256.0).astype(int)
    png.from_array(im, 'L;16').save('./hres_tir/%s.png'%(i[:-4]))
    os.system("exiftool -b -EmbeddedImage %s > ./hres_vis/%s"%(os.path.join(root,i),i))
print('{} images'.format(n))
