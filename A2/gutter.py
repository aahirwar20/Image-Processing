import numpy as np
from skimage import io
import cv2 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def conv(img,k):
  (H,W)=img.shape
  (KH,KW)=k.shape
  F=np.zeros(img.shape)
  pad_h=int((KH-1)/2)
  pad_w=int((KW-1)/2)
  padded_img=np.zeros((H+2*pad_h,W+2*pad_w))
  padded_img[pad_h:H+pad_h,pad_w:W+pad_w]=img
  for i in range(H):
    for j in range(W):
      F[i, j] = int(np.sum(k * padded_img[i:i + KH, j:j + KW]))

      #F[i,j] =int( F[i,j]/(KH*KW)) 
  F = F / F.max() * 255
  F=F.astype(int)  
  return np.uint8(F)  

import sys
path=str(sys.argv[1])
image=io.imread(path)
(H,W,f)=image.shape
j=15
k=np.ones((j,j))
img=np.zeros((H,W))
for i in range(H):
   for z in range(W):
    for x in range(3):
      img[i][z]+=image[i][z][x]
img=np.multiply(img,1/3)   
gh= np.ones((7,7))/49 
dilated_img=conv(np.uint8(img),gh)
s=21
k=gaussian_kernel(s, sigma=1)
img2=conv(dilated_img,k)
 
o_img = 255 - cv2.absdiff(np.uint8(img), img2)
f_img=np.zeros((H,W,3))
for i in range(H):
    for j in range(W):
        for z in range(3):
            f_img[i][j][z]=o_img[i][j]
 
from PIL import Image

pilimg = Image.fromarray(np.uint8(f_img))
pilimg.save('cleaned-gutter.jpg')

 