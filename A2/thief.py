#  theft
import numpy as np
import math 
from PIL import Image
import sys
from skimage import io

path=str(sys.argv[1])
imag= np.array(io.imread(path))
img=np.copy(imag)
gama=2
H=img.shape[0]
W=img.shape[1]
LT=0
HT=130
for i in range(H):
  for j in range(W):
    for z in range(3):
      img[i][j][z]=math.pow(img[i][j][z],1/gama)*math.pow(255,(gama-1)/gama)
      if img[i][j][z]<LT:
        img[i][j][z]=0
      elif img[i][j][z]>=LT and img[i][j][z] <=HT :
        img[i][j][z]=(255/(HT-LT))*(img[i][j][z]-LT)
      else :
        img[i][j][z]=255  
plimg=Image.fromarray(np.uint8(img))
#plt.imshow(plimg)
plimg.save('enhanced-cctv3.jpg')
