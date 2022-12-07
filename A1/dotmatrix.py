import numpy as np

import matplotlib.pyplot as plt
import math
img_1d=[]
for i in range(3):
  img_1d.append(0)
img_1=[]
for i in range(500):
  img_1.append(img_1d)
img1=[]  
for i in range(300):
  img1.append(img_1)
img =np.array(img1)  

def draw_circle(c,img):
  
   for i in range(50):
     for j in range(50):
       for z in range(3):
         if math.pow(i-25,2)+math.pow(j-25,2)<math.pow(25,2):
           img[i+c[1]-25][j+c[0]-25][z]=255
  
   return img

mat = {} 

mat['0'] = [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]]
mat['1'] = [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
mat['2'] = [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]]
mat['3'] = [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]]
mat['4'] = [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]]
mat['5'] = [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]]
mat['6'] = [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]]
mat['7'] = [[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
mat['8'] = [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]]
mat['9'] = [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]]

def draw_no(img,s,ch):
  t=[60+s,0]
  c=t.copy()
  for i in range(5):
     c[1]=c[1]+30
     c[0]=t[0]
     for j in range(3):
       if mat[ch][i][j]==1:
          img =draw_circle(c,img)
       c[0]=c[0]+60
     c[1]=c[1]+30     
  return img
import sys

d=int(sys.argv[1])

if(d<10):
  num='0'+str(d)
else:  
  num=str(d)
s=-250
for ch in num:
  s=s+250
  img=draw_no(img,s,ch)
from PIL import Image

pilimg = Image.fromarray(np.uint8(img))
pilimg.save('dotmatrix.jpg')  


# plt.imshow(img)
# plt.show()