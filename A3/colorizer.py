#3
import numpy as np
import sys
from skimage import io

path=str(sys.argv[1])
path2=str(sys.argv[2])
path3=str(sys.argv[3])


img= np.array(io.imread(path))

C1=np.copy(np.array(io.imread(path3)))
C2=np.copy(np.array(io.imread(path2)))
Y=np.copy(np.array(io.imread(path)))
Cr=np.uint8(np.zeros((C1.shape[0]*4-2,C1.shape[1]*4)))
Cb=np.uint8(np.zeros((C1.shape[0]*4-2,C1.shape[1]*4)))



def gaussian_kernel(size, sigma=3):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def g2(size, sigma=3):
    size = int(size) 
    x = np.mgrid[0:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2) / (2.0*sigma**2))) * normal
    return g    
k2=g2(255,10)
def rt(x,y):
  return k2[int(abs(x-y))]
def conv(img,k,k2):
  H=img.shape[0]
  W=img.shape[1]
  KH=k.shape[0]
  KW=k.shape[1]
  F=np.zeros(img.shape)
  pad_h=int((KH-1)/2)
  pad_w=int((KW-1)/2)
  padded_img=np.zeros((H+2*pad_h,W+2*pad_w))
  padded_img[pad_h:H+pad_h,pad_w:W+pad_w]=img
  Kr=np.zeros((KH,KW))
  # print(Kr.shape)
  # print(k.shape)
  for i in range(H):
    for j in range(W):
      Krf=np.vectorize(rt)
      Kr=Krf(padded_img[i:i+KH,j:j+KW],padded_img[i+int(KH/2)][j+int(KW/2)])
      Ka=np.multiply(Kr,k)    
      F[i, j] = int(np.sum(Ka* padded_img[i:i + KH, j:j + KW])/np.sum(Ka))
  F=F.astype(int)  
  return np.uint8(F) 


k=gaussian_kernel(5,10)
def rgb(Cr,Cb,Y):
  img=np.uint8(np.zeros((Y.shape[0],Y.shape[1],3)))
  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      img[i][j][0]=Y[i][j]+1.402*(Cr[i][j]-128)
      img[i][j][1]=Y[i][j]-0.3441136*(Cb[i][j]-128)-0.714136*(Cr[i][j]-128)
      img[i][j][2]=Y[i][j]+1.772*(Cb[i][j]-128) 

  return img

#return np.array(img, dtype='i8')


for i in range(C1.shape[0]):
  for j in range(C1.shape[1]):
    for z in range(4):
      for d in range(4):
        if int(4*i)+z<Y.shape[0] and int(4*j)+d<Y.shape[1]:
         Cr[int(4*i)+z][int(4*j)+d]=C1[i][j]
         Cb[int(4*i)+z][int(4*j)+d]=C2[i][j]

# print(Cb)
Cr=conv(Cr,k,k2)    
Cb=conv(Cb,k,k2)
img=rgb(Cr,Cb,Y)         

from PIL import Image

pilimg = Image.fromarray(np.uint8(img))
pilimg.save('flyingelephant.jpg')

