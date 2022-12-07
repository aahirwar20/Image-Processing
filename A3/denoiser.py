#q1
import numpy as np
import sys
from skimage import io
kr=12
sigma_s=10
sigma_r=20


path=str(sys.argv[1])
img= np.array(io.imread(path))
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
k2=g2(255,sigma_r)
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
  for i in range(H):
    for j in range(W):
      Krf=np.vectorize(rt)
      Kr=Krf(padded_img[i:i+KH,j:j+KW],padded_img[i+int(KH/2)][j+int(KW/2)])
      Ka=np.multiply(Kr,k)    
      F[i, j] = int(np.sum(Ka* padded_img[i:i + KH, j:j + KW])/np.sum(Ka))

  F=F.astype(int)  
  return np.uint8(F) 

H=img.shape[0]
W=img.shape[1]
k=gaussian_kernel(kr,sigma_s)
img2=np.copy(img)
img2b=np.zeros((img.shape[0],img.shape[1]))
img2g=np.copy(img2b)
img2r=np.copy(img2b)
for i in range(H):
  for j in range(W):
    img2r[i][j]=img[i][j][0]
    img2g[i][j]=img[i][j][1]
    img2b[i][j]=img[i][j][2]
img2r=conv(img2r,k,k2)    
img2g=conv(img2g,k,k2)
img2b=conv(img2b,k,k2)
#print(k2)
#img2 = (img2 / img2.max()) * 255
img3=np.copy(img)
for i in range(H):
  for j in range(W):
    img3[i][j][0]=img2r[i][j]
    img3[i][j][1]=img2g[i][j]
    img3[i][j][2]=img2b[i][j]    

from PIL import Image

pilimg = Image.fromarray(np.uint8(img3))
pilimg.save('denoised.jpg')
