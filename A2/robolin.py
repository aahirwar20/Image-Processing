# 3rd
import cv2
import numpy as np
from PIL import Image
from skimage import io
import sys

def Hough_trans(image, edge_image, n_rhos=200, n_thetas=200, count=330):
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
  #
  d = pow(np.square(edge_height) + np.square(edge_width),1/2)
  dtheta = 200 / n_thetas
  drho = (2 * d) / n_rhos
  #
  thetas = np.arange(0, 200, step=dtheta)
  rhos = np.arange(-d, d, step=drho)
  #
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  #
  accumulator = np.zeros((len(rhos), len(rhos)))
  #
  
  #
  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho))
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        

  for y in range(1,accumulator.shape[0]-1):
   for x in range(1,accumulator.shape[1]-1):

    if accumulator[y][x] > count :
     if accumulator[y][x] > accumulator[y][x-1] and accumulator[y][x] > accumulator[y][x+1] and accumulator[y][x] > accumulator[y-1][x] and accumulator[y][x] > accumulator[y+1][x]:
       if accumulator[y][x] > accumulator[y-1][x-1] and accumulator[y][x] > accumulator[y+1][x-1] and accumulator[y][x] > accumulator[y-1][x+1] and accumulator[y][x] > accumulator[y+1][x+1]:  
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        image=cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
  return image    

  
  #return accumulator, rhos, thetas
def gaussian_kernel(size):
    sigma=1
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

def gray(image):
 (H,W,f)=image.shape   
 img=np.zeros((H,W))
 for i in range(H):
   for z in range(W):
    for x in range(3):
      img[i][z]+=image[i][z][x]
 img=np.multiply(img,1/3) 
 return img

path=str(sys.argv[1])
image = io.imread(path)
img=gray(image)
g_kernel=gaussian_kernel("3")
img=conv(img,g_kernel)
img = cv2.Canny(img, 100, 200)
image= Hough_trans(image, img)
plimg=Image.fromarray(np.uint8(image))
#plt.imshow(plimg)
plimg.save('enhanced-cctv3.jpg')