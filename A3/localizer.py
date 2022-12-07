#q2
import numpy as np
import sys
from skimage import io

path=str(sys.argv[1])

img= np.array(io.imread(path))
img2=np.copy(img)
b1=0
g1=0
r1=0

H=img.shape[0]
W=img.shape[1]

for i in range(H ):
  for j in range(0,W):
    b1+=img2[i][j][2]
    r1+=img2[i][j][0]
    g1+=img2[i][j][1]
    
    

b1 =b1/(H*W)
r1 =r1/(H*W)
g1 =g1/(H*W)
#print(abs(r1-g1)) 
bp=0
gp=0
rp=0

if(abs(r1-g1)<=9 ):
  bp=bp+1
  
elif(abs(r1-g1)<=13.5 ):
  gp=gp+1
else :
  rp=rp+1

if(abs(r1-b1)<=14 ):
  bp=bp+1
  
elif(abs(r1-b1)<=23.5 ):
  gp=gp+1
  
else :
  rp=rp+1 
 

if(abs(g1-b1)<=7 ):
  bp=bp+1
elif(abs(g1-b1)<=19 ):
  rp=rp+1
else :
  gp=gp+1    
  
if(abs(2*g1-r1-b1)<=3.68 ):
  bp=bp+1
elif(abs(g1-b1)<=16 ):
  rp=rp+1
  
else :
  gp=gp+1 
    

if bp>gp and bp>rp:
  print(1)
elif gp>bp and gp>rp:
  print(2) 
else:
  print(3)   