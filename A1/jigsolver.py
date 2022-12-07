import numpy as np
import matplotlib.pyplot as plt
import sys
file_path = str(sys.argv[1])
img2 =plt.imread(file_path)
img3=np.copy(img2)
w=img2.shape[1]
h=img2.shape[0]
t=0
sq1=np.copy(img2[0:200,0:190,:])
sq2=np.copy(img2[200:410,0:190,:])
sq3=np.copy(img2[150:330,515:700,:])
sq4=np.copy(img2[370:h,370:w,:])
for i in range(105):
  for j in range(190):
    c=sq2[[i],[j],:]
    sq2[[i],[j],:]=sq2[[210-i-1],[j],:]
    sq2[[210-i-1],[j],:]=c

for i in range(25):
  for j in range(w-370):
    c=sq4[[i],[j],:]
    sq4[[i],[j],:]=sq4[[51-i-1],[j],:]
    sq4[[51-i-1],[j],:]=c

for i in range(180):
  for j in range(92):
    c=sq3[[i],[j],:]
    sq3[[i],[j],:]=sq3[[i],[185-j-1],:]
    sq3[[i],[185-j-1],:]=c 
    
bg=sq1[:,:,[1]]
sq1[:,:,[1]]= sq1[:,:,[2]]   
sq1[:,:,[2]]=bg

img3[200:400,0:190,:]=sq1
img3[0:210,0:190,:]=sq2
img3[150:330,515:700,:]=sq3

img3[370:h,370:w,:]=sq4
pad1=img3[395:400,:190,]
pad2=img3[410:415,:190,]
for i in range(3):
  for j in range(190):
    c1=pad1[[i],[j]]
    pad1[[i],[j]]=pad1[[4-i],[j]]
    pad1[[4-i],[j]]=c1
    c2=pad2[[i],[j]]
    pad2[[i],[j]]=pad2[[4-i],[j]]
    pad2[[4-i],[j]]=c2

img3[400:405,:190]=pad1
img3[405:410,:190]=pad2    
# plt.imshow(img3)

# plt.axis("off") 
# plt.show()
from PIL import Image
pilimg = Image.fromarray(np.uint8(img3))
pilimg.save('jigsaw.jpg')

