import os
import numpy as np
import matplotlib.pyplot as plt


infile=open("10.pgm",'rb')
header=infile.readline()
width,height, maxval = [int(item) for item in header.split()[1:]]
infile.seek(len(header))
img=np.fromfile(infile, dtype=np.uint8).reshape(height, width)
print(img)
np.savetxt("10.txt",img)
        
        
