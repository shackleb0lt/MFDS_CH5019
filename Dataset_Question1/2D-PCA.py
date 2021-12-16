import os
import sys
import numpy as np
import matplotlib.pyplot as plt

images=[]


def write_image(image,file_name,height,width):
    
    #storing the matrix data into a 1-D array
    buff=np.array('B')
    image=np.array(image,dtype=np.uint8).reshape(height*width)
    
    
    #opening file
    try:
      fout=open(file_name, 'wb')
    except IOError:
      print ("Cannot open",file_name)
      sys.exit()
      
    # define PGM Header
    pgmHeader = 'P5' + '\n' + str(width) + '  ' + str(height) + '  ' + str(255) + '\n'
    pgmHeader= bytes(pgmHeader, 'ascii')
    # write the header to the file
    fout.write(pgmHeader)

    # write the data to the file 
    buff.tofile(fout)
    image.tofile(fout)

    # close the file
    fout.close()
    
def read_image(file_name):
    
    #opening file
    try:
      infile=open(file_name, 'rb')
    except IOError:
      print ("Cannot open",file_name)
      sys.exit()
      
    #reading and processing file details
    header = infile.readline()
    global width,height
    width, height, maxval = [int(item) for item in header.split()[1:]]

    #reading the image into a matrix of unsigned 8 bit data type
    infile.seek(len(header))
    img = np.fromfile(infile,dtype=np.uint8)

    #closing file
    infile.close()
    
    #returning the matrix 
    return img

def shift(V):
    temp=(V-V.mean())/V.std()
    return temp
#getting the directory location
location=os.getcwd()
location=location+"/Dataset_Question1/"

#accessing each pgm file
for per in range(1,16):
    for i in range(1,11):
        name=location+str(per)+"/"+str(i)+".pgm"
        img=read_image(name)
        images.append(img)
        


images=np.array(images,dtype=int).reshape(15,10,width,height)
images_shift=images
for i in range(15):
    for j in range(10):
        images_shift[i][j]=shift(images[i][j])
        
mean=np.zeros((64,64))   
for i in range(15):
    for j in range(4):
        mean=np.add(mean,images_shift[i][j])
mean=mean/75

U,S,Vt=np.linalg.svd(mean)
k=min(width,height)
A=[]
A.append(S[0])
for i in range(1,k):
    A.append(A[i-1]+S[i])
j=0
while  A[j]/A[k-1] <0.97:
    j+=1
r=j+1
c=j+1


V=Vt.transpose()
U_1=U[:,:r]
V_1=V[:,:c]
U_1_t=U_1.transpose()


new_images=np.zeros((15,10,r,c))
for i in range(15):
    for j in range(10):
        new_images[i][j]=np.dot(U_1.transpose(),np.dot(images_shift[i][j],V_1))
    
total=150
match=0
nm=0

for i in range(15):
    for j in range(4,10):
        min_cov=1000000000
        m=-1
        
        for k in range(15):
            for l in range(4):
                curr=np.linalg.norm(new_images[i][j]-new_images[k][l])
                if curr<min_cov:
                    m=k
                    min_cov=curr
        if m==i:
            match+=1
        else:
            nm+=1        
print(match,nm)



