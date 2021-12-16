import os
import sys
import numpy as np
'''
CH5019:Mathematical Foundations of Data Science
Term Project,January-May 2020,Group-9
Question 1
Paste the code in Term_Project folder and run it,
Representative images are stored in Dataset_Q1  folder,
'''
images=[]                                    #Array containg all the input images

def vector_shift(V):
    return (V-V.mean())/V.std()


#function to write a matrix into a pgm image
def write_image(image,file_name,width,height):
    
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


#function to reaed pgm file
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
    img = np.fromfile(infile,dtype=np.uint8).reshape(height, width)

    #closing file
    infile.close()
    
    #returning the matrix 
    return img
    

#getting the directory location
location=os.getcwd()
location=location+"/Dataset_Question1/"

#accessing each pgm file
for per in range(1,16):
    for i in range(1,11):
        name=location+str(per)+"/"+str(i)+".pgm"
        img=read_image(name)
        images.append(img)

   
images=np.array(images,dtype=int).reshape(15,10,width*height)       #reshaping the 3-D array into 4-D

#initializing the pca and representative matrices
rep=np.zeros((15,64*64))
pca=np.zeros((10,64*64))

#Computing the representative images
for i in range(15):
    for j in range(10):
        pca[j]=vector_shift(images[i][j])
        
    U,S,Vt=np.linalg.svd(pca)
    V=Vt.transpose()
    rep[i]=-1*np.dot(pca.transpose(),U[:,0]+U[:,1]+U[:,2])
    name=location+str(i)+"_rep.pgm"
    write_image(rep[i],name,width,height)                        #Storing the representative images for each subject
            

#Comparing all images with their representative images 
match=0
nm=0
for i in range(15):
    for j in range(10):
        min_cov=1000000000
        m=-1
        V=vector_shift(images[i][j])
        for k in range(15):
            curr=np.linalg.norm(rep[k]-V) #taking norm of the difference in representative image and test image
            if curr<min_cov:
                m=k
                min_cov=curr
        if m==i:
            match+=1
        else:
            nm+=1       
print(match,nm)  #Printing the number of matching and non-matching images
