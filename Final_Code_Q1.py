import os
import sys
import numpy as np
'''
CH5019:Mathematical Foundations of Data Science
Term Project          January-May 2020          Group-9
                                     Question 1
Paste the code in Term_Project folder and run it,
Representative images are stored in Dataset_Q1  folder,
'''
class Read_Write:

    def __init__(self,location):
        self.loc=location
        self.width=0
        self.height=0

    def read_image(self,file_name):
        try:
            infile=open(file_name, 'rb')
        except IOError:
            print ("Cannot open",file_name)
            sys.exit()

        # reading and processing file details
        header = infile.readline()
        
        self.width, self.height, maxval = [int(item) for item in header.split()[1:]]

        #reading the image into a matrix of unsigned 8 bit data type
        infile.seek(len(header))
        img = np.fromfile(infile,dtype=np.uint8).reshape(self.height,self.width)

        #closing file
        infile.close()

        #returning the matrix 
        return img

    def write_image(self,image,i):
        image=-1*image
        file_name=self.loc+str(i)+"_rep.pgm"
        #storing the matrix data into a 1-D array
        buff=np.array('B')
        image=np.array(image,dtype=np.uint8).reshape(self.height*self.width)
        
        
        #opening file
        try:
            fout=open(file_name, 'wb')
        except IOError:
            print ("Cannot open",file_name)
            sys.exit()
        

        # define PGM Header
        pgmHeader = 'P5' + '\n' + str(self.width) + '  ' + str(self.height) + '  ' + str(255) + '\n'
        pgmHeader= bytes(pgmHeader, 'ascii')

        # write the header to the file
        fout.write(pgmHeader)

        # write the data to the file 
        buff.tofile(fout)
        image.tofile(fout)

        # close the file
        fout.close()

class Images:
    def __init__(self,images,width,height):
        
        self.images=np.array(images,dtype=int).reshape(15,10,width*height)
        self.rep_img=np.zeros((15,height*width))

        self.width=width
        self.height=height

    def shift_all(self):
        self.images_shifted=np.zeros((15,10,self.width*self.height))
        for i in range(15):
            for j in range(10):
                self.images_shifted[i][j]=np.array(self.shift(i,j))

    def shift(self,i,j):
        temp=np.array(self.images[i][j])
        temp= (temp-temp.min())/(temp.max()-temp.min())
        return temp
    
    def compute_rep(self):
        #pca=np.zeros((10,self.width*self.height))
        for i in range(15):
            pca=np.array(self.images_shifted[i])
            U,S,Vt=np.linalg.svd(pca)
            print(U.shape, Vt.shape)
            self.rep_img[i]=np.dot(pca.transpose(),S[0]*U[:,0]+S[1]*U[:,1]+S[2]*U[:,2])
    
    def match_faces(self):
        match=0
        non_match=0
        for i in range(15):
            for j in range(10):
                min_cov=10000000000
                m=-1
                for k in range(15):
                    curr=np.linalg.norm(self.images_shifted[i][j]-self.rep_img[k])
                    if curr<min_cov:
                        m=k
                        min_cov=curr
                if m==i:
                    match+=1
                else:
                    non_match+=1
        return match,non_match



#getting the directory location
location=os.getcwd()
location=location+"/Dataset_Question1/"
Bob=Read_Write(location)

images=[]
#accessing each pgm file
for per in range(1,16):
    for i in range(1,11):
        name=location+str(per)+"/"+str(i)+".pgm"
        img=Bob.read_image(name)
        images.append(img)

#creating an pbject image
images=Images(images,Bob.width,Bob.height)
images.shift_all()

#computing the represesntative images
images.compute_rep()

#Writing the representative images into pgm files
i=1
for img in images.rep_img:
    Bob.write_image(img,i)
    i+=1
match,non_match=images.match_faces()
print(match,"faces out of 150 match with thier respective subjects")
print("efficiency of",match*100/150,"%")





