import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

DataSer = r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train/"

CATAGORIES = ["Dog","Cat"]

''''
for i in CATAGORIES:
    path = os.path.join(DataSer,i)
    for img in os.listdir(path):
        img_arry = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arry,cmap = "gray")
        plt.show()
        break
    break

IMG_SIZE = 150

new_array = cv2.resize(img_arry,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap = "gray")
plt.show()

training_data = []

def creat_training_data():
    for category in CATAGORIES : #do dogs and cats
        path= os.path.join(DataSer,category)  #create path to dogs and cats
        class_num = CATAGORIES.index(category)  #get classifications (0 or 1). 0 = dog , 1 = cat

        for img in os.listdir(path) :  # iterate over each image perdogsand cats
            try :
                img_arry = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  #convert to arry
                new_arry = cv2.resize(img_arry,(IMG_SIZE,IMG_SIZE))  # resize to normalize data size
                training_data.append([new_arry,class_num])  #add this to our training_data
            
            except Exception as e: # in the interest in keeping the output clean . . .
                pass
            #ex



creat_training_data()

print(len(training_data))

import random

random.shuffle(training_data)
for sample in training_data[:20]:
    print(sample)


X = []
Y = []
for features,lable in training_data:
    X.append(features)
    Y.append(lable)
print(X[0].reshape(-1,IMG_SIZE,IMG_SIZE,1))

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)


import pickle

# pickle is use to dump the data 
# Everytime we don't want to load and reshape data again and again
# using pickle we store the data here to use it efficently
 
pickle_out = open(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\X1.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\Y1.pickle", "wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

''''


