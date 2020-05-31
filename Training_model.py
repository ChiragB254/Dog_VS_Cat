import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten

from keras.layers import Conv2D,MaxPooling2D   # Conv2D = Convolutional layer
import pickle

import time

pickle_in = open(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\X1.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\Y1.pickle", "rb")
Y = pickle.load(pickle_in)

X = X/255.0  # to scale done the image   (Normalization) 


model = Sequential()


model.add(Conv2D(256,(3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Flatten()) # this convert our 3D feature map to 1D feature vector

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

model.fit(X , Y , batch_size = 4 , epochs = 10 , validation_split = 0.1)

model.save(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\Dog_chr_Cat3.model")
