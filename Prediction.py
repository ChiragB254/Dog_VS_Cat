import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpim

CATEGORIES = ['Dog', 'Cat']

# Image which u want to check
IMAGE = r"E:\ML_Material\Projects\Project_1\kagglecatsanddogs_3367a\PetImages\Cat\2.jpg"

def prepare(filepath):
    img_size = 150
    img_array = cv2.imread(IMAGE,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)
'''
from keras.model import load_model
model =load_model(r"E:\ML_Material\Project\Project_1\kagglecatsanddogs_3367a\Dog_Vs_Cat.model")

# or we can use this
'''
model = tf.keras.models.load_model(
    r"E:\ML_Material\Projects\Project_1\Dog_vs_Cat\train\Dog_ch_Cat3.model")

prediction = model.predict([prepare(IMAGE)])
print(CATEGORIES[int(prediction[0][0])])


img = cv2.imread(IMAGE)
imgplot = plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.axis('off')
plt.show()
