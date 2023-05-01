
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing import image
# %matplotlib inline

"""Importing Dataset"""

from google.colab import drive
drive.mount('/content/drive')

dir="/content/drive/My Drive/facemaskdetector/dataset"
categories=["with_mask","without_mask"]
dataset=[]
for category in categories: 
    path = os.path.join(dir,category) 
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        new_array=cv2.resize(img_array,(64,64))
        label=categories.index(category)
        dataset.append([new_array,label])

"""Building Dataset"""

random.shuffle(dataset)

for data in dataset[:10]:
    print(data[1])

X = []
y = []
for features,label in dataset:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,64,64, 1)
for x in X:
  x=x/255.0

"""Building Model"""

model=Sequential()

model.add(Conv2D(32,(3,3),padding="same",strides=(1,1),input_shape=(64,64,1),activation="relu"))
model.add(Conv2D(32,(3,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),padding="same"))
model.add(Conv2D(16,(3,3),activation="relu"))
model.add(Conv2D(16,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=10,batch_size=32,validation_split=0.3)

"""Predicting """

def cascade(test_image):
   haar_cascade_face =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
   faces_rects = haar_cascade_face.detectMultiScale(test_image, scaleFactor = 1.2, minNeighbors = 5);
   print('Faces found: ', len(faces_rects))
   for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 5)
     font = cv2.FONT_HERSHEY_SIMPLEX
     text = cv2.putText(test_image,print(output),(x,y-10), font, 10,(0,255,0),10)
   return plt.imshow(test_image)

def predict(output):
   if output[0][0]>=0.5:
     prediction = 'without mask'
     print(prediction )
   else:
     prediction = 'with mask'
     print(prediction )

test_image=cv2.imread("/content/drive/My Drive/facemaskdetector/dataset/with_mask/0-with-mask.jpg")
tested_image = cv2.imread("/content/drive/My Drive/facemaskdetector/dataset/with_mask/0-with-mask.jpg",0)
tested_image = image.img_to_array(tested_image)
tested_image=cv2.resize(tested_image,(64,64))
tested_image = np.expand_dims(tested_image,axis = 0)
tested_image=tested_image.reshape(-1,64,64,1)
output = model.predict(tested_image)
predict(output)
cascade(test_image)

test_image=cv2.imread("/content/drive/My Drive/facemaskdetector/dataset/without_mask/0.jpg")
tested_image = cv2.imread("/content/drive/My Drive/facemaskdetector/dataset/without_mask/0.jpg",0)
tested_image = image.img_to_array(tested_image)
tested_image=cv2.resize(tested_image,(64,64))
tested_image = np.expand_dims(tested_image,axis = 0)
tested_image=tested_image.reshape(-1,64,64,1)
output = model.predict(tested_image)
predict(output)
cascade(test_image)

test_image=cv2.imread("/content/drive/My Drive/facemaskdetector/faceimg.png")
tested_image = cv2.imread("/content/drive/My Drive/facemaskdetector/faceimg.png",0)
tested_image = image.img_to_array(tested_image)
tested_image=cv2.resize(tested_image,(64,64))
tested_image = np.expand_dims(tested_image,axis = 0)
tested_image=tested_image.reshape(-1,64,64,1)
output = model.predict(tested_image)
predict(output)
cascade(test_image)

test_image=cv2.imread("/content/drive/My Drive/facemaskdetector/facemask.jpg")
tested_image = cv2.imread("/content/drive/My Drive/facemaskdetector/facemask.jpg",0)
tested_image = image.img_to_array(tested_image)
tested_image=cv2.resize(tested_image,(64,64))
tested_image = np.expand_dims(tested_image,axis = 0)
tested_image=tested_image.reshape(-1,64,64,1)
output = model.predict(tested_image)
predict(output)
cascade(test_image)
