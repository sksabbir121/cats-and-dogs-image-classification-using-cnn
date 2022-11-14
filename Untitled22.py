#!/usr/bin/env python
# coding: utf-8

# In[1]:


# library
# tensorflow: fast numerical computing 
# os: functions for interacting with the operating system
# NumPy is a Python library used for working with arrays


# In[2]:


import tensorflow as tf
import os
import numpy as np


# In[ ]:


# Database file path


# In[2]:


base_dir=r"D:\cse 475\Database\testing"


# In[ ]:


# tf.keras.preprocessing.image.ImageDataGenerator:augment images in real-time while model is still training
# Flip Horizontally command reverses the active layer horizontally.That is from left to right.
# validation_split allows users to split their data into training and testing sets.


# In[3]:


IMAGE_SIZE=224
BATCH_SIZE=64

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    
    
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)

validation_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    
    validation_split=0.1
)


# In[ ]:


# For training there are 902 images
# For validation there are 100 images
# 2 classes


# In[4]:


train_genarator=train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
    

) 

validation_generator=validation_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
    

) 


# In[ ]:


# library
# Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
# glob module is used to retrieve pathnames matching a specified pattern.
# Flatten:This function converts the multi-dimensional arrays into flattened one-dimensional arrays or single-dimensional arrays.


# In[5]:


from tensorflow.keras.layers import Input,Flatten,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from glob import glob


# In[ ]:


#Imagenet:labeling and categorizing images 


# In[6]:


IMAGE_SIZE=[224,224]
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
vgg.output


# In[7]:


for layer in vgg.layers:
    layer.trainable=False


# In[8]:


folders=glob(r"D:\cse 475\A Database of Leaf Images\Mango (P0)\*")
print(len(folders))


# In[ ]:


#Dense Layer is used to classify image based on output from convolutional layers.
#Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities.


# In[9]:


x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=vgg.input,outputs=prediction)
model.summary()


# In[10]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


epoch=1

history=model.fit(train_genarator,
                  steps_per_epoch=len(train_genarator),
                  epochs=epoch,
                  validation_data=validation_generator,
                  validation_steps=len(validation_generator)
                 )


# In[ ]:


# library
#Converts a PIL Image instance to a Numpy array.
#Insert a new axis that will appear at the axis position in the expanded array shape


# In[ ]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

img_pred=image.load_img(r"D:\Cat vs Dog\Dog_test\588.JPG",target_size=(224,224))

img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred, axis=0)


rslt= model.predict(img_pred)

print(rslt)
if rslt[0][0]>rslt[0][1]:
    prediction="Cat"   
else:
    prediction="Dog"
print(prediction)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

