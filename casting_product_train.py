#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
import pickle


# In[5]:


train_path="E:/Downloads/casting_product_image/casting_data/casting_data/train/"
test_path="E:/Downloads/casting_product_image/casting_data/casting_data/test/"


# In[6]:


classes=os.listdir(train_path)
print((classes))
no_of_classes=len(classes)


# In[84]:


images=[]
labels=[]
for i in range(0,len(classes)):
    images_list=os.listdir(train_path+classes[i])
    for y in images_list:
        img=cv2.imread(train_path+classes[i]+'/'+y)
        img=cv2.resize(img,(50,50))
        images.append(img)
        labels.append(i)


# In[85]:


x_test=[]
y_test=[]

for i in range(0,len(classes)):
    images_list=os.listdir(test_path+classes[i])
    for z in images_list:
        img=cv2.imread(test_path+classes[i]+'/'+z)
        img=cv2.resize(img,(50,50))
        x_test.append(img)
        y_test.append(i)


# In[86]:


images=np.array(images)
labels=np.array(labels)
x_test=np.array(x_test)
y_test=np.array(y_test)


# In[87]:


print(images.shape)
print(labels.shape)
print(x_test.shape)
print(y_test.shape)


# In[88]:


x_train,x_valid,y_train,y_valid=train_test_split(images,labels,test_size=0.3)


# In[89]:


print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[11]:


def pre_processing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img


x_train=np.array(list(map(pre_processing,x_train)))
x_valid=np.array(list(map(pre_processing,x_valid)))
x_test=np.array(list(map(pre_processing,x_test)))


# In[19]:


x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_valid=x_valid.reshape((x_valid.shape[0],x_valid.shape[1],x_valid.shape[2],1))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))


# In[20]:


print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)


# In[21]:


datagen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
datagen.fit(x_train)


# In[22]:


y_train=to_categorical(y_train,no_of_classes)
y_valid=to_categorical(y_valid,no_of_classes)
y_test=to_categorical(y_test,no_of_classes)


# In[30]:


def my_model():
    no_of_filters=60
    size_of_filter1=(5,5)
    size_of_filter2=(3,3)
    size_of_pool=(2,2)
    no_of_nodes=500
    
    model=Sequential()
    model.add((Conv2D(no_of_filters,size_of_filter1,input_shape=(50,50,1),activation='relu')))
    model.add((Conv2D(no_of_filters,size_of_filter1,activation='relu')))
    model.add((MaxPooling2D(pool_size=size_of_pool)))
    model.add((Conv2D(no_of_filters//2,size_of_filter2,activation='relu')))
    model.add((Conv2D(no_of_filters//2,size_of_filter2,activation='relu')))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(no_of_nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes,activation='softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


# In[37]:


model=my_model()
#model.summary()

history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=50),
                       steps_per_epoch=60,
                       epochs=100,
                       validation_data=(x_valid,y_valid),shuffle=1)


# In[40]:


model.evaluate(x_test,y_test,verbose=0)


# In[67]:


model.save("trained_model.h5")


# In[2]:


model1=load_model('trained_model1.h5')


# In[39]:


out_dict={}
count=0
for i in range(0,len(classes)):
    images_list=os.listdir(test_path+classes[i])
    for z in images_list:
        image_no=z.split('_')[-1]
        image_no=image_no.split(".")[0]
        img=cv2.imread(test_path+classes[i]+'/'+z)
        img=cv2.resize(img,(50,50))
        img=pre_processing(img)
        img=img.reshape(1,50,50,1)
        class_index=int(model1.predict_classes(img))
        out=classes[class_index]+":"+str(image_no)+"\n"
        file=open("Output_results.txt","a+")
        file.write(out)

