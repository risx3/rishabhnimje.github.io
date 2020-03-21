---
title: "Dogs vs Cats"
date: 2020-03-21
tags: [Kaggle, Image Classification, Keras, Machine Learning, CNN]
excerpt: "Classify whether images contain either a dog or a cat"
header:
  overlay_image: "/images/dogs-vs-cats/home-page.jpg"
  caption: "Photo by Tran Mau Tri Tam on Unsplash"
mathjax: "true"
---

## Overview

## Data Description

## Files

## So let's begin here...

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import cv2
from zipfile import ZipFile
import os
```

    Using TensorFlow backend.
    


```python
file_name = "/kaggle/input/dogs-vs-cats/train.zip"
file_test = "/kaggle/input/dogs-vs-cats/test1.zip"

with ZipFile(file_name, 'r') as zip:
    zip.extractall('/train')
    print('Train Extract Done!') 
    
with ZipFile(file_test,'r') as zip:
    zip.extractall('/test1')
    print('Test Extract Done!')
```

    Train Extract Done!
    Test Extract Done!
    


```python
print('Train Data ',len(os.listdir('/train/train/')))
print('Test Data ',len(os.listdir('/test1/test1/')))
```

    Train Data  25000
    Test Data  12500
    


```python
train_path = '/train/train/'
for p in os.listdir(train_path):
    category = p.split(".")[0]
    img_array = cv2.imread(os.path.join(train_path,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(80, 80))
    plt.imshow(new_img_array,cmap="gray")
    break
```


![png](notebook_files/notebook_3_0.png)



```python
X_train = []
y_train = []
convert = lambda category : int(category == 'dog')
def create_train_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_train.append(new_img_array)
        y_train.append(category)
create_train_data(train_path)
X_train = np.array(X_train).reshape(-1, 80,80,1)
y_train = np.array(y_train)
X_train = X_train/255.0
```


```python
model = Sequential()

model.add(Conv2D(16,(3,3), activation = 'relu', input_shape = X_train.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 78, 78, 16)        160       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 39, 39, 16)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 37, 37, 32)        4640      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 18, 18, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 16, 16, 64)        18496     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 6, 6, 128)         73856     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 3, 3, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               590336    
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 688,001
    Trainable params: 688,001
    Non-trainable params: 0
    _________________________________________________________________
    


```python
history = model.fit(X_train, y_train, epochs=20, batch_size=200, validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/20
    20000/20000 [==============================] - 7s 330us/step - loss: 0.6698 - accuracy: 0.5849 - val_loss: 0.6291 - val_accuracy: 0.6508
    Epoch 2/20
    20000/20000 [==============================] - 2s 101us/step - loss: 0.5790 - accuracy: 0.6973 - val_loss: 0.5370 - val_accuracy: 0.7300
    Epoch 3/20
    20000/20000 [==============================] - 2s 103us/step - loss: 0.4898 - accuracy: 0.7671 - val_loss: 0.4684 - val_accuracy: 0.7792
    Epoch 4/20
    20000/20000 [==============================] - 2s 107us/step - loss: 0.4322 - accuracy: 0.8031 - val_loss: 0.4145 - val_accuracy: 0.8112
    Epoch 5/20
    20000/20000 [==============================] - 2s 103us/step - loss: 0.3896 - accuracy: 0.8239 - val_loss: 0.3974 - val_accuracy: 0.8194
    Epoch 6/20
    20000/20000 [==============================] - 2s 102us/step - loss: 0.3651 - accuracy: 0.8400 - val_loss: 0.3800 - val_accuracy: 0.8264
    Epoch 7/20
    20000/20000 [==============================] - 2s 103us/step - loss: 0.3197 - accuracy: 0.8599 - val_loss: 0.3781 - val_accuracy: 0.8282
    Epoch 8/20
    20000/20000 [==============================] - 2s 105us/step - loss: 0.2928 - accuracy: 0.8730 - val_loss: 0.3793 - val_accuracy: 0.8288
    Epoch 9/20
    20000/20000 [==============================] - 2s 107us/step - loss: 0.2525 - accuracy: 0.8952 - val_loss: 0.3691 - val_accuracy: 0.8414
    Epoch 10/20
    20000/20000 [==============================] - 2s 105us/step - loss: 0.2167 - accuracy: 0.9102 - val_loss: 0.4136 - val_accuracy: 0.8200
    Epoch 11/20
    20000/20000 [==============================] - 2s 103us/step - loss: 0.2141 - accuracy: 0.9100 - val_loss: 0.3947 - val_accuracy: 0.8280
    Epoch 12/20
    20000/20000 [==============================] - 2s 101us/step - loss: 0.1640 - accuracy: 0.9344 - val_loss: 0.4463 - val_accuracy: 0.8254
    Epoch 13/20
    20000/20000 [==============================] - 2s 114us/step - loss: 0.1256 - accuracy: 0.9527 - val_loss: 0.4623 - val_accuracy: 0.8372
    Epoch 14/20
    20000/20000 [==============================] - 2s 104us/step - loss: 0.1251 - accuracy: 0.9520 - val_loss: 0.4947 - val_accuracy: 0.8374
    Epoch 15/20
    20000/20000 [==============================] - 2s 115us/step - loss: 0.0827 - accuracy: 0.9711 - val_loss: 0.6302 - val_accuracy: 0.8252
    Epoch 16/20
    20000/20000 [==============================] - 2s 104us/step - loss: 0.0628 - accuracy: 0.9792 - val_loss: 0.6034 - val_accuracy: 0.8312
    Epoch 17/20
    20000/20000 [==============================] - 2s 106us/step - loss: 0.0458 - accuracy: 0.9844 - val_loss: 0.6821 - val_accuracy: 0.8302
    Epoch 18/20
    20000/20000 [==============================] - 2s 104us/step - loss: 0.0349 - accuracy: 0.9898 - val_loss: 0.6707 - val_accuracy: 0.8360
    Epoch 19/20
    20000/20000 [==============================] - 2s 105us/step - loss: 0.0253 - accuracy: 0.9934 - val_loss: 0.8074 - val_accuracy: 0.8264
    Epoch 20/20
    20000/20000 [==============================] - 2s 110us/step - loss: 0.0310 - accuracy: 0.9904 - val_loss: 0.9238 - val_accuracy: 0.8246
    


```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
```


![png](notebook_files/notebook_7_0.png)



    <Figure size 432x288 with 0 Axes>



```python
test_path = "/test1/test1/"

X_test = []
id_line = []
def create_test_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)
create_test_data(test_path)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255
```


```python
predictions = model.predict(X_test)
predicted_val = [int(round(p[0])) for p in predictions]
```


```python
submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})
submission_df.to_csv("submission.csv", index=False)
```
