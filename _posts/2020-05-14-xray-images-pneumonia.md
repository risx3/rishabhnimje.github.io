---
title: "Chect X-ray Images (Pneumonia)"
date: 2020-05-14
tags: [Kaggle, Keras, Machine Learning, Neural Network]
excerpt: "Identify the x-rays with pneumonia"
header:
  overlay_image: "/images/xray-images-pneumonia/home-page.jpeg"
mathjax: "true"
---

## Overview

Automated methods to detect and classify human diseases from medical images.

## Data Description

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Files

* test
1. NORMAL
2. PNEUMONIA
* train
1. NORMAL
2. PNEUMONIA
* val
1. NORMAL
2. PNEUMONIA

## So let's begin here...

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img

import os
```

> Using TensorFlow backend.

```python
mainDIR = os.listdir('../input/chest-xray-pneumonia/chest_xray')
print(mainDIR)
```

> ['val', '__MACOSX', 'chest_xray', 'train', 'test']

## Load Data

```python
train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'
val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'
test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'
```

### Train Data
```python
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
```

```python
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
```

    1341
    normal picture title:  IM-0481-0001.jpeg
    pneumonia picture title: person428_virus_876.jpeg
    Text(0.5, 1.0, 'Pneumonia')


![png](/images/xray-images-pneumonia/notebook_4_2.png)

## Defining Model

```python
cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Conv2D(32, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Flatten())

cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'sigmoid', units = 1))
```

## Compile Model

```python
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

```python
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

```

> Found 5216 images belonging to 2 classes.<br>
> Found 16 images belonging to 2 classes.<br>
> Found 624 images belonging to 2 classes.

### Model Summary

```python
cnn.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 29, 29, 32)        9248      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 6272)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               802944    
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 813,217
    Trainable params: 813,217
    Non-trainable params: 0
    _________________________________________________________________

## Fit Model

```python
cnn_model = cnn.fit_generator(training_set,
                         steps_per_epoch = 163,
                         epochs = 10,
                         validation_data = validation_generator,
                         validation_steps = 624)
```

    Epoch 1/10
    163/163 [==============================] - 206s 1s/step - loss: 0.3757 - accuracy: 0.8391 - val_loss: 0.4808 - val_accuracy: 0.7500
    Epoch 2/10
    163/163 [==============================] - 532s 3s/step - loss: 0.2294 - accuracy: 0.9045 - val_loss: 0.7449 - val_accuracy: 0.5625
    Epoch 3/10
    163/163 [==============================] - 186s 1s/step - loss: 0.2025 - accuracy: 0.9206 - val_loss: 0.4746 - val_accuracy: 0.6250
    Epoch 4/10
    163/163 [==============================] - 186s 1s/step - loss: 0.1861 - accuracy: 0.9241 - val_loss: 0.5135 - val_accuracy: 0.6875
    Epoch 5/10
    163/163 [==============================] - 187s 1s/step - loss: 0.1759 - accuracy: 0.9293 - val_loss: 0.5023 - val_accuracy: 0.7500
    Epoch 6/10
    163/163 [==============================] - 188s 1s/step - loss: 0.1562 - accuracy: 0.9411 - val_loss: 0.4430 - val_accuracy: 0.6875
    Epoch 7/10
    163/163 [==============================] - 188s 1s/step - loss: 0.1474 - accuracy: 0.9434 - val_loss: 0.2139 - val_accuracy: 1.0000
    Epoch 8/10
    163/163 [==============================] - 186s 1s/step - loss: 0.1521 - accuracy: 0.9419 - val_loss: 0.2099 - val_accuracy: 1.0000
    Epoch 9/10
    163/163 [==============================] - 187s 1s/step - loss: 0.1384 - accuracy: 0.9450 - val_loss: 0.3839 - val_accuracy: 0.8125
    Epoch 10/10
    163/163 [==============================] - 187s 1s/step - loss: 0.1323 - accuracy: 0.9480 - val_loss: 0.7050 - val_accuracy: 0.6250

## Evaluate Model

```python
test_accu = cnn.evaluate_generator(test_set,steps=624)
print('The testing accuracy is :',test_accu[1]*100, '%')
```

> The testing accuracy is : 84.15673971176147 %

### Accuracy

```python
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
```

![png](/images/xray-images-pneumonia/notebook_12_0.png)

### Loss

```python
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()
```

![png](/images/xray-images-pneumonia/notebook_13_0.png)