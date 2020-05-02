---
title: "Digit Recognizer"
date: 2019-10-19
tags: [Kaggle, Digit Recognition, Keras, Machine Learning, ]
excerpt: "Learn computer vision fundamentals with the famous MNIST data"
header:
  overlay_image: "/images/digit-recognizer/home-page.png"
  caption: #
mathjax: "true"
---

## Overview

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.<br>
In this project, our goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.

## Data Description

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

  000 001 002 003 ... 026 027<br>
  028 029 030 031 ... 054 055<br>
  056 057 058 059 ... 082 083<br>
   |   |   |   |  ...  |   |<br>
  728 729 730 731 ... 754 755<br>
  756 757 758 759 ... 782 783<br>

The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:

ImageId,Label<br>
1,3<br>
2,7<br>
3,8<br>
(27997 more lines)<br>

The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

## Files

* test.csv
* train.csv

You can find the dataset [here](https://www.kaggle.com/c/digit-recognizer/data).


## So let's begin here...


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input/digit-recognizer"))
```

> ['sample_submission.csv', 'test.csv', 'train.csv']

```python
FAST_RUN=False
batch_size=32
epochs=100
if FAST_RUN:
    epochs=1
```

## Load data
Data input train and test data


```python
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
```

## Data exploration

**Columns**

```python
print(train_data.columns)
```

    Index(['label', 'pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5',
           'pixel6', 'pixel7', 'pixel8',
           ...
           'pixel774', 'pixel775', 'pixel776', 'pixel777', 'pixel778', 'pixel779',
           'pixel780', 'pixel781', 'pixel782', 'pixel783'],
          dtype='object', length=785)

Our data have label and pixels column where label represent the digit and pixel presents pixel of image.

**Show Image**<br>
Lets form images from pixel data

```python
def show_image(train_image, label, index):
    image_shaped = train_image.values.reshape(28,28)
    plt.subplot(3, 6, index+1)
    plt.imshow(image_shaped, cmap=plt.cm.gray)
    plt.title(label)


plt.figure(figsize=(18, 8))
sample_image = train_data.sample(18).reset_index(drop=True)
for index, row in sample_image.iterrows():
    label = row['label']
    image_pixels = row.drop('label')
    show_image(image_pixels, label, index)
plt.tight_layout()
```

![png](/images/digit-recognizer/notebook_10_0.png)

## Pre-process Data
We split data for test and train

```python
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

x = train_data.drop(columns=['label']).values.reshape(train_data.shape[0],28,28,1)
y = to_categorical(train_data['label'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
```
> Using TensorFlow backend.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1, # Randomly zoom image
    width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1 # randomly shift images vertically (fraction of total height)
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
train_datagen.fit(x_test)
validation_generator = validation_datagen.flow(
    x_test,
    y_test
)
```

## Define Model

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
```

## Compile Model

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Callbacks

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
```
## Fit model

```python
model.fit_generator(
    train_generator, 
    steps_per_epoch=len(x_train) // batch_size, 
    validation_data=validation_generator,
    validation_steps=len(x_test) // batch_size,
    epochs=epochs,
    callbacks=callbacks
)
```
    Epoch 1/100
    1181/1181 [==============================] - 20s 17ms/step - loss: 0.3980 - accuracy: 0.8756 - val_loss: 0.0356 - val_accuracy: 0.9697
    
    Epoch 00001: val_loss improved from inf to 0.03563, saving model to model.h5
    Epoch 2/100
    1181/1181 [==============================] - 17s 14ms/step - loss: 0.1723 - accuracy: 0.9492 - val_loss: 0.0228 - val_accuracy: 0.9777
    
    Epoch 00002: val_loss improved from 0.03563 to 0.02276, saving model to model.h5
    Epoch 3/100
    1181/1181 [==============================] - 16s 13ms/step - loss: 0.1320 - accuracy: 0.9618 - val_loss: 0.0169 - val_accuracy: 0.9834
    
    Epoch 00003: val_loss improved from 0.02276 to 0.01686, saving model to model.h5
    Epoch 4/100
    1181/1181 [==============================] - 16s 13ms/step - loss: 0.1109 - accuracy: 0.9674 - val_loss: 0.0058 - val_accuracy: 0.9760
    
    Epoch 00004: val_loss improved from 0.01686 to 0.00576, saving model to model.h5
    Epoch 5/100
    1181/1181 [==============================] - 16s 14ms/step - loss: 0.1041 - accuracy: 0.9697 - val_loss: 0.0063 - val_accuracy: 0.9837
    
    Epoch 00005: val_loss did not improve from 0.00576
    Epoch 6/100
    1181/1181 [==============================] - 16s 13ms/step - loss: 0.0930 - accuracy: 0.9722 - val_loss: 0.0047 - val_accuracy: 0.9846
    
    Epoch 00006: val_loss improved from 0.00576 to 0.00466, saving model to model.h5
    Epoch 7/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0925 - accuracy: 0.9726 - val_loss: 0.0150 - val_accuracy: 0.9858
    
    Epoch 00007: val_loss did not improve from 0.00466
    Epoch 8/100
    1181/1181 [==============================] - 16s 13ms/step - loss: 0.0903 - accuracy: 0.9741 - val_loss: 0.0624 - val_accuracy: 0.9854
    
    Epoch 00008: val_loss did not improve from 0.00466
    Epoch 9/100
    1181/1181 [==============================] - 16s 14ms/step - loss: 0.0873 - accuracy: 0.9750 - val_loss: 0.0058 - val_accuracy: 0.9810
    
    Epoch 00009: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    
    Epoch 00009: val_loss did not improve from 0.00466
    Epoch 10/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0744 - accuracy: 0.9786 - val_loss: 0.0115 - val_accuracy: 0.9866
    
    Epoch 00010: val_loss did not improve from 0.00466
    Epoch 11/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0696 - accuracy: 0.9796 - val_loss: 0.0019 - val_accuracy: 0.9866
    
    Epoch 00011: val_loss improved from 0.00466 to 0.00186, saving model to model.h5
    Epoch 12/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0674 - accuracy: 0.9805 - val_loss: 0.0052 - val_accuracy: 0.9873
    
    Epoch 00012: val_loss did not improve from 0.00186
    Epoch 13/100
    1181/1181 [==============================] - 16s 14ms/step - loss: 0.0680 - accuracy: 0.9790 - val_loss: 0.0015 - val_accuracy: 0.9868
    
    Epoch 00013: val_loss improved from 0.00186 to 0.00152, saving model to model.h5
    Epoch 14/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0672 - accuracy: 0.9799 - val_loss: 0.1304 - val_accuracy: 0.9880
    
    Epoch 00014: val_loss did not improve from 0.00152
    Epoch 15/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0687 - accuracy: 0.9802 - val_loss: 0.0011 - val_accuracy: 0.9875
    
    Epoch 00015: val_loss improved from 0.00152 to 0.00113, saving model to model.h5
    Epoch 16/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0671 - accuracy: 0.9805 - val_loss: 0.0344 - val_accuracy: 0.9863
    
    Epoch 00016: val_loss did not improve from 0.00113
    Epoch 17/100
    1181/1181 [==============================] - 16s 14ms/step - loss: 0.0650 - accuracy: 0.9807 - val_loss: 0.0010 - val_accuracy: 0.9880
    
    Epoch 00017: val_loss improved from 0.00113 to 0.00103, saving model to model.h5
    Epoch 18/100
    1181/1181 [==============================] - 15s 13ms/step - loss: 0.0678 - accuracy: 0.9807 - val_loss: 0.0179 - val_accuracy: 0.9861
    
    Epoch 00018: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    
    Epoch 00018: val_loss did not improve from 0.00103
    Epoch 19/100
     898/1181 [=====================>........] - ETA: 3s - loss: 0.0631 - accuracy: 0.9809

## Evaluate Model

```python
x_test_recaled = (x_test.astype("float32") / 255)
scores = model.evaluate(x_test_recaled, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
```

> accuracy: 98.71%<br>
> loss: 4.27%

## Prediction

```python
test_digit_data = test_data.values.reshape(test_data.shape[0],28,28,1).astype("float32") / 255
predictions = model.predict(test_digit_data)
results = np.argmax(predictions, axis = 1)
```

### Set how is our prediction


```python
plt.figure(figsize=(18, 8))
sample_test = test_data.head(18)
for index, image_pixels in sample_test.iterrows():
    label = results[index]
    show_image(image_pixels, label, index)
plt.tight_layout()
```


![png](/images/digit-recognizer/notebook_27_0.png)


### Create submission file

```python
submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submissions['Label'] = results
submissions.to_csv('submission.csv', index = False)
```

## Saving Model

```python
model.save('model.h5')
```

### Download submission.csv

> [Google Drive](https://drive.google.com/file/d/1MPBdMjIdckiOQYQL0AWx0VRMyYe9JGKU/view?usp=sharing)<br/>
> [OneDrive](https://1drv.ms/u/s!AjWO46TOTFj4p1mz_WkWNbyFSEQ-?e=uUYlJA)<br/>
> [Mediafire](http://www.mediafire.com/file/6p2pqb5dgv6klfp/ieee-cis-fraud-detection-prediction.zip/file)<br/>
> [Mega](https://mega.nz/#!vj4D0KJY!kavcsUxRE6XhdpDzjPUVypCBuLiOiDKT3KYV_ve0rPs)<br/>