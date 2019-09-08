---
title: "Hindi Handwriting Recognition"
date: 2019-09-07
tags: [hindi dataset, hindi recognition, computer vision, keras, machine learning]
header:
  image: #"/images/perceptron/percept.jpg"
excerpt: "Hindi Dataset, Hindi Recognition, Computer Vision, Keras, Machine Learning"
mathjax: "true"
---

## Objective

Classify Hindi alphabets using Convolutional Neural Network.

## In this project

We will use Devnagiri Handwritten Character Dataset which can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset).

Also, this project is implemented in Python 3.7.

And, libraries used are-

1. [Numpy](https://numpy.org/)
2. [Pandas](https://pandas.pydata.org/)
3. [TensorFlow](https://www.tensorflow.org/)
4. [Keras](https://keras.io/)
5. [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)

## Design

We will create two classes here.

1. Model
2. Application

Model class will be responsible for creating a model using character dataset and Application class will recognize Hindi characters in runtime.

## Diagram

## We begin here...

## Creating model (model.py)

```python
import numpy as np
import pandas as pd
```

```python
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
```

```python
data = pd.read_csv("data.csv")
dataset = np.array(data)
np.random.shuffle(dataset)
X = dataset
Y = dataset
X = X[:, 0:1024]
Y = Y[:, 1024]
```

### Train and Test data variables

```python
X_train = X[0:70000, :]
X_train = X_train / 255.
X_test = X[70000:72001, :]
X_test = X_test / 255.
```

```python
# Reshape
Y = Y.reshape(Y.shape[0],1)
Y_train = Y[0:70000, :]
Y_train = Y_train.T
Y_test = Y[70000:72001, :]
Y_test = Y_test.T
```

```python
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
```

### Lets see what we have

    number of training examples = 70000
    number of test examples = 2000
    X_train shape: (70000, 1024)
    Y_train shape: (1, 70000)
    X_test shape: (2000, 1024)
    Y_test shape: (1, 2000)

### Back to code...

```python
image_x = 32
image_y = 32
```

```python
train_y = np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical(Y_test)
train_y = train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1],test_y.shape[2])
X_train = X_train.reshape(X_train.shape[0],image_x, image_y,1)
X_test = X_test.reshape(X_test.shape[0],image_x, image_y,1)
```

```python
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(train_y.shape))
```

### What we got here

    X_train shape: (70000, 32, 32, 1)
    Y_train shape: (70000, 37)

### Back to code...

```python
def keras_model(image_x,image_y):
    num_of_classes = 37
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "devanagari.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list
```

```python
model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=8, batch_size=64,callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)
model.save('devanagari.h5')
```

    CNN Error: 2.20%
    
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        832       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 10, 10, 64)        51264     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 37)                9509      
    =================================================================
    Total params: 61,605
    Trainable params: 61,605
    Non-trainable params: 0
    _________________________________________________________________

### This program will create devnagari.h5 file

### So, now we have our .h5 model file, let's create our Application

## Creating application (application.py)

```python
import numpy as np
from keras.models import load_model
import cv2
from collections import deque
```

```python
model1 = load_model('devanagari.h5')
print(model1)
```

```python
letter_count = {0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: '06_cha',
                7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna', 11: '11_taa', 12: '12_thaa', 13: '13_daa', 
                14: '14_dhaa', 15: '15_adna', 16: '16_ta', 17: '17_tha', 18: '18_da', 19: '19_dha', 20: '20_na', 
                21: '21_pa', 22: '22_pha', 23: '23_ba', 24: '24_bha', 25: '25_ma', 26: '26_yaw', 27: '27_ra', 
                28: '28_la', 29: '29_waw', 30: '30_sha', 31: '31_sha',32: '32_sa', 33: '33_ha',
                34: '34_kshya', 35: '35_tra', 36: '36_gya', 37: 'CHECK'}
```

```python
def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class
```

```python
def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img
```

### Capturing characters 

```python
cap = cv2.VideoCapture(0)
Lower_green = np.array([110, 50, 50])
Upper_green = np.array([130, 255, 255])
pred_class=0
pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
while (cap.isOpened()):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, Lower_green, Upper_green)
    blur = cv2.medianBlur(mask, 15)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    center = None
    if len(cnts) >= 1:
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) > 250:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            M = cv2.moments(contour)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 5)
    elif len(cnts) == 0:
        if len(pts) != []:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y:y + h, x:x + w]
                    # newImage = process_letter(digit)
                    pred_probab, pred_class = keras_predict(model1, digit)
                    print(pred_class, pred_probab)

        pts = deque(maxlen=512)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Conv Network :  " + str(letter_count[pred_class]), (10, 470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(10)
    if k == 27:
        break
```

## That's it...

## So? What's next?

## This will open system's webcam and start capturing the characters.

!video[]( /videos/hindi-handwriting.mp4 ){ size=10 }