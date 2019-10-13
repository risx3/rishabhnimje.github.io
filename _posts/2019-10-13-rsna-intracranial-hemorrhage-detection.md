---
title: "RSNA Intracranial Hemorrhage Detection"
date: 2019-10-13
tags: [Kaggle, Hemorrhage Detection, RSNA, Machine Learning, Keras, Tensorflow, Medical]
excerpt: "Identify acute intracranial hemorrhage and its subtypes"
header:
  overlay_image: "/images/rsna-intracranial-hemorrhage-detection/home-page.jpg"
  caption: #"Photo by Arget on Unsplash"
mathjax: "true"
---

## Overview

Intracranial hemorrhage, bleeding that occurs inside the cranium, is a serious health problem requiring rapid and often intensive medical treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient.

Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

## Hemorrhage Types

Hemorrhage in the head (intracranial hemorrhage) is a relatively common condition that has many causes ranging from trauma, stroke, aneurysm, vascular malformations, high blood pressure, illicit drugs and blood clotting disorders. The neurologic consequences also vary extensively depending upon the size, type of hemorrhage and location ranging from headache to death. The role of the Radiologist is to detect the hemorrhage, characterize the hemorrhage subtype, its size and to determine if the hemorrhage might be jeopardizing critical areas of the brain that might require immediate surgery.<br/>
While all acute (i.e. new) hemorrhages appear dense (i.e. white) on computed tomography (CT), the primary imaging features that help Radiologists determine the subtype of hemorrhage are the location, shape and proximity to other structures (see table).<br/>
Intraparenchymal hemorrhage is blood that is located completely within the brain itself; intraventricular or subarachnoid hemorrhage is blood that has leaked into the spaces of the brain that normally contain cerebrospinal fluid (the ventricles or subarachnoid cisterns). Extra-axial hemorrhages are blood that collects in the tissue coverings that surround the brain (e.g. subdural or epidural subtypes). ee figure.) Patients may exhibit more than one type of cerebral hemorrhage, which c may appear on the same image. While small hemorrhages are less morbid than large hemorrhages typically, even a small hemorrhage can lead to death because it is an indicator of another type of serious abnormality (e.g. cerebral aneurysm).<br/>

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```


```python
train_data = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
print(train_data.head(10))
```

                                  ID  Label
    0          ID_63eb1e259_epidural      0
    1  ID_63eb1e259_intraparenchymal      0
    2  ID_63eb1e259_intraventricular      0
    3      ID_63eb1e259_subarachnoid      0
    4          ID_63eb1e259_subdural      0
    5               ID_63eb1e259_any      0
    6          ID_2669954a7_epidural      0
    7  ID_2669954a7_intraparenchymal      0
    8  ID_2669954a7_intraventricular      0
    9      ID_2669954a7_subarachnoid      0
    


```python
splitData = train_data['ID'].str.split('_', expand = True)
train_data['class'] = splitData[2]
train_data['fileName'] = splitData[0] + '_' + splitData[1]
train_data = train_data.drop(columns=['ID'],axis=1)
del splitData
print(train_data.head(10))
```

       Label             class      fileName
    0      0          epidural  ID_63eb1e259
    1      0  intraparenchymal  ID_63eb1e259
    2      0  intraventricular  ID_63eb1e259
    3      0      subarachnoid  ID_63eb1e259
    4      0          subdural  ID_63eb1e259
    5      0               any  ID_63eb1e259
    6      0          epidural  ID_2669954a7
    7      0  intraparenchymal  ID_2669954a7
    8      0  intraventricular  ID_2669954a7
    9      0      subarachnoid  ID_2669954a7
    

# EDA

### [Data Description](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)

The training data is provided as a set of image Ids and multiple labels, one for each of five sub-types of hemorrhage, plus an additional label for any, which should always be true if any of the sub-type labels is true.

There is also a target column, ```Label```, indicating the probability of whether that type of hemorrhage exists in the indicated image.

There will be **6** rows per image ```Id```. The label indicated by a particular row will look like ```[Image Id]_[Sub-type Name]```, as follows:

```
Id,Label
1_epidural_hemorrhage,0
1_intraparenchymal_hemorrhage,0
1_intraventricular_hemorrhage,0
1_subarachnoid_hemorrhage,0.6
1_subdural_hemorrhage,0
1_any,0.9
```


```python
pivot_train_data = train_data[['Label', 'fileName', 'class']].drop_duplicates().pivot_table(index = 'fileName',columns=['class'], values='Label')
pivot_train_data = pd.DataFrame(pivot_train_data.to_records())
print(pivot_train_data.head(10))
```

           fileName  any  epidural  intraparenchymal  intraventricular  \
    0  ID_000039fa0    0         0                 0                 0   
    1  ID_00005679d    0         0                 0                 0   
    2  ID_00008ce3c    0         0                 0                 0   
    3  ID_0000950d7    0         0                 0                 0   
    4  ID_0000aee4b    0         0                 0                 0   
    5  ID_0000f1657    0         0                 0                 0   
    6  ID_000178e76    0         0                 0                 0   
    7  ID_00019828f    0         0                 0                 0   
    8  ID_0001dcc25    0         0                 0                 0   
    9  ID_0001de0e8    0         0                 0                 0   
    
       subarachnoid  subdural  
    0             0         0  
    1             0         0  
    2             0         0  
    3             0         0  
    4             0         0  
    5             0         0  
    6             0         0  
    7             0         0  
    8             0         0  
    9             0         0  
    


```python
import matplotlib.image as pltimg
import pydicom

fig = plt.figure(figsize = (20,10))
rows = 5
columns = 5
trainImages = os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/')
for i in range(rows*columns):
    ds = pydicom.dcmread('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/' + trainImages[i*100+1])
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot
```


![png](/images/rsna-intracranial-hemorrhage-detection/rsna_5_0.png)



```python
colsToPlot = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
rows = 5
columns = 5
for i_col in colsToPlot:
    fig = plt.figure(figsize = (20,10))
    trainImages = list(pivot_train_data.loc[pivot_train_data[i_col]==1,'fileName'])
    plt.title(i_col + ' Images')
    for i in range(rows*columns):
        ds = pydicom.dcmread('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/' + trainImages[i*100+1] +'.dcm')
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)        
        fig.add_subplot
```


![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_0.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_1.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_2.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_3.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_4.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_6_5.png)



```python
for i_col in colsToPlot:
    plt.figure()
    ax = sns.countplot(pivot_train_data[i_col])
    ax.set_title(i_col + ' class count')
```


![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_0.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_1.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_2.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_3.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_4.png)



![png](/images/rsna-intracranial-hemorrhage-detection/rsna_7_5.png)



```python
# dropping of corrupted image from dataset
pivot_train_data = pivot_train_data.drop(list(pivot_train_data['fileName']).index('ID_6431af929'))
```


```python
import keras
from keras.layers import Dense, Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,LeakyReLU,ZeroPadding2D,Add
from keras.models import Sequential, Model
from keras.initializers import glorot_uniform
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

pivot_train_data = pivot_train_data.sample(frac=1).reset_index(drop=True)
train_df,val_df = train_test_split(pivot_train_data,test_size = 0.03, random_state = 42)
batch_size = 64
```

    Using TensorFlow backend.
    


```python
y_train = train_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
y_val = val_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
train_files = list(train_df['fileName'])

def readDCMFile(fileName):
    ds = pydicom.read_file(fileName) # read dicom image
    img = ds.pixel_array # get image array
    img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA) 
    return img

def generateImageData(train_files,y_train):
    numBatches = int(np.ceil(len(train_files)/batch_size))
    while True:
        for i in range(numBatches):
            batchFiles = train_files[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.array([readDCMFile('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/' + i_f +'.dcm') for i_f in tqdm(batchFiles)])
            y_batch_data = y_train[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))            
            yield x_batch_data,y_batch_data
            
def generateTestImageData(test_files):
    numBatches = int(np.ceil(len(test_files)/batch_size))
    while True:
        for i in range(numBatches):
            batchFiles = test_files[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.array([readDCMFile('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/' + i_f +'.dcm') for i_f in tqdm(batchFiles)])
            x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))
            yield x_batch_data
```


```python
dataGenerator = generateImageData(train_files,train_df[colsToPlot])
val_files = list(val_df['fileName'])
x_val = np.array([readDCMFile('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/' + i_f +'.dcm') for i_f in tqdm(val_files)])
```

    100%|██████████| 20228/20228 [02:28<00:00, 136.62it/s]
    


```python
y_val = val_df[colsToPlot]
```


```python
# loss function definition courtesy https://www.kaggle.com/akensert/resnet50-keras-baseline-model
from keras import backend as K
def logloss(y_true,y_pred):      
    eps = K.epsilon()
    class_weights = np.array([2., 1., 1., 1., 1., 1.])
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    #compute logloss function (vectorised)  
    out = -( y_true *K.log(y_pred)*class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights)
    return K.mean(out, axis=-1)

def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for the this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)

def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """      
    
    eps = K.epsilon()
    class_weights = K.variable([2., 1., 1., 1., 1., 1.])
    y_pred = K.clip(y_pred, eps, 1.0-eps)
    loss = -(y_true*K.log(y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred))
    loss_samples = _normalized_weighted_average(loss,class_weights)
    return K.mean(loss_samples)
```


```python
def convolutionBlock(X,f,filters,stage,block,s):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    X_shortcut = X
    F1,F2,F3 = filters
    X = Conv2D(filters = F1, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '1',
               kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'1')(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def identityBlock(X,f,filters,stage,block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    X_shortcut = X
    F1,F2,F3 = filters
    X = Conv2D(filters = F1, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
```


```python
input_img = Input((64,64,1))
X = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(input_img)
X = BatchNormalization(axis=3, name='initial_bn')(X)
X = Activation('relu', name='initial_relu')(X)
X = ZeroPadding2D((3, 3))(X)

# Stage 1
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# Stage 2
X = convolutionBlock(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
X = identityBlock(X, 3, [64, 64, 256], stage=2, block='b')
X = identityBlock(X, 3, [64, 64, 256], stage=2, block='c')

# Stage 3 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='b')
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='c')
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='d')

# Stage 4 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='b')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='c')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='d')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='e')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='f')

# Stage 5 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='b')
X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='c')


# AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
# output layer
X = Flatten()(X)
out = Dense(6,name='fc' + str(6),activation='sigmoid')(X)
```


```python
x_val = np.reshape(x_val,(x_val.shape[0],x_val.shape[1],x_val.shape[2],1))
```


```python
model_conv = Model(inputs = input_img, outputs = out)
#model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
model_conv.compile(optimizer='Adam',loss = logloss,metrics=[weighted_loss])
model_conv.summary()
history_conv = model_conv.fit_generator(dataGenerator,steps_per_epoch=500, epochs=20,validation_data = (x_val,y_val),verbose = False)
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 64, 64, 1)    0                                            
    __________________________________________________________________________________________________
    initial_conv2d (Conv2D)         (None, 64, 64, 3)    6           input_1[0][0]                    
    __________________________________________________________________________________________________
    initial_bn (BatchNormalization) (None, 64, 64, 3)    12          initial_conv2d[0][0]             
    __________________________________________________________________________________________________
    initial_relu (Activation)       (None, 64, 64, 3)    0           initial_bn[0][0]                 
    __________________________________________________________________________________________________
    zero_padding2d_1 (ZeroPadding2D (None, 70, 70, 3)    0           initial_relu[0][0]               
    __________________________________________________________________________________________________
    conv1 (Conv2D)                  (None, 32, 32, 64)   9472        zero_padding2d_1[0][0]           
    __________________________________________________________________________________________________
    bn_conv1 (BatchNormalization)   (None, 32, 32, 64)   256         conv1[0][0]                      
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 32, 32, 64)   0           bn_conv1[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 15, 15, 64)   0           activation_1[0][0]               
    __________________________________________________________________________________________________
    res2a_branch2a (Conv2D)         (None, 15, 15, 64)   4160        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    bn2a_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 15, 15, 64)   0           bn2a_branch2a[0][0]              
    __________________________________________________________________________________________________
    res2a_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_2[0][0]               
    __________________________________________________________________________________________________
    bn2a_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 15, 15, 64)   0           bn2a_branch2b[0][0]              
    __________________________________________________________________________________________________
    res2a_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_3[0][0]               
    __________________________________________________________________________________________________
    res2a_branch1 (Conv2D)          (None, 15, 15, 256)  16640       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    bn2a_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2a_branch2c[0][0]             
    __________________________________________________________________________________________________
    bn2a_branch1 (BatchNormalizatio (None, 15, 15, 256)  1024        res2a_branch1[0][0]              
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 15, 15, 256)  0           bn2a_branch2c[0][0]              
                                                                     bn2a_branch1[0][0]               
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 15, 15, 256)  0           add_1[0][0]                      
    __________________________________________________________________________________________________
    res2b_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_4[0][0]               
    __________________________________________________________________________________________________
    bn2b_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 15, 15, 64)   0           bn2b_branch2a[0][0]              
    __________________________________________________________________________________________________
    res2b_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_5[0][0]               
    __________________________________________________________________________________________________
    bn2b_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 15, 15, 64)   0           bn2b_branch2b[0][0]              
    __________________________________________________________________________________________________
    res2b_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_6[0][0]               
    __________________________________________________________________________________________________
    bn2b_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2b_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 15, 15, 256)  0           bn2b_branch2c[0][0]              
                                                                     activation_4[0][0]               
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 15, 15, 256)  0           add_2[0][0]                      
    __________________________________________________________________________________________________
    res2c_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_7[0][0]               
    __________________________________________________________________________________________________
    bn2c_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 15, 15, 64)   0           bn2c_branch2a[0][0]              
    __________________________________________________________________________________________________
    res2c_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_8[0][0]               
    __________________________________________________________________________________________________
    bn2c_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 15, 15, 64)   0           bn2c_branch2b[0][0]              
    __________________________________________________________________________________________________
    res2c_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_9[0][0]               
    __________________________________________________________________________________________________
    bn2c_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2c_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 15, 15, 256)  0           bn2c_branch2c[0][0]              
                                                                     activation_7[0][0]               
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 15, 15, 256)  0           add_3[0][0]                      
    __________________________________________________________________________________________________
    res3a_branch2a (Conv2D)         (None, 8, 8, 128)    32896       activation_10[0][0]              
    __________________________________________________________________________________________________
    bn3a_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2a[0][0]              
    __________________________________________________________________________________________________
    res3a_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_11[0][0]              
    __________________________________________________________________________________________________
    bn3a_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2b[0][0]              
    __________________________________________________________________________________________________
    res3a_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_12[0][0]              
    __________________________________________________________________________________________________
    res3a_branch1 (Conv2D)          (None, 8, 8, 512)    131584      activation_10[0][0]              
    __________________________________________________________________________________________________
    bn3a_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3a_branch2c[0][0]             
    __________________________________________________________________________________________________
    bn3a_branch1 (BatchNormalizatio (None, 8, 8, 512)    2048        res3a_branch1[0][0]              
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 8, 8, 512)    0           bn3a_branch2c[0][0]              
                                                                     bn3a_branch1[0][0]               
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 8, 8, 512)    0           add_4[0][0]                      
    __________________________________________________________________________________________________
    res3b_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_13[0][0]              
    __________________________________________________________________________________________________
    bn3b_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2a[0][0]              
    __________________________________________________________________________________________________
    res3b_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_14[0][0]              
    __________________________________________________________________________________________________
    bn3b_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2b[0][0]              
    __________________________________________________________________________________________________
    res3b_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_15[0][0]              
    __________________________________________________________________________________________________
    bn3b_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3b_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 8, 8, 512)    0           bn3b_branch2c[0][0]              
                                                                     activation_13[0][0]              
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 8, 8, 512)    0           add_5[0][0]                      
    __________________________________________________________________________________________________
    res3c_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_16[0][0]              
    __________________________________________________________________________________________________
    bn3c_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2a[0][0]              
    __________________________________________________________________________________________________
    res3c_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_17[0][0]              
    __________________________________________________________________________________________________
    bn3c_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2b[0][0]              
    __________________________________________________________________________________________________
    res3c_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_18[0][0]              
    __________________________________________________________________________________________________
    bn3c_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3c_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 8, 8, 512)    0           bn3c_branch2c[0][0]              
                                                                     activation_16[0][0]              
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 8, 8, 512)    0           add_6[0][0]                      
    __________________________________________________________________________________________________
    res3d_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_19[0][0]              
    __________________________________________________________________________________________________
    bn3d_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2a[0][0]              
    __________________________________________________________________________________________________
    res3d_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_20[0][0]              
    __________________________________________________________________________________________________
    bn3d_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2b[0][0]              
    __________________________________________________________________________________________________
    res3d_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_21[0][0]              
    __________________________________________________________________________________________________
    bn3d_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3d_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, 8, 8, 512)    0           bn3d_branch2c[0][0]              
                                                                     activation_19[0][0]              
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 8, 8, 512)    0           add_7[0][0]                      
    __________________________________________________________________________________________________
    res4a_branch2a (Conv2D)         (None, 4, 4, 256)    131328      activation_22[0][0]              
    __________________________________________________________________________________________________
    bn4a_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4a_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_23[0][0]              
    __________________________________________________________________________________________________
    bn4a_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4a_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_24[0][0]              
    __________________________________________________________________________________________________
    res4a_branch1 (Conv2D)          (None, 4, 4, 1024)   525312      activation_22[0][0]              
    __________________________________________________________________________________________________
    bn4a_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4a_branch2c[0][0]             
    __________________________________________________________________________________________________
    bn4a_branch1 (BatchNormalizatio (None, 4, 4, 1024)   4096        res4a_branch1[0][0]              
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, 4, 4, 1024)   0           bn4a_branch2c[0][0]              
                                                                     bn4a_branch1[0][0]               
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 4, 4, 1024)   0           add_8[0][0]                      
    __________________________________________________________________________________________________
    res4b_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_25[0][0]              
    __________________________________________________________________________________________________
    bn4b_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4b_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_26[0][0]              
    __________________________________________________________________________________________________
    bn4b_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4b_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_27[0][0]              
    __________________________________________________________________________________________________
    bn4b_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4b_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, 4, 4, 1024)   0           bn4b_branch2c[0][0]              
                                                                     activation_25[0][0]              
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 4, 4, 1024)   0           add_9[0][0]                      
    __________________________________________________________________________________________________
    res4c_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_28[0][0]              
    __________________________________________________________________________________________________
    bn4c_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4c_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_29[0][0]              
    __________________________________________________________________________________________________
    bn4c_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4c_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_30[0][0]              
    __________________________________________________________________________________________________
    bn4c_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4c_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, 4, 4, 1024)   0           bn4c_branch2c[0][0]              
                                                                     activation_28[0][0]              
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 4, 4, 1024)   0           add_10[0][0]                     
    __________________________________________________________________________________________________
    res4d_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_31[0][0]              
    __________________________________________________________________________________________________
    bn4d_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4d_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_32[0][0]              
    __________________________________________________________________________________________________
    bn4d_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4d_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_33[0][0]              
    __________________________________________________________________________________________________
    bn4d_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4d_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, 4, 4, 1024)   0           bn4d_branch2c[0][0]              
                                                                     activation_31[0][0]              
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 4, 4, 1024)   0           add_11[0][0]                     
    __________________________________________________________________________________________________
    res4e_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_34[0][0]              
    __________________________________________________________________________________________________
    bn4e_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4e_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_35[0][0]              
    __________________________________________________________________________________________________
    bn4e_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4e_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_36[0][0]              
    __________________________________________________________________________________________________
    bn4e_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4e_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, 4, 4, 1024)   0           bn4e_branch2c[0][0]              
                                                                     activation_34[0][0]              
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 4, 4, 1024)   0           add_12[0][0]                     
    __________________________________________________________________________________________________
    res4f_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_37[0][0]              
    __________________________________________________________________________________________________
    bn4f_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2a[0][0]              
    __________________________________________________________________________________________________
    res4f_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_38[0][0]              
    __________________________________________________________________________________________________
    bn4f_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2b[0][0]              
    __________________________________________________________________________________________________
    res4f_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_39[0][0]              
    __________________________________________________________________________________________________
    bn4f_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4f_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_13 (Add)                    (None, 4, 4, 1024)   0           bn4f_branch2c[0][0]              
                                                                     activation_37[0][0]              
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 4, 4, 1024)   0           add_13[0][0]                     
    __________________________________________________________________________________________________
    res5a_branch2a (Conv2D)         (None, 2, 2, 512)    524800      activation_40[0][0]              
    __________________________________________________________________________________________________
    bn5a_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2a[0][0]              
    __________________________________________________________________________________________________
    res5a_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_41[0][0]              
    __________________________________________________________________________________________________
    bn5a_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2b[0][0]              
    __________________________________________________________________________________________________
    res5a_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_42[0][0]              
    __________________________________________________________________________________________________
    res5a_branch1 (Conv2D)          (None, 2, 2, 2048)   2099200     activation_40[0][0]              
    __________________________________________________________________________________________________
    bn5a_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5a_branch2c[0][0]             
    __________________________________________________________________________________________________
    bn5a_branch1 (BatchNormalizatio (None, 2, 2, 2048)   8192        res5a_branch1[0][0]              
    __________________________________________________________________________________________________
    add_14 (Add)                    (None, 2, 2, 2048)   0           bn5a_branch2c[0][0]              
                                                                     bn5a_branch1[0][0]               
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 2, 2, 2048)   0           add_14[0][0]                     
    __________________________________________________________________________________________________
    res5b_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_43[0][0]              
    __________________________________________________________________________________________________
    bn5b_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2a[0][0]              
    __________________________________________________________________________________________________
    res5b_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_44[0][0]              
    __________________________________________________________________________________________________
    bn5b_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2b[0][0]              
    __________________________________________________________________________________________________
    res5b_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_45[0][0]              
    __________________________________________________________________________________________________
    bn5b_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5b_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_15 (Add)                    (None, 2, 2, 2048)   0           bn5b_branch2c[0][0]              
                                                                     activation_43[0][0]              
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 2, 2, 2048)   0           add_15[0][0]                     
    __________________________________________________________________________________________________
    res5c_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_46[0][0]              
    __________________________________________________________________________________________________
    bn5c_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2a[0][0]             
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2a[0][0]              
    __________________________________________________________________________________________________
    res5c_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_47[0][0]              
    __________________________________________________________________________________________________
    bn5c_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2b[0][0]             
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2b[0][0]              
    __________________________________________________________________________________________________
    res5c_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_48[0][0]              
    __________________________________________________________________________________________________
    bn5c_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5c_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_16 (Add)                    (None, 2, 2, 2048)   0           bn5c_branch2c[0][0]              
                                                                     activation_46[0][0]              
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 2, 2, 2048)   0           add_16[0][0]                     
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 1, 1, 2048)   0           activation_49[0][0]              
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 2048)         0           average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    fc6 (Dense)                     (None, 6)            12294       flatten_1[0][0]                  
    ==================================================================================================
    Total params: 23,600,024
    Trainable params: 23,546,898
    Non-trainable params: 53,126
    __________________________________________________________________________________________________
    

    100%|██████████| 64/64 [00:00<00:00, 116.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 110.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 114.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 141.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 110.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 141.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 143.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 110.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 98.09it/s] 
    100%|██████████| 64/64 [00:00<00:00, 82.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 113.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 101.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 141.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 143.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 102.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 98.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 99.82it/s] 
    100%|██████████| 64/64 [00:00<00:00, 97.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 101.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 102.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 100.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 113.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 99.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 98.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 102.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 114.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 113.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 113.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 113.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 114.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 116.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 98.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 101.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 112.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 101.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 107.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 99.73it/s] 
    100%|██████████| 64/64 [00:00<00:00, 103.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 103.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 100.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 140.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 141.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 111.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 118.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 141.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.00it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.05it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 139.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.87it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 142.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.81it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.49it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.51it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.53it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.15it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.29it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.03it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 138.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.94it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.39it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.62it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.25it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.90it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.69it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.54it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.01it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.74it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.38it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.72it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.61it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.22it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.80it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.89it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.43it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.63it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 102.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 104.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 109.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 100.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 99.64it/s] 
    100%|██████████| 64/64 [00:00<00:00, 102.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.27it/s]
    100%|██████████| 64/64 [00:00<00:00, 101.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 100.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.52it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.93it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.55it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.95it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.04it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 115.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.09it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.96it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.79it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.17it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.40it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.37it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.76it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.46it/s]
    100%|██████████| 64/64 [00:00<00:00, 135.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.02it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.23it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.20it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.92it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.59it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.13it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.91it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.32it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.41it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.14it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.57it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.45it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.16it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 120.08it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.24it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.83it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.68it/s]
    100%|██████████| 64/64 [00:00<00:00, 121.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.75it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.31it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.42it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.98it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.71it/s]
    100%|██████████| 64/64 [00:00<00:00, 124.97it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.86it/s]
    100%|██████████| 64/64 [00:00<00:00, 136.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.56it/s]
    100%|██████████| 64/64 [00:00<00:00, 83.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 117.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.07it/s]
    100%|██████████| 64/64 [00:00<00:00, 123.11it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.58it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.44it/s]
    100%|██████████| 64/64 [00:00<00:00, 122.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.12it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.30it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 119.18it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.67it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.35it/s]
    100%|██████████| 64/64 [00:00<00:00, 133.06it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.50it/s]
    100%|██████████| 64/64 [00:00<00:00, 125.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.78it/s]
    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.85it/s]
    100%|██████████| 64/64 [00:00<00:00, 134.19it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.36it/s]
    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
    100%|██████████| 64/64 [00:00<00:00, 137.70it/s]
    100%|██████████| 64/64 [00:00<00:00, 128.65it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
    100%|██████████| 64/64 [00:00<00:00, 127.28it/s]
    100%|██████████| 64/64 [00:00<00:00, 129.82it/s]
    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]


```python
test_data = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
splitData = test_data['ID'].str.split('_', expand = True)
test_data['class'] = splitData[2]
test_data['fileName'] = splitData[0] + '_' + splitData[1]
test_data = test_data.drop(columns=['ID'],axis=1)
del splitData
pivot_test_data = test_data[['fileName','class','Label']].drop_duplicates().pivot_table(index = 'fileName',columns=['class'], values='Label')
pivot_test_data = pd.DataFrame(pivot_test_data.to_records())
test_files = list(pivot_test_data['fileName'])
testDataGenerator = generateTestImageData(test_files)
temp_pred = model_conv.predict_generator(testDataGenerator,steps = pivot_test_data.shape[0]/batch_size,verbose = True)
```

    100%|██████████| 64/64 [00:00<00:00, 121.64it/s]
    100%|██████████| 64/64 [00:00<00:00, 106.34it/s]
    100%|██████████| 64/64 [00:00<00:00, 108.99it/s]
    100%|██████████| 64/64 [00:00<00:00, 105.13it/s]
     55%|█████▍    | 35/64 [00:00<00:00, 110.47it/s]

       1/1227 [..............................] - ETA: 56:19

     70%|███████   | 45/64 [00:00<00:00, 105.82it/s]

       3/1227 [..............................] - ETA: 19:08

    100%|██████████| 64/64 [00:00<00:00, 106.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

       5/1227 [..............................] - ETA: 12:09

    100%|██████████| 64/64 [00:00<00:00, 125.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

       6/1227 [..............................] - ETA: 11:53

    100%|██████████| 64/64 [00:00<00:00, 127.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

       7/1227 [..............................] - ETA: 11:40

    100%|██████████| 64/64 [00:00<00:00, 128.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

       8/1227 [..............................] - ETA: 11:29

    100%|██████████| 64/64 [00:00<00:00, 128.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

       9/1227 [..............................] - ETA: 11:20

    100%|██████████| 64/64 [00:00<00:00, 128.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      10/1227 [..............................] - ETA: 11:13

    100%|██████████| 64/64 [00:00<00:00, 136.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      11/1227 [..............................] - ETA: 11:04

    100%|██████████| 64/64 [00:00<00:00, 126.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      12/1227 [..............................] - ETA: 11:00

    100%|██████████| 64/64 [00:00<00:00, 133.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      13/1227 [..............................] - ETA: 10:54

    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      14/1227 [..............................] - ETA: 10:50

    100%|██████████| 64/64 [00:00<00:00, 144.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      15/1227 [..............................] - ETA: 10:43

    100%|██████████| 64/64 [00:00<00:00, 131.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      16/1227 [..............................] - ETA: 10:40

    100%|██████████| 64/64 [00:00<00:00, 141.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      17/1227 [..............................] - ETA: 10:35

    100%|██████████| 64/64 [00:00<00:00, 138.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      18/1227 [..............................] - ETA: 10:31

    100%|██████████| 64/64 [00:00<00:00, 142.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      19/1227 [..............................] - ETA: 10:26

    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      20/1227 [..............................] - ETA: 10:24

    100%|██████████| 64/64 [00:00<00:00, 138.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      21/1227 [..............................] - ETA: 10:20

    100%|██████████| 64/64 [00:00<00:00, 133.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      22/1227 [..............................] - ETA: 10:18

    100%|██████████| 64/64 [00:00<00:00, 133.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      23/1227 [..............................] - ETA: 10:16

    100%|██████████| 64/64 [00:00<00:00, 129.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      24/1227 [..............................] - ETA: 10:16

    100%|██████████| 64/64 [00:00<00:00, 133.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      25/1227 [..............................] - ETA: 10:14

    100%|██████████| 64/64 [00:00<00:00, 130.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      26/1227 [..............................] - ETA: 10:14

    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      27/1227 [..............................] - ETA: 10:13

    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      28/1227 [..............................] - ETA: 10:12

    100%|██████████| 64/64 [00:00<00:00, 135.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      29/1227 [..............................] - ETA: 10:10

    100%|██████████| 64/64 [00:00<00:00, 135.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      30/1227 [..............................] - ETA: 10:08

    100%|██████████| 64/64 [00:00<00:00, 133.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      31/1227 [..............................] - ETA: 10:07

    100%|██████████| 64/64 [00:00<00:00, 136.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      32/1227 [..............................] - ETA: 10:05

    100%|██████████| 64/64 [00:00<00:00, 130.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      33/1227 [..............................] - ETA: 10:04

    100%|██████████| 64/64 [00:00<00:00, 132.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      34/1227 [..............................] - ETA: 10:03

    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      35/1227 [..............................] - ETA: 10:03

    100%|██████████| 64/64 [00:00<00:00, 150.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      36/1227 [..............................] - ETA: 10:00

    100%|██████████| 64/64 [00:00<00:00, 141.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      37/1227 [..............................] - ETA: 9:58 

    100%|██████████| 64/64 [00:00<00:00, 144.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      38/1227 [..............................] - ETA: 9:56

    100%|██████████| 64/64 [00:00<00:00, 130.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      39/1227 [..............................] - ETA: 9:55

    100%|██████████| 64/64 [00:00<00:00, 136.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      40/1227 [..............................] - ETA: 9:54

    100%|██████████| 64/64 [00:00<00:00, 132.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      41/1227 [>.............................] - ETA: 9:53

    100%|██████████| 64/64 [00:00<00:00, 132.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      42/1227 [>.............................] - ETA: 9:52

    100%|██████████| 64/64 [00:00<00:00, 135.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      43/1227 [>.............................] - ETA: 9:51

    100%|██████████| 64/64 [00:00<00:00, 136.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      44/1227 [>.............................] - ETA: 9:50

    100%|██████████| 64/64 [00:00<00:00, 132.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      45/1227 [>.............................] - ETA: 9:49

    100%|██████████| 64/64 [00:00<00:00, 134.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      46/1227 [>.............................] - ETA: 9:48

    100%|██████████| 64/64 [00:00<00:00, 139.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      47/1227 [>.............................] - ETA: 9:47

    100%|██████████| 64/64 [00:00<00:00, 136.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      48/1227 [>.............................] - ETA: 9:46

    100%|██████████| 64/64 [00:00<00:00, 132.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      49/1227 [>.............................] - ETA: 9:46

    100%|██████████| 64/64 [00:00<00:00, 138.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      50/1227 [>.............................] - ETA: 9:45

    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      51/1227 [>.............................] - ETA: 9:44

    100%|██████████| 64/64 [00:00<00:00, 130.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      52/1227 [>.............................] - ETA: 9:44

    100%|██████████| 64/64 [00:00<00:00, 136.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      53/1227 [>.............................] - ETA: 9:43

    100%|██████████| 64/64 [00:00<00:00, 133.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      54/1227 [>.............................] - ETA: 9:42

    100%|██████████| 64/64 [00:00<00:00, 139.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      55/1227 [>.............................] - ETA: 9:41

    100%|██████████| 64/64 [00:00<00:00, 138.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      56/1227 [>.............................] - ETA: 9:40

    100%|██████████| 64/64 [00:00<00:00, 130.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      57/1227 [>.............................] - ETA: 9:39

    100%|██████████| 64/64 [00:00<00:00, 135.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      58/1227 [>.............................] - ETA: 9:38

    100%|██████████| 64/64 [00:00<00:00, 132.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      59/1227 [>.............................] - ETA: 9:38

    100%|██████████| 64/64 [00:00<00:00, 138.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      60/1227 [>.............................] - ETA: 9:37

    100%|██████████| 64/64 [00:00<00:00, 137.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      61/1227 [>.............................] - ETA: 9:36

    100%|██████████| 64/64 [00:00<00:00, 131.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      62/1227 [>.............................] - ETA: 9:35

    100%|██████████| 64/64 [00:00<00:00, 135.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      63/1227 [>.............................] - ETA: 9:35

    100%|██████████| 64/64 [00:00<00:00, 136.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      64/1227 [>.............................] - ETA: 9:34

    100%|██████████| 64/64 [00:00<00:00, 133.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      65/1227 [>.............................] - ETA: 9:33

    100%|██████████| 64/64 [00:00<00:00, 134.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      66/1227 [>.............................] - ETA: 9:33

    100%|██████████| 64/64 [00:00<00:00, 135.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      67/1227 [>.............................] - ETA: 9:32

    100%|██████████| 64/64 [00:00<00:00, 140.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      68/1227 [>.............................] - ETA: 9:31

    100%|██████████| 64/64 [00:00<00:00, 136.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      69/1227 [>.............................] - ETA: 9:30

    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      70/1227 [>.............................] - ETA: 9:30

    100%|██████████| 64/64 [00:00<00:00, 135.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      71/1227 [>.............................] - ETA: 9:29

    100%|██████████| 64/64 [00:00<00:00, 138.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      72/1227 [>.............................] - ETA: 9:28

    100%|██████████| 64/64 [00:00<00:00, 139.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      73/1227 [>.............................] - ETA: 9:27

    100%|██████████| 64/64 [00:00<00:00, 132.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      74/1227 [>.............................] - ETA: 9:27

    100%|██████████| 64/64 [00:00<00:00, 137.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      75/1227 [>.............................] - ETA: 9:26

    100%|██████████| 64/64 [00:00<00:00, 130.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      76/1227 [>.............................] - ETA: 9:25

    100%|██████████| 64/64 [00:00<00:00, 135.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      77/1227 [>.............................] - ETA: 9:25

    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      78/1227 [>.............................] - ETA: 9:24

    100%|██████████| 64/64 [00:00<00:00, 133.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      79/1227 [>.............................] - ETA: 9:24

    100%|██████████| 64/64 [00:00<00:00, 131.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      80/1227 [>.............................] - ETA: 9:23

    100%|██████████| 64/64 [00:00<00:00, 135.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      81/1227 [>.............................] - ETA: 9:23

    100%|██████████| 64/64 [00:00<00:00, 128.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      82/1227 [=>............................] - ETA: 9:22

    100%|██████████| 64/64 [00:00<00:00, 134.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      83/1227 [=>............................] - ETA: 9:22

    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      84/1227 [=>............................] - ETA: 9:21

    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      85/1227 [=>............................] - ETA: 9:21

    100%|██████████| 64/64 [00:00<00:00, 136.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      86/1227 [=>............................] - ETA: 9:20

    100%|██████████| 64/64 [00:00<00:00, 129.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      87/1227 [=>............................] - ETA: 9:20

    100%|██████████| 64/64 [00:00<00:00, 134.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      88/1227 [=>............................] - ETA: 9:19

    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      89/1227 [=>............................] - ETA: 9:19

    100%|██████████| 64/64 [00:00<00:00, 138.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      90/1227 [=>............................] - ETA: 9:18

    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      91/1227 [=>............................] - ETA: 9:18

    100%|██████████| 64/64 [00:00<00:00, 137.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      92/1227 [=>............................] - ETA: 9:17

    100%|██████████| 64/64 [00:00<00:00, 134.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      93/1227 [=>............................] - ETA: 9:17

    100%|██████████| 64/64 [00:00<00:00, 136.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      94/1227 [=>............................] - ETA: 9:16

    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      95/1227 [=>............................] - ETA: 9:16

    100%|██████████| 64/64 [00:00<00:00, 135.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      96/1227 [=>............................] - ETA: 9:15

    100%|██████████| 64/64 [00:00<00:00, 131.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      97/1227 [=>............................] - ETA: 9:14

    100%|██████████| 64/64 [00:00<00:00, 136.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      98/1227 [=>............................] - ETA: 9:14

    100%|██████████| 64/64 [00:00<00:00, 132.78it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

      99/1227 [=>............................] - ETA: 9:13

    100%|██████████| 64/64 [00:00<00:00, 138.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     100/1227 [=>............................] - ETA: 9:13

    100%|██████████| 64/64 [00:00<00:00, 127.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     101/1227 [=>............................] - ETA: 9:12

    100%|██████████| 64/64 [00:00<00:00, 131.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     102/1227 [=>............................] - ETA: 9:12

    100%|██████████| 64/64 [00:00<00:00, 133.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     103/1227 [=>............................] - ETA: 9:11

    100%|██████████| 64/64 [00:00<00:00, 133.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     104/1227 [=>............................] - ETA: 9:11

    100%|██████████| 64/64 [00:00<00:00, 134.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     105/1227 [=>............................] - ETA: 9:10

    100%|██████████| 64/64 [00:00<00:00, 133.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     106/1227 [=>............................] - ETA: 9:10

    100%|██████████| 64/64 [00:00<00:00, 132.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     107/1227 [=>............................] - ETA: 9:09

    100%|██████████| 64/64 [00:00<00:00, 133.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     108/1227 [=>............................] - ETA: 9:09

    100%|██████████| 64/64 [00:00<00:00, 132.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     109/1227 [=>............................] - ETA: 9:08

    100%|██████████| 64/64 [00:00<00:00, 125.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     110/1227 [=>............................] - ETA: 9:08

    100%|██████████| 64/64 [00:00<00:00, 130.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     111/1227 [=>............................] - ETA: 9:08

    100%|██████████| 64/64 [00:00<00:00, 130.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     112/1227 [=>............................] - ETA: 9:07

    100%|██████████| 64/64 [00:00<00:00, 132.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     113/1227 [=>............................] - ETA: 9:07

    100%|██████████| 64/64 [00:00<00:00, 132.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     114/1227 [=>............................] - ETA: 9:06

    100%|██████████| 64/64 [00:00<00:00, 128.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     115/1227 [=>............................] - ETA: 9:06

    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     116/1227 [=>............................] - ETA: 9:06

    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     117/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 125.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     118/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 123.57it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     119/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 120.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     120/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 123.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     121/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 119.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     122/1227 [=>............................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 117.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     123/1227 [==>...........................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 118.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     124/1227 [==>...........................] - ETA: 9:05

    100%|██████████| 64/64 [00:00<00:00, 122.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     125/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 120.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     126/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 123.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     127/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 120.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     128/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 122.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     129/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 119.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     130/1227 [==>...........................] - ETA: 9:04

    100%|██████████| 64/64 [00:00<00:00, 123.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     131/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 117.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     132/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 118.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     133/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 119.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     134/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 124.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     135/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 121.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     136/1227 [==>...........................] - ETA: 9:03

    100%|██████████| 64/64 [00:00<00:00, 126.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     137/1227 [==>...........................] - ETA: 9:02

    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     138/1227 [==>...........................] - ETA: 9:02

    100%|██████████| 64/64 [00:00<00:00, 127.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     139/1227 [==>...........................] - ETA: 9:02

    100%|██████████| 64/64 [00:00<00:00, 122.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     140/1227 [==>...........................] - ETA: 9:01

    100%|██████████| 64/64 [00:00<00:00, 122.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     141/1227 [==>...........................] - ETA: 9:01

    100%|██████████| 64/64 [00:00<00:00, 118.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     142/1227 [==>...........................] - ETA: 9:01

    100%|██████████| 64/64 [00:00<00:00, 126.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     143/1227 [==>...........................] - ETA: 9:01

    100%|██████████| 64/64 [00:00<00:00, 131.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     144/1227 [==>...........................] - ETA: 9:00

    100%|██████████| 64/64 [00:00<00:00, 133.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     145/1227 [==>...........................] - ETA: 8:59

    100%|██████████| 64/64 [00:00<00:00, 125.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     146/1227 [==>...........................] - ETA: 8:59

    100%|██████████| 64/64 [00:00<00:00, 131.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     147/1227 [==>...........................] - ETA: 8:59

    100%|██████████| 64/64 [00:00<00:00, 127.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     148/1227 [==>...........................] - ETA: 8:58

    100%|██████████| 64/64 [00:00<00:00, 126.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     149/1227 [==>...........................] - ETA: 8:58

    100%|██████████| 64/64 [00:00<00:00, 130.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     150/1227 [==>...........................] - ETA: 8:57

    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     151/1227 [==>...........................] - ETA: 8:57

    100%|██████████| 64/64 [00:00<00:00, 129.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     152/1227 [==>...........................] - ETA: 8:56

    100%|██████████| 64/64 [00:00<00:00, 129.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     153/1227 [==>...........................] - ETA: 8:56

    100%|██████████| 64/64 [00:00<00:00, 124.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     154/1227 [==>...........................] - ETA: 8:55

    100%|██████████| 64/64 [00:00<00:00, 128.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     155/1227 [==>...........................] - ETA: 8:55

    100%|██████████| 64/64 [00:00<00:00, 126.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     156/1227 [==>...........................] - ETA: 8:55

    100%|██████████| 64/64 [00:00<00:00, 128.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     157/1227 [==>...........................] - ETA: 8:54

    100%|██████████| 64/64 [00:00<00:00, 128.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     158/1227 [==>...........................] - ETA: 8:54

    100%|██████████| 64/64 [00:00<00:00, 128.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     159/1227 [==>...........................] - ETA: 8:53

    100%|██████████| 64/64 [00:00<00:00, 123.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     160/1227 [==>...........................] - ETA: 8:53

    100%|██████████| 64/64 [00:00<00:00, 126.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     161/1227 [==>...........................] - ETA: 8:53

    100%|██████████| 64/64 [00:00<00:00, 124.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     162/1227 [==>...........................] - ETA: 8:52

    100%|██████████| 64/64 [00:00<00:00, 132.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     163/1227 [==>...........................] - ETA: 8:52

    100%|██████████| 64/64 [00:00<00:00, 123.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     164/1227 [===>..........................] - ETA: 8:51

    100%|██████████| 64/64 [00:00<00:00, 128.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     165/1227 [===>..........................] - ETA: 8:51

    100%|██████████| 64/64 [00:00<00:00, 131.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     166/1227 [===>..........................] - ETA: 8:50

    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     167/1227 [===>..........................] - ETA: 8:50

    100%|██████████| 64/64 [00:00<00:00, 129.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     168/1227 [===>..........................] - ETA: 8:49

    100%|██████████| 64/64 [00:00<00:00, 133.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     169/1227 [===>..........................] - ETA: 8:49

    100%|██████████| 64/64 [00:00<00:00, 125.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     170/1227 [===>..........................] - ETA: 8:48

    100%|██████████| 64/64 [00:00<00:00, 124.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     171/1227 [===>..........................] - ETA: 8:48

    100%|██████████| 64/64 [00:00<00:00, 131.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     172/1227 [===>..........................] - ETA: 8:47

    100%|██████████| 64/64 [00:00<00:00, 129.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     173/1227 [===>..........................] - ETA: 8:47

    100%|██████████| 64/64 [00:00<00:00, 124.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     174/1227 [===>..........................] - ETA: 8:47

    100%|██████████| 64/64 [00:00<00:00, 125.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     175/1227 [===>..........................] - ETA: 8:46

    100%|██████████| 64/64 [00:00<00:00, 122.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     176/1227 [===>..........................] - ETA: 8:46

    100%|██████████| 64/64 [00:00<00:00, 121.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     177/1227 [===>..........................] - ETA: 8:46

    100%|██████████| 64/64 [00:00<00:00, 120.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     178/1227 [===>..........................] - ETA: 8:45

    100%|██████████| 64/64 [00:00<00:00, 121.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     179/1227 [===>..........................] - ETA: 8:45

    100%|██████████| 64/64 [00:00<00:00, 117.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     180/1227 [===>..........................] - ETA: 8:45

    100%|██████████| 64/64 [00:00<00:00, 122.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     181/1227 [===>..........................] - ETA: 8:44

    100%|██████████| 64/64 [00:00<00:00, 117.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     182/1227 [===>..........................] - ETA: 8:44

    100%|██████████| 64/64 [00:00<00:00, 124.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     183/1227 [===>..........................] - ETA: 8:44

    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     184/1227 [===>..........................] - ETA: 8:43

    100%|██████████| 64/64 [00:00<00:00, 131.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     185/1227 [===>..........................] - ETA: 8:43

    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     186/1227 [===>..........................] - ETA: 8:42

    100%|██████████| 64/64 [00:00<00:00, 136.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     187/1227 [===>..........................] - ETA: 8:42

    100%|██████████| 64/64 [00:00<00:00, 126.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     188/1227 [===>..........................] - ETA: 8:41

    100%|██████████| 64/64 [00:00<00:00, 130.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     189/1227 [===>..........................] - ETA: 8:41

    100%|██████████| 64/64 [00:00<00:00, 125.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     190/1227 [===>..........................] - ETA: 8:40

    100%|██████████| 64/64 [00:00<00:00, 128.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     191/1227 [===>..........................] - ETA: 8:40

    100%|██████████| 64/64 [00:00<00:00, 128.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     192/1227 [===>..........................] - ETA: 8:39

    100%|██████████| 64/64 [00:00<00:00, 131.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     193/1227 [===>..........................] - ETA: 8:39

    100%|██████████| 64/64 [00:00<00:00, 129.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     194/1227 [===>..........................] - ETA: 8:38

    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     195/1227 [===>..........................] - ETA: 8:38

    100%|██████████| 64/64 [00:00<00:00, 128.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     196/1227 [===>..........................] - ETA: 8:37

    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     197/1227 [===>..........................] - ETA: 8:37

    100%|██████████| 64/64 [00:00<00:00, 125.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     198/1227 [===>..........................] - ETA: 8:36

    100%|██████████| 64/64 [00:00<00:00, 130.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     199/1227 [===>..........................] - ETA: 8:36

    100%|██████████| 64/64 [00:00<00:00, 126.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     200/1227 [===>..........................] - ETA: 8:35

    100%|██████████| 64/64 [00:00<00:00, 130.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     201/1227 [===>..........................] - ETA: 8:35

    100%|██████████| 64/64 [00:00<00:00, 125.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     202/1227 [===>..........................] - ETA: 8:34

    100%|██████████| 64/64 [00:00<00:00, 125.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     203/1227 [===>..........................] - ETA: 8:34

    100%|██████████| 64/64 [00:00<00:00, 123.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     204/1227 [===>..........................] - ETA: 8:34

    100%|██████████| 64/64 [00:00<00:00, 124.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     205/1227 [====>.........................] - ETA: 8:33

    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     206/1227 [====>.........................] - ETA: 8:33

    100%|██████████| 64/64 [00:00<00:00, 135.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     207/1227 [====>.........................] - ETA: 8:32

    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     208/1227 [====>.........................] - ETA: 8:32

    100%|██████████| 64/64 [00:00<00:00, 132.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     209/1227 [====>.........................] - ETA: 8:31

    100%|██████████| 64/64 [00:00<00:00, 130.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     210/1227 [====>.........................] - ETA: 8:31

    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     211/1227 [====>.........................] - ETA: 8:30

    100%|██████████| 64/64 [00:00<00:00, 130.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     212/1227 [====>.........................] - ETA: 8:29

    100%|██████████| 64/64 [00:00<00:00, 126.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     213/1227 [====>.........................] - ETA: 8:29

    100%|██████████| 64/64 [00:00<00:00, 129.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     214/1227 [====>.........................] - ETA: 8:29

    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     215/1227 [====>.........................] - ETA: 8:28

    100%|██████████| 64/64 [00:00<00:00, 124.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     216/1227 [====>.........................] - ETA: 8:28

    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     217/1227 [====>.........................] - ETA: 8:27

    100%|██████████| 64/64 [00:00<00:00, 127.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     218/1227 [====>.........................] - ETA: 8:27

    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     219/1227 [====>.........................] - ETA: 8:26

    100%|██████████| 64/64 [00:00<00:00, 126.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     220/1227 [====>.........................] - ETA: 8:26

    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     221/1227 [====>.........................] - ETA: 8:25

    100%|██████████| 64/64 [00:00<00:00, 135.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     222/1227 [====>.........................] - ETA: 8:25

    100%|██████████| 64/64 [00:00<00:00, 129.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     223/1227 [====>.........................] - ETA: 8:24

    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     224/1227 [====>.........................] - ETA: 8:24

    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     225/1227 [====>.........................] - ETA: 8:23

    100%|██████████| 64/64 [00:00<00:00, 131.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     226/1227 [====>.........................] - ETA: 8:23

    100%|██████████| 64/64 [00:00<00:00, 127.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     227/1227 [====>.........................] - ETA: 8:22

    100%|██████████| 64/64 [00:00<00:00, 128.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     228/1227 [====>.........................] - ETA: 8:22

    100%|██████████| 64/64 [00:00<00:00, 128.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     229/1227 [====>.........................] - ETA: 8:21

    100%|██████████| 64/64 [00:00<00:00, 120.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     230/1227 [====>.........................] - ETA: 8:21

    100%|██████████| 64/64 [00:00<00:00, 121.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     231/1227 [====>.........................] - ETA: 8:21

    100%|██████████| 64/64 [00:00<00:00, 120.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     232/1227 [====>.........................] - ETA: 8:20

    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     233/1227 [====>.........................] - ETA: 8:20

    100%|██████████| 64/64 [00:00<00:00, 127.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     234/1227 [====>.........................] - ETA: 8:19

    100%|██████████| 64/64 [00:00<00:00, 122.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     235/1227 [====>.........................] - ETA: 8:19

    100%|██████████| 64/64 [00:00<00:00, 123.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     236/1227 [====>.........................] - ETA: 8:18

    100%|██████████| 64/64 [00:00<00:00, 114.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     237/1227 [====>.........................] - ETA: 8:18

    100%|██████████| 64/64 [00:00<00:00, 125.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     238/1227 [====>.........................] - ETA: 8:18

    100%|██████████| 64/64 [00:00<00:00, 121.78it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     239/1227 [====>.........................] - ETA: 8:17

    100%|██████████| 64/64 [00:00<00:00, 125.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     240/1227 [====>.........................] - ETA: 8:17

    100%|██████████| 64/64 [00:00<00:00, 120.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     241/1227 [====>.........................] - ETA: 8:17

    100%|██████████| 64/64 [00:00<00:00, 127.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     242/1227 [====>.........................] - ETA: 8:16

    100%|██████████| 64/64 [00:00<00:00, 124.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     243/1227 [====>.........................] - ETA: 8:16

    100%|██████████| 64/64 [00:00<00:00, 124.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     244/1227 [====>.........................] - ETA: 8:15

    100%|██████████| 64/64 [00:00<00:00, 126.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     245/1227 [====>.........................] - ETA: 8:15

    100%|██████████| 64/64 [00:00<00:00, 124.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     246/1227 [=====>........................] - ETA: 8:14

    100%|██████████| 64/64 [00:00<00:00, 123.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     247/1227 [=====>........................] - ETA: 8:14

    100%|██████████| 64/64 [00:00<00:00, 115.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     248/1227 [=====>........................] - ETA: 8:14

    100%|██████████| 64/64 [00:00<00:00, 120.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     249/1227 [=====>........................] - ETA: 8:13

    100%|██████████| 64/64 [00:00<00:00, 121.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     250/1227 [=====>........................] - ETA: 8:13

    100%|██████████| 64/64 [00:00<00:00, 101.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     251/1227 [=====>........................] - ETA: 8:13

    100%|██████████| 64/64 [00:00<00:00, 119.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     252/1227 [=====>........................] - ETA: 8:13

    100%|██████████| 64/64 [00:00<00:00, 132.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     253/1227 [=====>........................] - ETA: 8:12

    100%|██████████| 64/64 [00:00<00:00, 118.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     254/1227 [=====>........................] - ETA: 8:12

    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     255/1227 [=====>........................] - ETA: 8:11

    100%|██████████| 64/64 [00:00<00:00, 127.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     256/1227 [=====>........................] - ETA: 8:11

    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     257/1227 [=====>........................] - ETA: 8:10

    100%|██████████| 64/64 [00:00<00:00, 130.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     258/1227 [=====>........................] - ETA: 8:10

    100%|██████████| 64/64 [00:00<00:00, 142.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     259/1227 [=====>........................] - ETA: 8:09

    100%|██████████| 64/64 [00:00<00:00, 126.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     260/1227 [=====>........................] - ETA: 8:08

    100%|██████████| 64/64 [00:00<00:00, 138.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     261/1227 [=====>........................] - ETA: 8:08

    100%|██████████| 64/64 [00:00<00:00, 124.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     262/1227 [=====>........................] - ETA: 8:07

    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     263/1227 [=====>........................] - ETA: 8:07

    100%|██████████| 64/64 [00:00<00:00, 141.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     264/1227 [=====>........................] - ETA: 8:06

    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     265/1227 [=====>........................] - ETA: 8:06

    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     266/1227 [=====>........................] - ETA: 8:05

    100%|██████████| 64/64 [00:00<00:00, 127.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     267/1227 [=====>........................] - ETA: 8:05

    100%|██████████| 64/64 [00:00<00:00, 129.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     268/1227 [=====>........................] - ETA: 8:04

    100%|██████████| 64/64 [00:00<00:00, 129.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     269/1227 [=====>........................] - ETA: 8:04

    100%|██████████| 64/64 [00:00<00:00, 129.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     270/1227 [=====>........................] - ETA: 8:03

    100%|██████████| 64/64 [00:00<00:00, 130.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     271/1227 [=====>........................] - ETA: 8:03

    100%|██████████| 64/64 [00:00<00:00, 132.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     272/1227 [=====>........................] - ETA: 8:02

    100%|██████████| 64/64 [00:00<00:00, 120.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     273/1227 [=====>........................] - ETA: 8:02

    100%|██████████| 64/64 [00:00<00:00, 128.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     274/1227 [=====>........................] - ETA: 8:01

    100%|██████████| 64/64 [00:00<00:00, 125.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     275/1227 [=====>........................] - ETA: 8:01

    100%|██████████| 64/64 [00:00<00:00, 126.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     276/1227 [=====>........................] - ETA: 8:00

    100%|██████████| 64/64 [00:00<00:00, 122.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     277/1227 [=====>........................] - ETA: 8:00

    100%|██████████| 64/64 [00:00<00:00, 131.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     278/1227 [=====>........................] - ETA: 7:59

    100%|██████████| 64/64 [00:00<00:00, 131.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     279/1227 [=====>........................] - ETA: 7:59

    100%|██████████| 64/64 [00:00<00:00, 128.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     280/1227 [=====>........................] - ETA: 7:58

    100%|██████████| 64/64 [00:00<00:00, 128.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     281/1227 [=====>........................] - ETA: 7:58

    100%|██████████| 64/64 [00:00<00:00, 127.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     282/1227 [=====>........................] - ETA: 7:57

    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     283/1227 [=====>........................] - ETA: 7:57

    100%|██████████| 64/64 [00:00<00:00, 129.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     284/1227 [=====>........................] - ETA: 7:56

    100%|██████████| 64/64 [00:00<00:00, 123.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     285/1227 [=====>........................] - ETA: 7:56

    100%|██████████| 64/64 [00:00<00:00, 129.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     286/1227 [=====>........................] - ETA: 7:55

    100%|██████████| 64/64 [00:00<00:00, 132.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     287/1227 [======>.......................] - ETA: 7:55

    100%|██████████| 64/64 [00:00<00:00, 128.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     288/1227 [======>.......................] - ETA: 7:54

    100%|██████████| 64/64 [00:00<00:00, 129.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     289/1227 [======>.......................] - ETA: 7:54

    100%|██████████| 64/64 [00:00<00:00, 131.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     290/1227 [======>.......................] - ETA: 7:53

    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     291/1227 [======>.......................] - ETA: 7:53

    100%|██████████| 64/64 [00:00<00:00, 125.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     292/1227 [======>.......................] - ETA: 7:52

    100%|██████████| 64/64 [00:00<00:00, 127.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     293/1227 [======>.......................] - ETA: 7:52

    100%|██████████| 64/64 [00:00<00:00, 130.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     294/1227 [======>.......................] - ETA: 7:51

    100%|██████████| 64/64 [00:00<00:00, 127.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     295/1227 [======>.......................] - ETA: 7:51

    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     296/1227 [======>.......................] - ETA: 7:50

    100%|██████████| 64/64 [00:00<00:00, 125.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     297/1227 [======>.......................] - ETA: 7:50

    100%|██████████| 64/64 [00:00<00:00, 125.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     298/1227 [======>.......................] - ETA: 7:49

    100%|██████████| 64/64 [00:00<00:00, 139.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     299/1227 [======>.......................] - ETA: 7:49

    100%|██████████| 64/64 [00:00<00:00, 133.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     300/1227 [======>.......................] - ETA: 7:48

    100%|██████████| 64/64 [00:00<00:00, 130.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     301/1227 [======>.......................] - ETA: 7:48

    100%|██████████| 64/64 [00:00<00:00, 128.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     302/1227 [======>.......................] - ETA: 7:47

    100%|██████████| 64/64 [00:00<00:00, 130.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     303/1227 [======>.......................] - ETA: 7:47

    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     304/1227 [======>.......................] - ETA: 7:46

    100%|██████████| 64/64 [00:00<00:00, 131.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     305/1227 [======>.......................] - ETA: 7:45

    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     306/1227 [======>.......................] - ETA: 7:45

    100%|██████████| 64/64 [00:00<00:00, 130.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     307/1227 [======>.......................] - ETA: 7:44

    100%|██████████| 64/64 [00:00<00:00, 126.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     308/1227 [======>.......................] - ETA: 7:44

    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     309/1227 [======>.......................] - ETA: 7:43

    100%|██████████| 64/64 [00:00<00:00, 127.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     310/1227 [======>.......................] - ETA: 7:43

    100%|██████████| 64/64 [00:00<00:00, 130.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     311/1227 [======>.......................] - ETA: 7:42

    100%|██████████| 64/64 [00:00<00:00, 124.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     312/1227 [======>.......................] - ETA: 7:42

    100%|██████████| 64/64 [00:00<00:00, 131.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     313/1227 [======>.......................] - ETA: 7:41

    100%|██████████| 64/64 [00:00<00:00, 132.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     314/1227 [======>.......................] - ETA: 7:41

    100%|██████████| 64/64 [00:00<00:00, 134.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     315/1227 [======>.......................] - ETA: 7:40

    100%|██████████| 64/64 [00:00<00:00, 124.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     316/1227 [======>.......................] - ETA: 7:40

    100%|██████████| 64/64 [00:00<00:00, 135.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     317/1227 [======>.......................] - ETA: 7:39

    100%|██████████| 64/64 [00:00<00:00, 120.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     318/1227 [======>.......................] - ETA: 7:39

    100%|██████████| 64/64 [00:00<00:00, 127.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     319/1227 [======>.......................] - ETA: 7:38

    100%|██████████| 64/64 [00:00<00:00, 122.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     320/1227 [======>.......................] - ETA: 7:38

    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     321/1227 [======>.......................] - ETA: 7:37

    100%|██████████| 64/64 [00:00<00:00, 127.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     322/1227 [======>.......................] - ETA: 7:37

    100%|██████████| 64/64 [00:00<00:00, 131.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     323/1227 [======>.......................] - ETA: 7:36

    100%|██████████| 64/64 [00:00<00:00, 130.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     324/1227 [======>.......................] - ETA: 7:36

    100%|██████████| 64/64 [00:00<00:00, 131.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     325/1227 [======>.......................] - ETA: 7:35

    100%|██████████| 64/64 [00:00<00:00, 133.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     326/1227 [======>.......................] - ETA: 7:35

    100%|██████████| 64/64 [00:00<00:00, 131.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     327/1227 [======>.......................] - ETA: 7:34

    100%|██████████| 64/64 [00:00<00:00, 123.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     328/1227 [=======>......................] - ETA: 7:34

    100%|██████████| 64/64 [00:00<00:00, 124.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     329/1227 [=======>......................] - ETA: 7:33

    100%|██████████| 64/64 [00:00<00:00, 130.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     330/1227 [=======>......................] - ETA: 7:33

    100%|██████████| 64/64 [00:00<00:00, 133.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     331/1227 [=======>......................] - ETA: 7:32

    100%|██████████| 64/64 [00:00<00:00, 130.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     332/1227 [=======>......................] - ETA: 7:32

    100%|██████████| 64/64 [00:00<00:00, 125.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     333/1227 [=======>......................] - ETA: 7:31

    100%|██████████| 64/64 [00:00<00:00, 115.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     334/1227 [=======>......................] - ETA: 7:31

    100%|██████████| 64/64 [00:00<00:00, 119.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     335/1227 [=======>......................] - ETA: 7:31

    100%|██████████| 64/64 [00:00<00:00, 125.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     336/1227 [=======>......................] - ETA: 7:30

    100%|██████████| 64/64 [00:00<00:00, 126.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     337/1227 [=======>......................] - ETA: 7:30

    100%|██████████| 64/64 [00:00<00:00, 128.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     338/1227 [=======>......................] - ETA: 7:29

    100%|██████████| 64/64 [00:00<00:00, 130.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     339/1227 [=======>......................] - ETA: 7:29

    100%|██████████| 64/64 [00:00<00:00, 127.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     340/1227 [=======>......................] - ETA: 7:28

    100%|██████████| 64/64 [00:00<00:00, 126.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     341/1227 [=======>......................] - ETA: 7:28

    100%|██████████| 64/64 [00:00<00:00, 128.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     342/1227 [=======>......................] - ETA: 7:27

    100%|██████████| 64/64 [00:00<00:00, 132.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     343/1227 [=======>......................] - ETA: 7:27

    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     344/1227 [=======>......................] - ETA: 7:26

    100%|██████████| 64/64 [00:00<00:00, 129.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     345/1227 [=======>......................] - ETA: 7:26

    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     346/1227 [=======>......................] - ETA: 7:25

    100%|██████████| 64/64 [00:00<00:00, 125.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     347/1227 [=======>......................] - ETA: 7:25

    100%|██████████| 64/64 [00:00<00:00, 130.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     348/1227 [=======>......................] - ETA: 7:24

    100%|██████████| 64/64 [00:00<00:00, 130.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     349/1227 [=======>......................] - ETA: 7:24

    100%|██████████| 64/64 [00:00<00:00, 134.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     350/1227 [=======>......................] - ETA: 7:23

    100%|██████████| 64/64 [00:00<00:00, 127.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     351/1227 [=======>......................] - ETA: 7:22

    100%|██████████| 64/64 [00:00<00:00, 130.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     352/1227 [=======>......................] - ETA: 7:22

    100%|██████████| 64/64 [00:00<00:00, 126.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     353/1227 [=======>......................] - ETA: 7:21

    100%|██████████| 64/64 [00:00<00:00, 120.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     354/1227 [=======>......................] - ETA: 7:21

    100%|██████████| 64/64 [00:00<00:00, 122.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     355/1227 [=======>......................] - ETA: 7:21

    100%|██████████| 64/64 [00:00<00:00, 129.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     356/1227 [=======>......................] - ETA: 7:20

    100%|██████████| 64/64 [00:00<00:00, 123.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     357/1227 [=======>......................] - ETA: 7:20

    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     358/1227 [=======>......................] - ETA: 7:19

    100%|██████████| 64/64 [00:00<00:00, 124.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     359/1227 [=======>......................] - ETA: 7:19

    100%|██████████| 64/64 [00:00<00:00, 130.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     360/1227 [=======>......................] - ETA: 7:18

    100%|██████████| 64/64 [00:00<00:00, 131.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     361/1227 [=======>......................] - ETA: 7:18

    100%|██████████| 64/64 [00:00<00:00, 131.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     362/1227 [=======>......................] - ETA: 7:17

    100%|██████████| 64/64 [00:00<00:00, 126.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     363/1227 [=======>......................] - ETA: 7:17

    100%|██████████| 64/64 [00:00<00:00, 128.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     364/1227 [=======>......................] - ETA: 7:16

    100%|██████████| 64/64 [00:00<00:00, 127.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     365/1227 [=======>......................] - ETA: 7:16

    100%|██████████| 64/64 [00:00<00:00, 127.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     366/1227 [=======>......................] - ETA: 7:15

    100%|██████████| 64/64 [00:00<00:00, 132.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     367/1227 [=======>......................] - ETA: 7:15

    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     368/1227 [=======>......................] - ETA: 7:14

    100%|██████████| 64/64 [00:00<00:00, 126.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     369/1227 [========>.....................] - ETA: 7:14

    100%|██████████| 64/64 [00:00<00:00, 129.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     370/1227 [========>.....................] - ETA: 7:13

    100%|██████████| 64/64 [00:00<00:00, 125.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     371/1227 [========>.....................] - ETA: 7:13

    100%|██████████| 64/64 [00:00<00:00, 133.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     372/1227 [========>.....................] - ETA: 7:12

    100%|██████████| 64/64 [00:00<00:00, 128.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     373/1227 [========>.....................] - ETA: 7:12

    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     374/1227 [========>.....................] - ETA: 7:11

    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     375/1227 [========>.....................] - ETA: 7:10

    100%|██████████| 64/64 [00:00<00:00, 132.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     376/1227 [========>.....................] - ETA: 7:10

    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     377/1227 [========>.....................] - ETA: 7:09

    100%|██████████| 64/64 [00:00<00:00, 132.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     378/1227 [========>.....................] - ETA: 7:09

    100%|██████████| 64/64 [00:00<00:00, 129.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     379/1227 [========>.....................] - ETA: 7:08

    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     380/1227 [========>.....................] - ETA: 7:08

    100%|██████████| 64/64 [00:00<00:00, 136.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     381/1227 [========>.....................] - ETA: 7:07

    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     382/1227 [========>.....................] - ETA: 7:07

    100%|██████████| 64/64 [00:00<00:00, 133.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     383/1227 [========>.....................] - ETA: 7:06

    100%|██████████| 64/64 [00:00<00:00, 129.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     384/1227 [========>.....................] - ETA: 7:06

    100%|██████████| 64/64 [00:00<00:00, 127.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     385/1227 [========>.....................] - ETA: 7:05

    100%|██████████| 64/64 [00:00<00:00, 129.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     386/1227 [========>.....................] - ETA: 7:05

    100%|██████████| 64/64 [00:00<00:00, 129.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     387/1227 [========>.....................] - ETA: 7:04

    100%|██████████| 64/64 [00:00<00:00, 130.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     388/1227 [========>.....................] - ETA: 7:04

    100%|██████████| 64/64 [00:00<00:00, 127.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     389/1227 [========>.....................] - ETA: 7:03

    100%|██████████| 64/64 [00:00<00:00, 131.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     390/1227 [========>.....................] - ETA: 7:03

    100%|██████████| 64/64 [00:00<00:00, 131.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     391/1227 [========>.....................] - ETA: 7:02

    100%|██████████| 64/64 [00:00<00:00, 125.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     392/1227 [========>.....................] - ETA: 7:02

    100%|██████████| 64/64 [00:00<00:00, 133.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     393/1227 [========>.....................] - ETA: 7:01

    100%|██████████| 64/64 [00:00<00:00, 126.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     394/1227 [========>.....................] - ETA: 7:01

    100%|██████████| 64/64 [00:00<00:00, 127.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     395/1227 [========>.....................] - ETA: 7:00

    100%|██████████| 64/64 [00:00<00:00, 125.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     396/1227 [========>.....................] - ETA: 7:00

    100%|██████████| 64/64 [00:00<00:00, 131.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     397/1227 [========>.....................] - ETA: 6:59

    100%|██████████| 64/64 [00:00<00:00, 126.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     398/1227 [========>.....................] - ETA: 6:59

    100%|██████████| 64/64 [00:00<00:00, 131.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     399/1227 [========>.....................] - ETA: 6:58

    100%|██████████| 64/64 [00:00<00:00, 122.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     400/1227 [========>.....................] - ETA: 6:58

    100%|██████████| 64/64 [00:00<00:00, 129.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     401/1227 [========>.....................] - ETA: 6:57

    100%|██████████| 64/64 [00:00<00:00, 127.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     402/1227 [========>.....................] - ETA: 6:57

    100%|██████████| 64/64 [00:00<00:00, 128.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     403/1227 [========>.....................] - ETA: 6:56

    100%|██████████| 64/64 [00:00<00:00, 125.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     404/1227 [========>.....................] - ETA: 6:56

    100%|██████████| 64/64 [00:00<00:00, 131.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     405/1227 [========>.....................] - ETA: 6:55

    100%|██████████| 64/64 [00:00<00:00, 130.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     406/1227 [========>.....................] - ETA: 6:55

    100%|██████████| 64/64 [00:00<00:00, 129.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     407/1227 [========>.....................] - ETA: 6:54

    100%|██████████| 64/64 [00:00<00:00, 123.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     408/1227 [========>.....................] - ETA: 6:54

    100%|██████████| 64/64 [00:00<00:00, 123.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     409/1227 [========>.....................] - ETA: 6:53

    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     410/1227 [=========>....................] - ETA: 6:53

    100%|██████████| 64/64 [00:00<00:00, 129.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     411/1227 [=========>....................] - ETA: 6:52

    100%|██████████| 64/64 [00:00<00:00, 130.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     412/1227 [=========>....................] - ETA: 6:52

    100%|██████████| 64/64 [00:00<00:00, 134.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     413/1227 [=========>....................] - ETA: 6:51

    100%|██████████| 64/64 [00:00<00:00, 131.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     414/1227 [=========>....................] - ETA: 6:51

    100%|██████████| 64/64 [00:00<00:00, 130.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     415/1227 [=========>....................] - ETA: 6:50

    100%|██████████| 64/64 [00:00<00:00, 132.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     416/1227 [=========>....................] - ETA: 6:50

    100%|██████████| 64/64 [00:00<00:00, 130.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     417/1227 [=========>....................] - ETA: 6:49

    100%|██████████| 64/64 [00:00<00:00, 131.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     418/1227 [=========>....................] - ETA: 6:49

    100%|██████████| 64/64 [00:00<00:00, 126.57it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     419/1227 [=========>....................] - ETA: 6:48

    100%|██████████| 64/64 [00:00<00:00, 129.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     420/1227 [=========>....................] - ETA: 6:47

    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     421/1227 [=========>....................] - ETA: 6:47

    100%|██████████| 64/64 [00:00<00:00, 134.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     422/1227 [=========>....................] - ETA: 6:46

    100%|██████████| 64/64 [00:00<00:00, 131.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     423/1227 [=========>....................] - ETA: 6:46

    100%|██████████| 64/64 [00:00<00:00, 132.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     424/1227 [=========>....................] - ETA: 6:45

    100%|██████████| 64/64 [00:00<00:00, 131.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     425/1227 [=========>....................] - ETA: 6:45

    100%|██████████| 64/64 [00:00<00:00, 130.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     426/1227 [=========>....................] - ETA: 6:44

    100%|██████████| 64/64 [00:00<00:00, 126.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     427/1227 [=========>....................] - ETA: 6:44

    100%|██████████| 64/64 [00:00<00:00, 132.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     428/1227 [=========>....................] - ETA: 6:43

    100%|██████████| 64/64 [00:00<00:00, 123.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     429/1227 [=========>....................] - ETA: 6:43

    100%|██████████| 64/64 [00:00<00:00, 130.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     430/1227 [=========>....................] - ETA: 6:42

    100%|██████████| 64/64 [00:00<00:00, 126.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     431/1227 [=========>....................] - ETA: 6:42

    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     432/1227 [=========>....................] - ETA: 6:41

    100%|██████████| 64/64 [00:00<00:00, 135.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     433/1227 [=========>....................] - ETA: 6:41

    100%|██████████| 64/64 [00:00<00:00, 135.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     434/1227 [=========>....................] - ETA: 6:40

    100%|██████████| 64/64 [00:00<00:00, 132.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     435/1227 [=========>....................] - ETA: 6:40

    100%|██████████| 64/64 [00:00<00:00, 132.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     436/1227 [=========>....................] - ETA: 6:39

    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     437/1227 [=========>....................] - ETA: 6:39

    100%|██████████| 64/64 [00:00<00:00, 134.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     438/1227 [=========>....................] - ETA: 6:38

    100%|██████████| 64/64 [00:00<00:00, 131.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     439/1227 [=========>....................] - ETA: 6:38

    100%|██████████| 64/64 [00:00<00:00, 136.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     440/1227 [=========>....................] - ETA: 6:37

    100%|██████████| 64/64 [00:00<00:00, 132.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     441/1227 [=========>....................] - ETA: 6:37

    100%|██████████| 64/64 [00:00<00:00, 132.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     442/1227 [=========>....................] - ETA: 6:36

    100%|██████████| 64/64 [00:00<00:00, 133.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     443/1227 [=========>....................] - ETA: 6:35

    100%|██████████| 64/64 [00:00<00:00, 133.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     444/1227 [=========>....................] - ETA: 6:35

    100%|██████████| 64/64 [00:00<00:00, 132.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     445/1227 [=========>....................] - ETA: 6:34

    100%|██████████| 64/64 [00:00<00:00, 132.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     446/1227 [=========>....................] - ETA: 6:34

    100%|██████████| 64/64 [00:00<00:00, 136.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     447/1227 [=========>....................] - ETA: 6:33

    100%|██████████| 64/64 [00:00<00:00, 130.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     448/1227 [=========>....................] - ETA: 6:33

    100%|██████████| 64/64 [00:00<00:00, 133.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     449/1227 [=========>....................] - ETA: 6:32

    100%|██████████| 64/64 [00:00<00:00, 132.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     450/1227 [==========>...................] - ETA: 6:32

    100%|██████████| 64/64 [00:00<00:00, 133.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     451/1227 [==========>...................] - ETA: 6:31

    100%|██████████| 64/64 [00:00<00:00, 127.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     452/1227 [==========>...................] - ETA: 6:31

    100%|██████████| 64/64 [00:00<00:00, 134.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     453/1227 [==========>...................] - ETA: 6:30

    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     454/1227 [==========>...................] - ETA: 6:30

    100%|██████████| 64/64 [00:00<00:00, 129.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     455/1227 [==========>...................] - ETA: 6:29

    100%|██████████| 64/64 [00:00<00:00, 130.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     456/1227 [==========>...................] - ETA: 6:29

    100%|██████████| 64/64 [00:00<00:00, 136.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     457/1227 [==========>...................] - ETA: 6:28

    100%|██████████| 64/64 [00:00<00:00, 134.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     458/1227 [==========>...................] - ETA: 6:27

    100%|██████████| 64/64 [00:00<00:00, 132.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     459/1227 [==========>...................] - ETA: 6:27

    100%|██████████| 64/64 [00:00<00:00, 133.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     460/1227 [==========>...................] - ETA: 6:26

    100%|██████████| 64/64 [00:00<00:00, 128.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     461/1227 [==========>...................] - ETA: 6:26

    100%|██████████| 64/64 [00:00<00:00, 136.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     462/1227 [==========>...................] - ETA: 6:25

    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     463/1227 [==========>...................] - ETA: 6:25

    100%|██████████| 64/64 [00:00<00:00, 140.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     464/1227 [==========>...................] - ETA: 6:24

    100%|██████████| 64/64 [00:00<00:00, 140.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     465/1227 [==========>...................] - ETA: 6:24

    100%|██████████| 64/64 [00:00<00:00, 138.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     466/1227 [==========>...................] - ETA: 6:23

    100%|██████████| 64/64 [00:00<00:00, 133.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     467/1227 [==========>...................] - ETA: 6:23

    100%|██████████| 64/64 [00:00<00:00, 136.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     468/1227 [==========>...................] - ETA: 6:22

    100%|██████████| 64/64 [00:00<00:00, 127.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     469/1227 [==========>...................] - ETA: 6:22

    100%|██████████| 64/64 [00:00<00:00, 134.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     470/1227 [==========>...................] - ETA: 6:21

    100%|██████████| 64/64 [00:00<00:00, 128.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     471/1227 [==========>...................] - ETA: 6:21

    100%|██████████| 64/64 [00:00<00:00, 130.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     472/1227 [==========>...................] - ETA: 6:20

    100%|██████████| 64/64 [00:00<00:00, 133.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     473/1227 [==========>...................] - ETA: 6:19

    100%|██████████| 64/64 [00:00<00:00, 135.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     474/1227 [==========>...................] - ETA: 6:19

    100%|██████████| 64/64 [00:00<00:00, 128.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     475/1227 [==========>...................] - ETA: 6:18

    100%|██████████| 64/64 [00:00<00:00, 136.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     476/1227 [==========>...................] - ETA: 6:18

    100%|██████████| 64/64 [00:00<00:00, 132.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     477/1227 [==========>...................] - ETA: 6:17

    100%|██████████| 64/64 [00:00<00:00, 126.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     478/1227 [==========>...................] - ETA: 6:17

    100%|██████████| 64/64 [00:00<00:00, 128.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     479/1227 [==========>...................] - ETA: 6:16

    100%|██████████| 64/64 [00:00<00:00, 127.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     480/1227 [==========>...................] - ETA: 6:16

    100%|██████████| 64/64 [00:00<00:00, 131.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     481/1227 [==========>...................] - ETA: 6:15

    100%|██████████| 64/64 [00:00<00:00, 129.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     482/1227 [==========>...................] - ETA: 6:15

    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     483/1227 [==========>...................] - ETA: 6:14

    100%|██████████| 64/64 [00:00<00:00, 120.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     484/1227 [==========>...................] - ETA: 6:14

    100%|██████████| 64/64 [00:00<00:00, 128.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     485/1227 [==========>...................] - ETA: 6:13

    100%|██████████| 64/64 [00:00<00:00, 120.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     486/1227 [==========>...................] - ETA: 6:13

    100%|██████████| 64/64 [00:00<00:00, 122.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     487/1227 [==========>...................] - ETA: 6:12

    100%|██████████| 64/64 [00:00<00:00, 125.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     488/1227 [==========>...................] - ETA: 6:12

    100%|██████████| 64/64 [00:00<00:00, 138.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     489/1227 [==========>...................] - ETA: 6:11

    100%|██████████| 64/64 [00:00<00:00, 135.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     490/1227 [==========>...................] - ETA: 6:11

    100%|██████████| 64/64 [00:00<00:00, 138.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     491/1227 [===========>..................] - ETA: 6:10

    100%|██████████| 64/64 [00:00<00:00, 128.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     492/1227 [===========>..................] - ETA: 6:10

    100%|██████████| 64/64 [00:00<00:00, 133.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     493/1227 [===========>..................] - ETA: 6:09

    100%|██████████| 64/64 [00:00<00:00, 130.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     494/1227 [===========>..................] - ETA: 6:09

    100%|██████████| 64/64 [00:00<00:00, 136.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     495/1227 [===========>..................] - ETA: 6:08

    100%|██████████| 64/64 [00:00<00:00, 128.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     496/1227 [===========>..................] - ETA: 6:08

    100%|██████████| 64/64 [00:00<00:00, 135.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     497/1227 [===========>..................] - ETA: 6:07

    100%|██████████| 64/64 [00:00<00:00, 126.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     498/1227 [===========>..................] - ETA: 6:07

    100%|██████████| 64/64 [00:00<00:00, 135.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     499/1227 [===========>..................] - ETA: 6:06

    100%|██████████| 64/64 [00:00<00:00, 136.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     500/1227 [===========>..................] - ETA: 6:06

    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     501/1227 [===========>..................] - ETA: 6:05

    100%|██████████| 64/64 [00:00<00:00, 133.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     502/1227 [===========>..................] - ETA: 6:05

    100%|██████████| 64/64 [00:00<00:00, 135.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     503/1227 [===========>..................] - ETA: 6:04

    100%|██████████| 64/64 [00:00<00:00, 134.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     504/1227 [===========>..................] - ETA: 6:04

    100%|██████████| 64/64 [00:00<00:00, 134.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     505/1227 [===========>..................] - ETA: 6:03

    100%|██████████| 64/64 [00:00<00:00, 137.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     506/1227 [===========>..................] - ETA: 6:02

    100%|██████████| 64/64 [00:00<00:00, 132.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     507/1227 [===========>..................] - ETA: 6:02

    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     508/1227 [===========>..................] - ETA: 6:01

    100%|██████████| 64/64 [00:00<00:00, 131.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     509/1227 [===========>..................] - ETA: 6:01

    100%|██████████| 64/64 [00:00<00:00, 135.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     510/1227 [===========>..................] - ETA: 6:00

    100%|██████████| 64/64 [00:00<00:00, 133.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     511/1227 [===========>..................] - ETA: 6:00

    100%|██████████| 64/64 [00:00<00:00, 134.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     512/1227 [===========>..................] - ETA: 5:59

    100%|██████████| 64/64 [00:00<00:00, 133.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     513/1227 [===========>..................] - ETA: 5:59

    100%|██████████| 64/64 [00:00<00:00, 137.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     514/1227 [===========>..................] - ETA: 5:58

    100%|██████████| 64/64 [00:00<00:00, 134.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     515/1227 [===========>..................] - ETA: 5:58

    100%|██████████| 64/64 [00:00<00:00, 136.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     516/1227 [===========>..................] - ETA: 5:57

    100%|██████████| 64/64 [00:00<00:00, 132.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     517/1227 [===========>..................] - ETA: 5:57

    100%|██████████| 64/64 [00:00<00:00, 138.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     518/1227 [===========>..................] - ETA: 5:56

    100%|██████████| 64/64 [00:00<00:00, 137.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     519/1227 [===========>..................] - ETA: 5:56

    100%|██████████| 64/64 [00:00<00:00, 133.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     520/1227 [===========>..................] - ETA: 5:55

    100%|██████████| 64/64 [00:00<00:00, 139.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     521/1227 [===========>..................] - ETA: 5:54

    100%|██████████| 64/64 [00:00<00:00, 129.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     522/1227 [===========>..................] - ETA: 5:54

    100%|██████████| 64/64 [00:00<00:00, 136.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     523/1227 [===========>..................] - ETA: 5:53

    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     524/1227 [===========>..................] - ETA: 5:53

    100%|██████████| 64/64 [00:00<00:00, 134.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     525/1227 [===========>..................] - ETA: 5:52

    100%|██████████| 64/64 [00:00<00:00, 136.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     526/1227 [===========>..................] - ETA: 5:52

    100%|██████████| 64/64 [00:00<00:00, 142.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     527/1227 [===========>..................] - ETA: 5:51

    100%|██████████| 64/64 [00:00<00:00, 128.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     528/1227 [===========>..................] - ETA: 5:51

    100%|██████████| 64/64 [00:00<00:00, 135.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     529/1227 [===========>..................] - ETA: 5:50

    100%|██████████| 64/64 [00:00<00:00, 131.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     530/1227 [===========>..................] - ETA: 5:50

    100%|██████████| 64/64 [00:00<00:00, 139.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     531/1227 [===========>..................] - ETA: 5:49

    100%|██████████| 64/64 [00:00<00:00, 135.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     532/1227 [============>.................] - ETA: 5:49

    100%|██████████| 64/64 [00:00<00:00, 133.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     533/1227 [============>.................] - ETA: 5:48

    100%|██████████| 64/64 [00:00<00:00, 141.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     534/1227 [============>.................] - ETA: 5:48

    100%|██████████| 64/64 [00:00<00:00, 130.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     535/1227 [============>.................] - ETA: 5:47

    100%|██████████| 64/64 [00:00<00:00, 136.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     536/1227 [============>.................] - ETA: 5:47

    100%|██████████| 64/64 [00:00<00:00, 137.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     537/1227 [============>.................] - ETA: 5:46

    100%|██████████| 64/64 [00:00<00:00, 141.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     538/1227 [============>.................] - ETA: 5:45

    100%|██████████| 64/64 [00:00<00:00, 134.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     539/1227 [============>.................] - ETA: 5:45

    100%|██████████| 64/64 [00:00<00:00, 135.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     540/1227 [============>.................] - ETA: 5:44

    100%|██████████| 64/64 [00:00<00:00, 135.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     541/1227 [============>.................] - ETA: 5:44

    100%|██████████| 64/64 [00:00<00:00, 140.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     542/1227 [============>.................] - ETA: 5:43

    100%|██████████| 64/64 [00:00<00:00, 132.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     543/1227 [============>.................] - ETA: 5:43

    100%|██████████| 64/64 [00:00<00:00, 135.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     544/1227 [============>.................] - ETA: 5:42

    100%|██████████| 64/64 [00:00<00:00, 136.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     545/1227 [============>.................] - ETA: 5:42

    100%|██████████| 64/64 [00:00<00:00, 134.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     546/1227 [============>.................] - ETA: 5:41

    100%|██████████| 64/64 [00:00<00:00, 139.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     547/1227 [============>.................] - ETA: 5:41

    100%|██████████| 64/64 [00:00<00:00, 136.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     548/1227 [============>.................] - ETA: 5:40

    100%|██████████| 64/64 [00:00<00:00, 134.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     549/1227 [============>.................] - ETA: 5:40

    100%|██████████| 64/64 [00:00<00:00, 134.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     550/1227 [============>.................] - ETA: 5:39

    100%|██████████| 64/64 [00:00<00:00, 141.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     551/1227 [============>.................] - ETA: 5:39

    100%|██████████| 64/64 [00:00<00:00, 136.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     552/1227 [============>.................] - ETA: 5:38

    100%|██████████| 64/64 [00:00<00:00, 136.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     553/1227 [============>.................] - ETA: 5:38

    100%|██████████| 64/64 [00:00<00:00, 135.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     554/1227 [============>.................] - ETA: 5:37

    100%|██████████| 64/64 [00:00<00:00, 133.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     555/1227 [============>.................] - ETA: 5:36

    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     556/1227 [============>.................] - ETA: 5:36

    100%|██████████| 64/64 [00:00<00:00, 140.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     557/1227 [============>.................] - ETA: 5:35

    100%|██████████| 64/64 [00:00<00:00, 138.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     558/1227 [============>.................] - ETA: 5:35

    100%|██████████| 64/64 [00:00<00:00, 145.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     559/1227 [============>.................] - ETA: 5:34

    100%|██████████| 64/64 [00:00<00:00, 134.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     560/1227 [============>.................] - ETA: 5:34

    100%|██████████| 64/64 [00:00<00:00, 137.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     561/1227 [============>.................] - ETA: 5:33

    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     562/1227 [============>.................] - ETA: 5:33

    100%|██████████| 64/64 [00:00<00:00, 136.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     563/1227 [============>.................] - ETA: 5:32

    100%|██████████| 64/64 [00:00<00:00, 138.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     564/1227 [============>.................] - ETA: 5:32

    100%|██████████| 64/64 [00:00<00:00, 134.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     565/1227 [============>.................] - ETA: 5:31

    100%|██████████| 64/64 [00:00<00:00, 136.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     566/1227 [============>.................] - ETA: 5:31

    100%|██████████| 64/64 [00:00<00:00, 137.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     567/1227 [============>.................] - ETA: 5:30

    100%|██████████| 64/64 [00:00<00:00, 138.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     568/1227 [============>.................] - ETA: 5:30

    100%|██████████| 64/64 [00:00<00:00, 133.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     569/1227 [============>.................] - ETA: 5:29

    100%|██████████| 64/64 [00:00<00:00, 132.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     570/1227 [============>.................] - ETA: 5:29

    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     571/1227 [============>.................] - ETA: 5:28

    100%|██████████| 64/64 [00:00<00:00, 133.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     572/1227 [============>.................] - ETA: 5:28

    100%|██████████| 64/64 [00:00<00:00, 139.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     573/1227 [=============>................] - ETA: 5:27

    100%|██████████| 64/64 [00:00<00:00, 133.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     574/1227 [=============>................] - ETA: 5:27

    100%|██████████| 64/64 [00:00<00:00, 131.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     575/1227 [=============>................] - ETA: 5:26

    100%|██████████| 64/64 [00:00<00:00, 140.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     576/1227 [=============>................] - ETA: 5:25

    100%|██████████| 64/64 [00:00<00:00, 138.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     577/1227 [=============>................] - ETA: 5:25

    100%|██████████| 64/64 [00:00<00:00, 139.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     578/1227 [=============>................] - ETA: 5:24

    100%|██████████| 64/64 [00:00<00:00, 137.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     579/1227 [=============>................] - ETA: 5:24

    100%|██████████| 64/64 [00:00<00:00, 133.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     580/1227 [=============>................] - ETA: 5:23

    100%|██████████| 64/64 [00:00<00:00, 134.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     581/1227 [=============>................] - ETA: 5:23

    100%|██████████| 64/64 [00:00<00:00, 135.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     582/1227 [=============>................] - ETA: 5:22

    100%|██████████| 64/64 [00:00<00:00, 133.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     583/1227 [=============>................] - ETA: 5:22

    100%|██████████| 64/64 [00:00<00:00, 132.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     584/1227 [=============>................] - ETA: 5:21

    100%|██████████| 64/64 [00:00<00:00, 134.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     585/1227 [=============>................] - ETA: 5:21

    100%|██████████| 64/64 [00:00<00:00, 132.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     586/1227 [=============>................] - ETA: 5:20

    100%|██████████| 64/64 [00:00<00:00, 135.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     587/1227 [=============>................] - ETA: 5:20

    100%|██████████| 64/64 [00:00<00:00, 137.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     588/1227 [=============>................] - ETA: 5:19

    100%|██████████| 64/64 [00:00<00:00, 135.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     589/1227 [=============>................] - ETA: 5:19

    100%|██████████| 64/64 [00:00<00:00, 136.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     590/1227 [=============>................] - ETA: 5:18

    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     591/1227 [=============>................] - ETA: 5:18

    100%|██████████| 64/64 [00:00<00:00, 132.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     592/1227 [=============>................] - ETA: 5:17

    100%|██████████| 64/64 [00:00<00:00, 128.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     593/1227 [=============>................] - ETA: 5:17

    100%|██████████| 64/64 [00:00<00:00, 133.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     594/1227 [=============>................] - ETA: 5:16

    100%|██████████| 64/64 [00:00<00:00, 124.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     595/1227 [=============>................] - ETA: 5:16

    100%|██████████| 64/64 [00:00<00:00, 134.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     596/1227 [=============>................] - ETA: 5:15

    100%|██████████| 64/64 [00:00<00:00, 127.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     597/1227 [=============>................] - ETA: 5:15

    100%|██████████| 64/64 [00:00<00:00, 126.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     598/1227 [=============>................] - ETA: 5:14

    100%|██████████| 64/64 [00:00<00:00, 123.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     599/1227 [=============>................] - ETA: 5:14

    100%|██████████| 64/64 [00:00<00:00, 126.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     600/1227 [=============>................] - ETA: 5:13

    100%|██████████| 64/64 [00:00<00:00, 128.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     601/1227 [=============>................] - ETA: 5:13

    100%|██████████| 64/64 [00:00<00:00, 131.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     602/1227 [=============>................] - ETA: 5:12

    100%|██████████| 64/64 [00:00<00:00, 131.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     603/1227 [=============>................] - ETA: 5:12

    100%|██████████| 64/64 [00:00<00:00, 134.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     604/1227 [=============>................] - ETA: 5:11

    100%|██████████| 64/64 [00:00<00:00, 125.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     605/1227 [=============>................] - ETA: 5:11

    100%|██████████| 64/64 [00:00<00:00, 130.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     606/1227 [=============>................] - ETA: 5:10

    100%|██████████| 64/64 [00:00<00:00, 130.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     607/1227 [=============>................] - ETA: 5:10

    100%|██████████| 64/64 [00:00<00:00, 130.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     608/1227 [=============>................] - ETA: 5:09

    100%|██████████| 64/64 [00:00<00:00, 130.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     609/1227 [=============>................] - ETA: 5:09

    100%|██████████| 64/64 [00:00<00:00, 130.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     610/1227 [=============>................] - ETA: 5:08

    100%|██████████| 64/64 [00:00<00:00, 136.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     611/1227 [=============>................] - ETA: 5:08

    100%|██████████| 64/64 [00:00<00:00, 134.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     612/1227 [=============>................] - ETA: 5:07

    100%|██████████| 64/64 [00:00<00:00, 135.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     613/1227 [=============>................] - ETA: 5:07

    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     614/1227 [==============>...............] - ETA: 5:06

    100%|██████████| 64/64 [00:00<00:00, 127.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     615/1227 [==============>...............] - ETA: 5:06

    100%|██████████| 64/64 [00:00<00:00, 134.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     616/1227 [==============>...............] - ETA: 5:05

    100%|██████████| 64/64 [00:00<00:00, 137.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     617/1227 [==============>...............] - ETA: 5:05

    100%|██████████| 64/64 [00:00<00:00, 129.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     618/1227 [==============>...............] - ETA: 5:04

    100%|██████████| 64/64 [00:00<00:00, 133.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     619/1227 [==============>...............] - ETA: 5:04

    100%|██████████| 64/64 [00:00<00:00, 131.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     620/1227 [==============>...............] - ETA: 5:03

    100%|██████████| 64/64 [00:00<00:00, 134.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     621/1227 [==============>...............] - ETA: 5:03

    100%|██████████| 64/64 [00:00<00:00, 131.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     622/1227 [==============>...............] - ETA: 5:02

    100%|██████████| 64/64 [00:00<00:00, 137.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     623/1227 [==============>...............] - ETA: 5:02

    100%|██████████| 64/64 [00:00<00:00, 132.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     624/1227 [==============>...............] - ETA: 5:01

    100%|██████████| 64/64 [00:00<00:00, 133.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     625/1227 [==============>...............] - ETA: 5:01

    100%|██████████| 64/64 [00:00<00:00, 136.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     626/1227 [==============>...............] - ETA: 5:00

    100%|██████████| 64/64 [00:00<00:00, 137.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     627/1227 [==============>...............] - ETA: 4:59

    100%|██████████| 64/64 [00:00<00:00, 138.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     628/1227 [==============>...............] - ETA: 4:59

    100%|██████████| 64/64 [00:00<00:00, 130.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     629/1227 [==============>...............] - ETA: 4:58

    100%|██████████| 64/64 [00:00<00:00, 134.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     630/1227 [==============>...............] - ETA: 4:58

    100%|██████████| 64/64 [00:00<00:00, 126.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     631/1227 [==============>...............] - ETA: 4:57

    100%|██████████| 64/64 [00:00<00:00, 130.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     632/1227 [==============>...............] - ETA: 4:57

    100%|██████████| 64/64 [00:00<00:00, 127.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     633/1227 [==============>...............] - ETA: 4:56

    100%|██████████| 64/64 [00:00<00:00, 134.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     634/1227 [==============>...............] - ETA: 4:56

    100%|██████████| 64/64 [00:00<00:00, 131.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     635/1227 [==============>...............] - ETA: 4:55

    100%|██████████| 64/64 [00:00<00:00, 138.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     636/1227 [==============>...............] - ETA: 4:55

    100%|██████████| 64/64 [00:00<00:00, 135.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     637/1227 [==============>...............] - ETA: 4:54

    100%|██████████| 64/64 [00:00<00:00, 135.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     638/1227 [==============>...............] - ETA: 4:54

    100%|██████████| 64/64 [00:00<00:00, 129.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     639/1227 [==============>...............] - ETA: 4:53

    100%|██████████| 64/64 [00:00<00:00, 129.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     640/1227 [==============>...............] - ETA: 4:53

    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     641/1227 [==============>...............] - ETA: 4:52

    100%|██████████| 64/64 [00:00<00:00, 134.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     642/1227 [==============>...............] - ETA: 4:52

    100%|██████████| 64/64 [00:00<00:00, 131.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     643/1227 [==============>...............] - ETA: 4:51

    100%|██████████| 64/64 [00:00<00:00, 134.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     644/1227 [==============>...............] - ETA: 4:51

    100%|██████████| 64/64 [00:00<00:00, 128.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     645/1227 [==============>...............] - ETA: 4:50

    100%|██████████| 64/64 [00:00<00:00, 127.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     646/1227 [==============>...............] - ETA: 4:50

    100%|██████████| 64/64 [00:00<00:00, 138.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     647/1227 [==============>...............] - ETA: 4:49

    100%|██████████| 64/64 [00:00<00:00, 130.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     648/1227 [==============>...............] - ETA: 4:49

    100%|██████████| 64/64 [00:00<00:00, 137.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     649/1227 [==============>...............] - ETA: 4:48

    100%|██████████| 64/64 [00:00<00:00, 133.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     650/1227 [==============>...............] - ETA: 4:48

    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     651/1227 [==============>...............] - ETA: 4:47

    100%|██████████| 64/64 [00:00<00:00, 128.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     652/1227 [==============>...............] - ETA: 4:47

    100%|██████████| 64/64 [00:00<00:00, 137.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     653/1227 [==============>...............] - ETA: 4:46

    100%|██████████| 64/64 [00:00<00:00, 144.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     654/1227 [==============>...............] - ETA: 4:46

    100%|██████████| 64/64 [00:00<00:00, 143.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     655/1227 [===============>..............] - ETA: 4:45

    100%|██████████| 64/64 [00:00<00:00, 132.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     656/1227 [===============>..............] - ETA: 4:45

    100%|██████████| 64/64 [00:00<00:00, 133.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     657/1227 [===============>..............] - ETA: 4:44

    100%|██████████| 64/64 [00:00<00:00, 128.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     658/1227 [===============>..............] - ETA: 4:44

    100%|██████████| 64/64 [00:00<00:00, 131.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     659/1227 [===============>..............] - ETA: 4:43

    100%|██████████| 64/64 [00:00<00:00, 126.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     660/1227 [===============>..............] - ETA: 4:43

    100%|██████████| 64/64 [00:00<00:00, 132.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     661/1227 [===============>..............] - ETA: 4:42

    100%|██████████| 64/64 [00:00<00:00, 128.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     662/1227 [===============>..............] - ETA: 4:42

    100%|██████████| 64/64 [00:00<00:00, 132.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     663/1227 [===============>..............] - ETA: 4:41

    100%|██████████| 64/64 [00:00<00:00, 131.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     664/1227 [===============>..............] - ETA: 4:41

    100%|██████████| 64/64 [00:00<00:00, 126.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     665/1227 [===============>..............] - ETA: 4:40

    100%|██████████| 64/64 [00:00<00:00, 132.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     666/1227 [===============>..............] - ETA: 4:40

    100%|██████████| 64/64 [00:00<00:00, 128.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     667/1227 [===============>..............] - ETA: 4:39

    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     668/1227 [===============>..............] - ETA: 4:39

    100%|██████████| 64/64 [00:00<00:00, 127.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     669/1227 [===============>..............] - ETA: 4:38

    100%|██████████| 64/64 [00:00<00:00, 133.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     670/1227 [===============>..............] - ETA: 4:38

    100%|██████████| 64/64 [00:00<00:00, 126.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     671/1227 [===============>..............] - ETA: 4:37

    100%|██████████| 64/64 [00:00<00:00, 127.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     672/1227 [===============>..............] - ETA: 4:37

    100%|██████████| 64/64 [00:00<00:00, 127.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     673/1227 [===============>..............] - ETA: 4:36

    100%|██████████| 64/64 [00:00<00:00, 129.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     674/1227 [===============>..............] - ETA: 4:36

    100%|██████████| 64/64 [00:00<00:00, 128.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     675/1227 [===============>..............] - ETA: 4:35

    100%|██████████| 64/64 [00:00<00:00, 131.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     676/1227 [===============>..............] - ETA: 4:35

    100%|██████████| 64/64 [00:00<00:00, 133.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     677/1227 [===============>..............] - ETA: 4:34

    100%|██████████| 64/64 [00:00<00:00, 135.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     678/1227 [===============>..............] - ETA: 4:34

    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     679/1227 [===============>..............] - ETA: 4:33

    100%|██████████| 64/64 [00:00<00:00, 97.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     680/1227 [===============>..............] - ETA: 4:33

    100%|██████████| 64/64 [00:00<00:00, 119.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     681/1227 [===============>..............] - ETA: 4:32

    100%|██████████| 64/64 [00:00<00:00, 129.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     682/1227 [===============>..............] - ETA: 4:32

    100%|██████████| 64/64 [00:00<00:00, 127.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     683/1227 [===============>..............] - ETA: 4:31

    100%|██████████| 64/64 [00:00<00:00, 123.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     684/1227 [===============>..............] - ETA: 4:31

    100%|██████████| 64/64 [00:00<00:00, 124.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     685/1227 [===============>..............] - ETA: 4:30

    100%|██████████| 64/64 [00:00<00:00, 130.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     686/1227 [===============>..............] - ETA: 4:30

    100%|██████████| 64/64 [00:00<00:00, 126.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     687/1227 [===============>..............] - ETA: 4:29

    100%|██████████| 64/64 [00:00<00:00, 128.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     688/1227 [===============>..............] - ETA: 4:29

    100%|██████████| 64/64 [00:00<00:00, 127.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     689/1227 [===============>..............] - ETA: 4:28

    100%|██████████| 64/64 [00:00<00:00, 131.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     690/1227 [===============>..............] - ETA: 4:28

    100%|██████████| 64/64 [00:00<00:00, 129.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     691/1227 [===============>..............] - ETA: 4:27

    100%|██████████| 64/64 [00:00<00:00, 130.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     692/1227 [===============>..............] - ETA: 4:27

    100%|██████████| 64/64 [00:00<00:00, 129.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     693/1227 [===============>..............] - ETA: 4:26

    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     694/1227 [===============>..............] - ETA: 4:26

    100%|██████████| 64/64 [00:00<00:00, 127.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     695/1227 [===============>..............] - ETA: 4:25

    100%|██████████| 64/64 [00:00<00:00, 132.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     696/1227 [================>.............] - ETA: 4:25

    100%|██████████| 64/64 [00:00<00:00, 129.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     697/1227 [================>.............] - ETA: 4:24

    100%|██████████| 64/64 [00:00<00:00, 130.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     698/1227 [================>.............] - ETA: 4:24

    100%|██████████| 64/64 [00:00<00:00, 124.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     699/1227 [================>.............] - ETA: 4:23

    100%|██████████| 64/64 [00:00<00:00, 129.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     700/1227 [================>.............] - ETA: 4:23

    100%|██████████| 64/64 [00:00<00:00, 136.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     701/1227 [================>.............] - ETA: 4:22

    100%|██████████| 64/64 [00:00<00:00, 139.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     702/1227 [================>.............] - ETA: 4:22

    100%|██████████| 64/64 [00:00<00:00, 134.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     703/1227 [================>.............] - ETA: 4:21

    100%|██████████| 64/64 [00:00<00:00, 139.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     704/1227 [================>.............] - ETA: 4:21

    100%|██████████| 64/64 [00:00<00:00, 135.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     705/1227 [================>.............] - ETA: 4:20

    100%|██████████| 64/64 [00:00<00:00, 126.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     706/1227 [================>.............] - ETA: 4:20

    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     707/1227 [================>.............] - ETA: 4:19

    100%|██████████| 64/64 [00:00<00:00, 131.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     708/1227 [================>.............] - ETA: 4:19

    100%|██████████| 64/64 [00:00<00:00, 136.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     709/1227 [================>.............] - ETA: 4:18

    100%|██████████| 64/64 [00:00<00:00, 131.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     710/1227 [================>.............] - ETA: 4:18

    100%|██████████| 64/64 [00:00<00:00, 136.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     711/1227 [================>.............] - ETA: 4:17

    100%|██████████| 64/64 [00:00<00:00, 126.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     712/1227 [================>.............] - ETA: 4:17

    100%|██████████| 64/64 [00:00<00:00, 136.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     713/1227 [================>.............] - ETA: 4:16

    100%|██████████| 64/64 [00:00<00:00, 121.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     714/1227 [================>.............] - ETA: 4:16

    100%|██████████| 64/64 [00:00<00:00, 135.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     715/1227 [================>.............] - ETA: 4:15

    100%|██████████| 64/64 [00:00<00:00, 134.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     716/1227 [================>.............] - ETA: 4:15

    100%|██████████| 64/64 [00:00<00:00, 136.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     717/1227 [================>.............] - ETA: 4:14

    100%|██████████| 64/64 [00:00<00:00, 126.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     718/1227 [================>.............] - ETA: 4:14

    100%|██████████| 64/64 [00:00<00:00, 131.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     719/1227 [================>.............] - ETA: 4:13

    100%|██████████| 64/64 [00:00<00:00, 129.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     720/1227 [================>.............] - ETA: 4:13

    100%|██████████| 64/64 [00:00<00:00, 133.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     721/1227 [================>.............] - ETA: 4:12

    100%|██████████| 64/64 [00:00<00:00, 129.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     722/1227 [================>.............] - ETA: 4:12

    100%|██████████| 64/64 [00:00<00:00, 138.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     723/1227 [================>.............] - ETA: 4:11

    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     724/1227 [================>.............] - ETA: 4:11

    100%|██████████| 64/64 [00:00<00:00, 130.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     725/1227 [================>.............] - ETA: 4:10

    100%|██████████| 64/64 [00:00<00:00, 137.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     726/1227 [================>.............] - ETA: 4:10

    100%|██████████| 64/64 [00:00<00:00, 132.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     727/1227 [================>.............] - ETA: 4:09

    100%|██████████| 64/64 [00:00<00:00, 134.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     728/1227 [================>.............] - ETA: 4:09

    100%|██████████| 64/64 [00:00<00:00, 133.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     729/1227 [================>.............] - ETA: 4:08

    100%|██████████| 64/64 [00:00<00:00, 138.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     730/1227 [================>.............] - ETA: 4:08

    100%|██████████| 64/64 [00:00<00:00, 128.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     731/1227 [================>.............] - ETA: 4:07

    100%|██████████| 64/64 [00:00<00:00, 137.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     732/1227 [================>.............] - ETA: 4:07

    100%|██████████| 64/64 [00:00<00:00, 138.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     733/1227 [================>.............] - ETA: 4:06

    100%|██████████| 64/64 [00:00<00:00, 131.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     734/1227 [================>.............] - ETA: 4:06

    100%|██████████| 64/64 [00:00<00:00, 127.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     735/1227 [================>.............] - ETA: 4:05

    100%|██████████| 64/64 [00:00<00:00, 131.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     736/1227 [================>.............] - ETA: 4:05

    100%|██████████| 64/64 [00:00<00:00, 131.78it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     737/1227 [=================>............] - ETA: 4:04

    100%|██████████| 64/64 [00:00<00:00, 132.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     738/1227 [=================>............] - ETA: 4:04

    100%|██████████| 64/64 [00:00<00:00, 133.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     739/1227 [=================>............] - ETA: 4:03

    100%|██████████| 64/64 [00:00<00:00, 132.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     740/1227 [=================>............] - ETA: 4:03

    100%|██████████| 64/64 [00:00<00:00, 125.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     741/1227 [=================>............] - ETA: 4:02

    100%|██████████| 64/64 [00:00<00:00, 123.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     742/1227 [=================>............] - ETA: 4:02

    100%|██████████| 64/64 [00:00<00:00, 120.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     743/1227 [=================>............] - ETA: 4:01

    100%|██████████| 64/64 [00:00<00:00, 126.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     744/1227 [=================>............] - ETA: 4:01

    100%|██████████| 64/64 [00:00<00:00, 121.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     745/1227 [=================>............] - ETA: 4:00

    100%|██████████| 64/64 [00:00<00:00, 124.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     746/1227 [=================>............] - ETA: 4:00

    100%|██████████| 64/64 [00:00<00:00, 122.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     747/1227 [=================>............] - ETA: 3:59

    100%|██████████| 64/64 [00:00<00:00, 119.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     748/1227 [=================>............] - ETA: 3:59

    100%|██████████| 64/64 [00:00<00:00, 119.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     749/1227 [=================>............] - ETA: 3:58

    100%|██████████| 64/64 [00:00<00:00, 127.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     750/1227 [=================>............] - ETA: 3:58

    100%|██████████| 64/64 [00:00<00:00, 126.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     751/1227 [=================>............] - ETA: 3:57

    100%|██████████| 64/64 [00:00<00:00, 123.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     752/1227 [=================>............] - ETA: 3:57

    100%|██████████| 64/64 [00:00<00:00, 124.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     753/1227 [=================>............] - ETA: 3:56

    100%|██████████| 64/64 [00:00<00:00, 116.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     754/1227 [=================>............] - ETA: 3:56

    100%|██████████| 64/64 [00:00<00:00, 120.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     755/1227 [=================>............] - ETA: 3:56

    100%|██████████| 64/64 [00:00<00:00, 122.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     756/1227 [=================>............] - ETA: 3:55

    100%|██████████| 64/64 [00:00<00:00, 119.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     757/1227 [=================>............] - ETA: 3:55

    100%|██████████| 64/64 [00:00<00:00, 126.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     758/1227 [=================>............] - ETA: 3:54

    100%|██████████| 64/64 [00:00<00:00, 122.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     759/1227 [=================>............] - ETA: 3:54

    100%|██████████| 64/64 [00:00<00:00, 133.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     760/1227 [=================>............] - ETA: 3:53

    100%|██████████| 64/64 [00:00<00:00, 133.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     761/1227 [=================>............] - ETA: 3:53

    100%|██████████| 64/64 [00:00<00:00, 136.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     762/1227 [=================>............] - ETA: 3:52

    100%|██████████| 64/64 [00:00<00:00, 134.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     763/1227 [=================>............] - ETA: 3:52

    100%|██████████| 64/64 [00:00<00:00, 132.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     764/1227 [=================>............] - ETA: 3:51

    100%|██████████| 64/64 [00:00<00:00, 126.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     765/1227 [=================>............] - ETA: 3:51

    100%|██████████| 64/64 [00:00<00:00, 130.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     766/1227 [=================>............] - ETA: 3:50

    100%|██████████| 64/64 [00:00<00:00, 133.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     767/1227 [=================>............] - ETA: 3:50

    100%|██████████| 64/64 [00:00<00:00, 133.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     768/1227 [=================>............] - ETA: 3:49

    100%|██████████| 64/64 [00:00<00:00, 134.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     769/1227 [=================>............] - ETA: 3:49

    100%|██████████| 64/64 [00:00<00:00, 130.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     770/1227 [=================>............] - ETA: 3:48

    100%|██████████| 64/64 [00:00<00:00, 133.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     771/1227 [=================>............] - ETA: 3:48

    100%|██████████| 64/64 [00:00<00:00, 130.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     772/1227 [=================>............] - ETA: 3:47

    100%|██████████| 64/64 [00:00<00:00, 126.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     773/1227 [=================>............] - ETA: 3:47

    100%|██████████| 64/64 [00:00<00:00, 138.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     774/1227 [=================>............] - ETA: 3:46

    100%|██████████| 64/64 [00:00<00:00, 142.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     775/1227 [=================>............] - ETA: 3:46

    100%|██████████| 64/64 [00:00<00:00, 133.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     776/1227 [=================>............] - ETA: 3:45

    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     777/1227 [=================>............] - ETA: 3:44

    100%|██████████| 64/64 [00:00<00:00, 125.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     778/1227 [==================>...........] - ETA: 3:44

    100%|██████████| 64/64 [00:00<00:00, 137.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     779/1227 [==================>...........] - ETA: 3:43

    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     780/1227 [==================>...........] - ETA: 3:43

    100%|██████████| 64/64 [00:00<00:00, 134.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     781/1227 [==================>...........] - ETA: 3:42

    100%|██████████| 64/64 [00:00<00:00, 132.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     782/1227 [==================>...........] - ETA: 3:42

    100%|██████████| 64/64 [00:00<00:00, 137.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     783/1227 [==================>...........] - ETA: 3:41

    100%|██████████| 64/64 [00:00<00:00, 134.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     784/1227 [==================>...........] - ETA: 3:41

    100%|██████████| 64/64 [00:00<00:00, 131.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     785/1227 [==================>...........] - ETA: 3:40

    100%|██████████| 64/64 [00:00<00:00, 124.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     786/1227 [==================>...........] - ETA: 3:40

    100%|██████████| 64/64 [00:00<00:00, 127.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     787/1227 [==================>...........] - ETA: 3:39

    100%|██████████| 64/64 [00:00<00:00, 125.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     788/1227 [==================>...........] - ETA: 3:39

    100%|██████████| 64/64 [00:00<00:00, 125.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     789/1227 [==================>...........] - ETA: 3:38

    100%|██████████| 64/64 [00:00<00:00, 129.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     790/1227 [==================>...........] - ETA: 3:38

    100%|██████████| 64/64 [00:00<00:00, 129.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     791/1227 [==================>...........] - ETA: 3:38

    100%|██████████| 64/64 [00:00<00:00, 130.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     792/1227 [==================>...........] - ETA: 3:37

    100%|██████████| 64/64 [00:00<00:00, 126.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     793/1227 [==================>...........] - ETA: 3:37

    100%|██████████| 64/64 [00:00<00:00, 134.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     794/1227 [==================>...........] - ETA: 3:36

    100%|██████████| 64/64 [00:00<00:00, 128.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     795/1227 [==================>...........] - ETA: 3:36

    100%|██████████| 64/64 [00:00<00:00, 135.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     796/1227 [==================>...........] - ETA: 3:35

    100%|██████████| 64/64 [00:00<00:00, 130.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     797/1227 [==================>...........] - ETA: 3:34

    100%|██████████| 64/64 [00:00<00:00, 134.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     798/1227 [==================>...........] - ETA: 3:34

    100%|██████████| 64/64 [00:00<00:00, 135.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     799/1227 [==================>...........] - ETA: 3:33

    100%|██████████| 64/64 [00:00<00:00, 138.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     800/1227 [==================>...........] - ETA: 3:33

    100%|██████████| 64/64 [00:00<00:00, 136.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     801/1227 [==================>...........] - ETA: 3:32

    100%|██████████| 64/64 [00:00<00:00, 133.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     802/1227 [==================>...........] - ETA: 3:32

    100%|██████████| 64/64 [00:00<00:00, 132.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     803/1227 [==================>...........] - ETA: 3:31

    100%|██████████| 64/64 [00:00<00:00, 129.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     804/1227 [==================>...........] - ETA: 3:31

    100%|██████████| 64/64 [00:00<00:00, 126.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     805/1227 [==================>...........] - ETA: 3:30

    100%|██████████| 64/64 [00:00<00:00, 132.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     806/1227 [==================>...........] - ETA: 3:30

    100%|██████████| 64/64 [00:00<00:00, 129.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     807/1227 [==================>...........] - ETA: 3:29

    100%|██████████| 64/64 [00:00<00:00, 129.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     808/1227 [==================>...........] - ETA: 3:29

    100%|██████████| 64/64 [00:00<00:00, 128.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     809/1227 [==================>...........] - ETA: 3:28

    100%|██████████| 64/64 [00:00<00:00, 132.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     810/1227 [==================>...........] - ETA: 3:28

    100%|██████████| 64/64 [00:00<00:00, 132.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     811/1227 [==================>...........] - ETA: 3:27

    100%|██████████| 64/64 [00:00<00:00, 130.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     812/1227 [==================>...........] - ETA: 3:27

    100%|██████████| 64/64 [00:00<00:00, 130.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     813/1227 [==================>...........] - ETA: 3:26

    100%|██████████| 64/64 [00:00<00:00, 127.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     814/1227 [==================>...........] - ETA: 3:26

    100%|██████████| 64/64 [00:00<00:00, 131.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     815/1227 [==================>...........] - ETA: 3:25

    100%|██████████| 64/64 [00:00<00:00, 132.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     816/1227 [==================>...........] - ETA: 3:25

    100%|██████████| 64/64 [00:00<00:00, 132.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     817/1227 [==================>...........] - ETA: 3:24

    100%|██████████| 64/64 [00:00<00:00, 126.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     818/1227 [==================>...........] - ETA: 3:24

    100%|██████████| 64/64 [00:00<00:00, 133.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     819/1227 [===================>..........] - ETA: 3:23

    100%|██████████| 64/64 [00:00<00:00, 129.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     820/1227 [===================>..........] - ETA: 3:23

    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     821/1227 [===================>..........] - ETA: 3:22

    100%|██████████| 64/64 [00:00<00:00, 126.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     822/1227 [===================>..........] - ETA: 3:22

    100%|██████████| 64/64 [00:00<00:00, 130.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     823/1227 [===================>..........] - ETA: 3:21

    100%|██████████| 64/64 [00:00<00:00, 127.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     824/1227 [===================>..........] - ETA: 3:21

    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     825/1227 [===================>..........] - ETA: 3:20

    100%|██████████| 64/64 [00:00<00:00, 130.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     826/1227 [===================>..........] - ETA: 3:20

    100%|██████████| 64/64 [00:00<00:00, 140.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     827/1227 [===================>..........] - ETA: 3:19

    100%|██████████| 64/64 [00:00<00:00, 131.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     828/1227 [===================>..........] - ETA: 3:19

    100%|██████████| 64/64 [00:00<00:00, 131.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     829/1227 [===================>..........] - ETA: 3:18

    100%|██████████| 64/64 [00:00<00:00, 130.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     830/1227 [===================>..........] - ETA: 3:18

    100%|██████████| 64/64 [00:00<00:00, 135.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     831/1227 [===================>..........] - ETA: 3:17

    100%|██████████| 64/64 [00:00<00:00, 130.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     832/1227 [===================>..........] - ETA: 3:17

    100%|██████████| 64/64 [00:00<00:00, 132.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     833/1227 [===================>..........] - ETA: 3:16

    100%|██████████| 64/64 [00:00<00:00, 135.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     834/1227 [===================>..........] - ETA: 3:16

    100%|██████████| 64/64 [00:00<00:00, 133.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     835/1227 [===================>..........] - ETA: 3:15

    100%|██████████| 64/64 [00:00<00:00, 133.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     836/1227 [===================>..........] - ETA: 3:15

    100%|██████████| 64/64 [00:00<00:00, 131.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     837/1227 [===================>..........] - ETA: 3:14

    100%|██████████| 64/64 [00:00<00:00, 131.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     838/1227 [===================>..........] - ETA: 3:14

    100%|██████████| 64/64 [00:00<00:00, 129.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     839/1227 [===================>..........] - ETA: 3:13

    100%|██████████| 64/64 [00:00<00:00, 130.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     840/1227 [===================>..........] - ETA: 3:13

    100%|██████████| 64/64 [00:00<00:00, 130.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     841/1227 [===================>..........] - ETA: 3:12

    100%|██████████| 64/64 [00:00<00:00, 133.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     842/1227 [===================>..........] - ETA: 3:12

    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     843/1227 [===================>..........] - ETA: 3:11

    100%|██████████| 64/64 [00:00<00:00, 132.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     844/1227 [===================>..........] - ETA: 3:11

    100%|██████████| 64/64 [00:00<00:00, 128.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     845/1227 [===================>..........] - ETA: 3:10

    100%|██████████| 64/64 [00:00<00:00, 132.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     846/1227 [===================>..........] - ETA: 3:10

    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     847/1227 [===================>..........] - ETA: 3:09

    100%|██████████| 64/64 [00:00<00:00, 129.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     848/1227 [===================>..........] - ETA: 3:09

    100%|██████████| 64/64 [00:00<00:00, 129.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     849/1227 [===================>..........] - ETA: 3:08

    100%|██████████| 64/64 [00:00<00:00, 132.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     850/1227 [===================>..........] - ETA: 3:08

    100%|██████████| 64/64 [00:00<00:00, 136.00it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     851/1227 [===================>..........] - ETA: 3:07

    100%|██████████| 64/64 [00:00<00:00, 134.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     852/1227 [===================>..........] - ETA: 3:07

    100%|██████████| 64/64 [00:00<00:00, 124.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     853/1227 [===================>..........] - ETA: 3:06

    100%|██████████| 64/64 [00:00<00:00, 128.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     854/1227 [===================>..........] - ETA: 3:06

    100%|██████████| 64/64 [00:00<00:00, 127.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     855/1227 [===================>..........] - ETA: 3:05

    100%|██████████| 64/64 [00:00<00:00, 125.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     856/1227 [===================>..........] - ETA: 3:05

    100%|██████████| 64/64 [00:00<00:00, 126.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     857/1227 [===================>..........] - ETA: 3:04

    100%|██████████| 64/64 [00:00<00:00, 125.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     858/1227 [===================>..........] - ETA: 3:04

    100%|██████████| 64/64 [00:00<00:00, 125.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     859/1227 [===================>..........] - ETA: 3:03

    100%|██████████| 64/64 [00:00<00:00, 128.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     860/1227 [====================>.........] - ETA: 3:03

    100%|██████████| 64/64 [00:00<00:00, 124.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     861/1227 [====================>.........] - ETA: 3:02

    100%|██████████| 64/64 [00:00<00:00, 121.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     862/1227 [====================>.........] - ETA: 3:02

    100%|██████████| 64/64 [00:00<00:00, 126.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     863/1227 [====================>.........] - ETA: 3:01

    100%|██████████| 64/64 [00:00<00:00, 122.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     864/1227 [====================>.........] - ETA: 3:01

    100%|██████████| 64/64 [00:00<00:00, 124.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     865/1227 [====================>.........] - ETA: 3:00

    100%|██████████| 64/64 [00:00<00:00, 128.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     866/1227 [====================>.........] - ETA: 3:00

    100%|██████████| 64/64 [00:00<00:00, 128.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     867/1227 [====================>.........] - ETA: 2:59

    100%|██████████| 64/64 [00:00<00:00, 127.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     868/1227 [====================>.........] - ETA: 2:59

    100%|██████████| 64/64 [00:00<00:00, 122.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     869/1227 [====================>.........] - ETA: 2:59

    100%|██████████| 64/64 [00:00<00:00, 127.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     870/1227 [====================>.........] - ETA: 2:58

    100%|██████████| 64/64 [00:00<00:00, 124.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     871/1227 [====================>.........] - ETA: 2:58

    100%|██████████| 64/64 [00:00<00:00, 128.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     872/1227 [====================>.........] - ETA: 2:57

    100%|██████████| 64/64 [00:00<00:00, 133.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     873/1227 [====================>.........] - ETA: 2:57

    100%|██████████| 64/64 [00:00<00:00, 128.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     874/1227 [====================>.........] - ETA: 2:56

    100%|██████████| 64/64 [00:00<00:00, 133.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     875/1227 [====================>.........] - ETA: 2:56

    100%|██████████| 64/64 [00:00<00:00, 138.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     876/1227 [====================>.........] - ETA: 2:55

    100%|██████████| 64/64 [00:00<00:00, 142.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     877/1227 [====================>.........] - ETA: 2:55

    100%|██████████| 64/64 [00:00<00:00, 133.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     878/1227 [====================>.........] - ETA: 2:54

    100%|██████████| 64/64 [00:00<00:00, 141.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     879/1227 [====================>.........] - ETA: 2:53

    100%|██████████| 64/64 [00:00<00:00, 130.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     880/1227 [====================>.........] - ETA: 2:53

    100%|██████████| 64/64 [00:00<00:00, 140.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     881/1227 [====================>.........] - ETA: 2:52

    100%|██████████| 64/64 [00:00<00:00, 136.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     882/1227 [====================>.........] - ETA: 2:52

    100%|██████████| 64/64 [00:00<00:00, 136.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     883/1227 [====================>.........] - ETA: 2:51

    100%|██████████| 64/64 [00:00<00:00, 137.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     884/1227 [====================>.........] - ETA: 2:51

    100%|██████████| 64/64 [00:00<00:00, 136.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     885/1227 [====================>.........] - ETA: 2:50

    100%|██████████| 64/64 [00:00<00:00, 137.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     886/1227 [====================>.........] - ETA: 2:50

    100%|██████████| 64/64 [00:00<00:00, 149.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     887/1227 [====================>.........] - ETA: 2:49

    100%|██████████| 64/64 [00:00<00:00, 137.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     888/1227 [====================>.........] - ETA: 2:49

    100%|██████████| 64/64 [00:00<00:00, 133.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     889/1227 [====================>.........] - ETA: 2:48

    100%|██████████| 64/64 [00:00<00:00, 142.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     890/1227 [====================>.........] - ETA: 2:48

    100%|██████████| 64/64 [00:00<00:00, 135.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     891/1227 [====================>.........] - ETA: 2:47

    100%|██████████| 64/64 [00:00<00:00, 137.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     892/1227 [====================>.........] - ETA: 2:47

    100%|██████████| 64/64 [00:00<00:00, 133.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     893/1227 [====================>.........] - ETA: 2:46

    100%|██████████| 64/64 [00:00<00:00, 144.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     894/1227 [====================>.........] - ETA: 2:46

    100%|██████████| 64/64 [00:00<00:00, 134.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     895/1227 [====================>.........] - ETA: 2:45

    100%|██████████| 64/64 [00:00<00:00, 139.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     896/1227 [====================>.........] - ETA: 2:45

    100%|██████████| 64/64 [00:00<00:00, 136.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     897/1227 [====================>.........] - ETA: 2:44

    100%|██████████| 64/64 [00:00<00:00, 135.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     898/1227 [====================>.........] - ETA: 2:44

    100%|██████████| 64/64 [00:00<00:00, 136.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     899/1227 [====================>.........] - ETA: 2:43

    100%|██████████| 64/64 [00:00<00:00, 140.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     900/1227 [=====================>........] - ETA: 2:43

    100%|██████████| 64/64 [00:00<00:00, 139.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     901/1227 [=====================>........] - ETA: 2:42

    100%|██████████| 64/64 [00:00<00:00, 138.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     902/1227 [=====================>........] - ETA: 2:42

    100%|██████████| 64/64 [00:00<00:00, 144.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     903/1227 [=====================>........] - ETA: 2:41

    100%|██████████| 64/64 [00:00<00:00, 136.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     904/1227 [=====================>........] - ETA: 2:41

    100%|██████████| 64/64 [00:00<00:00, 139.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     905/1227 [=====================>........] - ETA: 2:40

    100%|██████████| 64/64 [00:00<00:00, 134.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     906/1227 [=====================>........] - ETA: 2:40

    100%|██████████| 64/64 [00:00<00:00, 140.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     907/1227 [=====================>........] - ETA: 2:39

    100%|██████████| 64/64 [00:00<00:00, 136.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     908/1227 [=====================>........] - ETA: 2:39

    100%|██████████| 64/64 [00:00<00:00, 135.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     909/1227 [=====================>........] - ETA: 2:38

    100%|██████████| 64/64 [00:00<00:00, 138.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     910/1227 [=====================>........] - ETA: 2:38

    100%|██████████| 64/64 [00:00<00:00, 138.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     911/1227 [=====================>........] - ETA: 2:37

    100%|██████████| 64/64 [00:00<00:00, 139.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     912/1227 [=====================>........] - ETA: 2:37

    100%|██████████| 64/64 [00:00<00:00, 133.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     913/1227 [=====================>........] - ETA: 2:36

    100%|██████████| 64/64 [00:00<00:00, 144.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     914/1227 [=====================>........] - ETA: 2:36

    100%|██████████| 64/64 [00:00<00:00, 138.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     915/1227 [=====================>........] - ETA: 2:35

    100%|██████████| 64/64 [00:00<00:00, 145.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     916/1227 [=====================>........] - ETA: 2:35

    100%|██████████| 64/64 [00:00<00:00, 137.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     917/1227 [=====================>........] - ETA: 2:34

    100%|██████████| 64/64 [00:00<00:00, 142.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     918/1227 [=====================>........] - ETA: 2:34

    100%|██████████| 64/64 [00:00<00:00, 135.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     919/1227 [=====================>........] - ETA: 2:33

    100%|██████████| 64/64 [00:00<00:00, 133.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     920/1227 [=====================>........] - ETA: 2:33

    100%|██████████| 64/64 [00:00<00:00, 138.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     921/1227 [=====================>........] - ETA: 2:32

    100%|██████████| 64/64 [00:00<00:00, 139.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     922/1227 [=====================>........] - ETA: 2:32

    100%|██████████| 64/64 [00:00<00:00, 143.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     923/1227 [=====================>........] - ETA: 2:31

    100%|██████████| 64/64 [00:00<00:00, 136.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     924/1227 [=====================>........] - ETA: 2:31

    100%|██████████| 64/64 [00:00<00:00, 137.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     925/1227 [=====================>........] - ETA: 2:30

    100%|██████████| 64/64 [00:00<00:00, 133.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     926/1227 [=====================>........] - ETA: 2:30

    100%|██████████| 64/64 [00:00<00:00, 133.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     927/1227 [=====================>........] - ETA: 2:29

    100%|██████████| 64/64 [00:00<00:00, 132.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     928/1227 [=====================>........] - ETA: 2:29

    100%|██████████| 64/64 [00:00<00:00, 137.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     929/1227 [=====================>........] - ETA: 2:28

    100%|██████████| 64/64 [00:00<00:00, 145.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     930/1227 [=====================>........] - ETA: 2:28

    100%|██████████| 64/64 [00:00<00:00, 132.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     931/1227 [=====================>........] - ETA: 2:27

    100%|██████████| 64/64 [00:00<00:00, 147.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     932/1227 [=====================>........] - ETA: 2:27

    100%|██████████| 64/64 [00:00<00:00, 143.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     933/1227 [=====================>........] - ETA: 2:26

    100%|██████████| 64/64 [00:00<00:00, 141.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     934/1227 [=====================>........] - ETA: 2:26

    100%|██████████| 64/64 [00:00<00:00, 137.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     935/1227 [=====================>........] - ETA: 2:25

    100%|██████████| 64/64 [00:00<00:00, 133.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     936/1227 [=====================>........] - ETA: 2:25

    100%|██████████| 64/64 [00:00<00:00, 129.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     937/1227 [=====================>........] - ETA: 2:24

    100%|██████████| 64/64 [00:00<00:00, 136.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     938/1227 [=====================>........] - ETA: 2:24

    100%|██████████| 64/64 [00:00<00:00, 134.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     939/1227 [=====================>........] - ETA: 2:23

    100%|██████████| 64/64 [00:00<00:00, 129.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     940/1227 [=====================>........] - ETA: 2:23

    100%|██████████| 64/64 [00:00<00:00, 135.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     941/1227 [======================>.......] - ETA: 2:22

    100%|██████████| 64/64 [00:00<00:00, 137.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     942/1227 [======================>.......] - ETA: 2:21

    100%|██████████| 64/64 [00:00<00:00, 135.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     943/1227 [======================>.......] - ETA: 2:21

    100%|██████████| 64/64 [00:00<00:00, 134.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     944/1227 [======================>.......] - ETA: 2:20

    100%|██████████| 64/64 [00:00<00:00, 144.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     945/1227 [======================>.......] - ETA: 2:20

    100%|██████████| 64/64 [00:00<00:00, 141.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     946/1227 [======================>.......] - ETA: 2:19

    100%|██████████| 64/64 [00:00<00:00, 141.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     947/1227 [======================>.......] - ETA: 2:19

    100%|██████████| 64/64 [00:00<00:00, 120.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     948/1227 [======================>.......] - ETA: 2:18

    100%|██████████| 64/64 [00:00<00:00, 121.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     949/1227 [======================>.......] - ETA: 2:18

    100%|██████████| 64/64 [00:00<00:00, 128.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     950/1227 [======================>.......] - ETA: 2:18

    100%|██████████| 64/64 [00:00<00:00, 138.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     951/1227 [======================>.......] - ETA: 2:17

    100%|██████████| 64/64 [00:00<00:00, 138.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     952/1227 [======================>.......] - ETA: 2:16

    100%|██████████| 64/64 [00:00<00:00, 137.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     953/1227 [======================>.......] - ETA: 2:16

    100%|██████████| 64/64 [00:00<00:00, 143.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     954/1227 [======================>.......] - ETA: 2:15

    100%|██████████| 64/64 [00:00<00:00, 134.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     955/1227 [======================>.......] - ETA: 2:15

    100%|██████████| 64/64 [00:00<00:00, 143.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     956/1227 [======================>.......] - ETA: 2:14

    100%|██████████| 64/64 [00:00<00:00, 135.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     957/1227 [======================>.......] - ETA: 2:14

    100%|██████████| 64/64 [00:00<00:00, 138.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     958/1227 [======================>.......] - ETA: 2:13

    100%|██████████| 64/64 [00:00<00:00, 136.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     959/1227 [======================>.......] - ETA: 2:13

    100%|██████████| 64/64 [00:00<00:00, 134.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     960/1227 [======================>.......] - ETA: 2:12

    100%|██████████| 64/64 [00:00<00:00, 138.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     961/1227 [======================>.......] - ETA: 2:12

    100%|██████████| 64/64 [00:00<00:00, 136.72it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     962/1227 [======================>.......] - ETA: 2:11

    100%|██████████| 64/64 [00:00<00:00, 125.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     963/1227 [======================>.......] - ETA: 2:11

    100%|██████████| 64/64 [00:00<00:00, 135.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     964/1227 [======================>.......] - ETA: 2:10

    100%|██████████| 64/64 [00:00<00:00, 134.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     965/1227 [======================>.......] - ETA: 2:10

    100%|██████████| 64/64 [00:00<00:00, 139.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     966/1227 [======================>.......] - ETA: 2:09

    100%|██████████| 64/64 [00:00<00:00, 139.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     967/1227 [======================>.......] - ETA: 2:09

    100%|██████████| 64/64 [00:00<00:00, 134.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     968/1227 [======================>.......] - ETA: 2:08

    100%|██████████| 64/64 [00:00<00:00, 138.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     969/1227 [======================>.......] - ETA: 2:08

    100%|██████████| 64/64 [00:00<00:00, 139.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     970/1227 [======================>.......] - ETA: 2:07

    100%|██████████| 64/64 [00:00<00:00, 135.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     971/1227 [======================>.......] - ETA: 2:07

    100%|██████████| 64/64 [00:00<00:00, 137.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     972/1227 [======================>.......] - ETA: 2:06

    100%|██████████| 64/64 [00:00<00:00, 143.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     973/1227 [======================>.......] - ETA: 2:06

    100%|██████████| 64/64 [00:00<00:00, 124.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     974/1227 [======================>.......] - ETA: 2:05

    100%|██████████| 64/64 [00:00<00:00, 141.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     975/1227 [======================>.......] - ETA: 2:05

    100%|██████████| 64/64 [00:00<00:00, 132.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     976/1227 [======================>.......] - ETA: 2:04

    100%|██████████| 64/64 [00:00<00:00, 137.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     977/1227 [======================>.......] - ETA: 2:04

    100%|██████████| 64/64 [00:00<00:00, 142.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     978/1227 [======================>.......] - ETA: 2:03

    100%|██████████| 64/64 [00:00<00:00, 136.69it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     979/1227 [======================>.......] - ETA: 2:03

    100%|██████████| 64/64 [00:00<00:00, 140.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     980/1227 [======================>.......] - ETA: 2:02

    100%|██████████| 64/64 [00:00<00:00, 133.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     981/1227 [======================>.......] - ETA: 2:02

    100%|██████████| 64/64 [00:00<00:00, 134.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     982/1227 [=======================>......] - ETA: 2:01

    100%|██████████| 64/64 [00:00<00:00, 142.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     983/1227 [=======================>......] - ETA: 2:01

    100%|██████████| 64/64 [00:00<00:00, 138.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     984/1227 [=======================>......] - ETA: 2:00

    100%|██████████| 64/64 [00:00<00:00, 139.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     985/1227 [=======================>......] - ETA: 2:00

    100%|██████████| 64/64 [00:00<00:00, 141.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     986/1227 [=======================>......] - ETA: 1:59

    100%|██████████| 64/64 [00:00<00:00, 134.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     987/1227 [=======================>......] - ETA: 1:59

    100%|██████████| 64/64 [00:00<00:00, 131.41it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     988/1227 [=======================>......] - ETA: 1:58

    100%|██████████| 64/64 [00:00<00:00, 131.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     989/1227 [=======================>......] - ETA: 1:58

    100%|██████████| 64/64 [00:00<00:00, 144.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     990/1227 [=======================>......] - ETA: 1:57

    100%|██████████| 64/64 [00:00<00:00, 140.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     991/1227 [=======================>......] - ETA: 1:57

    100%|██████████| 64/64 [00:00<00:00, 138.11it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     992/1227 [=======================>......] - ETA: 1:56

    100%|██████████| 64/64 [00:00<00:00, 134.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     993/1227 [=======================>......] - ETA: 1:56

    100%|██████████| 64/64 [00:00<00:00, 135.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     994/1227 [=======================>......] - ETA: 1:55

    100%|██████████| 64/64 [00:00<00:00, 138.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     995/1227 [=======================>......] - ETA: 1:55

    100%|██████████| 64/64 [00:00<00:00, 135.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     996/1227 [=======================>......] - ETA: 1:54

    100%|██████████| 64/64 [00:00<00:00, 131.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     997/1227 [=======================>......] - ETA: 1:54

    100%|██████████| 64/64 [00:00<00:00, 138.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     998/1227 [=======================>......] - ETA: 1:53

    100%|██████████| 64/64 [00:00<00:00, 144.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

     999/1227 [=======================>......] - ETA: 1:53

    100%|██████████| 64/64 [00:00<00:00, 134.76it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1000/1227 [=======================>......] - ETA: 1:52

    100%|██████████| 64/64 [00:00<00:00, 133.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1001/1227 [=======================>......] - ETA: 1:52

    100%|██████████| 64/64 [00:00<00:00, 130.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1002/1227 [=======================>......] - ETA: 1:51

    100%|██████████| 64/64 [00:00<00:00, 134.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1003/1227 [=======================>......] - ETA: 1:51

    100%|██████████| 64/64 [00:00<00:00, 141.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1004/1227 [=======================>......] - ETA: 1:50

    100%|██████████| 64/64 [00:00<00:00, 139.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1005/1227 [=======================>......] - ETA: 1:50

    100%|██████████| 64/64 [00:00<00:00, 137.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1006/1227 [=======================>......] - ETA: 1:49

    100%|██████████| 64/64 [00:00<00:00, 134.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1007/1227 [=======================>......] - ETA: 1:49

    100%|██████████| 64/64 [00:00<00:00, 139.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1008/1227 [=======================>......] - ETA: 1:48

    100%|██████████| 64/64 [00:00<00:00, 137.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1009/1227 [=======================>......] - ETA: 1:48

    100%|██████████| 64/64 [00:00<00:00, 136.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1010/1227 [=======================>......] - ETA: 1:47

    100%|██████████| 64/64 [00:00<00:00, 135.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1011/1227 [=======================>......] - ETA: 1:47

    100%|██████████| 64/64 [00:00<00:00, 135.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1012/1227 [=======================>......] - ETA: 1:46

    100%|██████████| 64/64 [00:00<00:00, 134.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1013/1227 [=======================>......] - ETA: 1:46

    100%|██████████| 64/64 [00:00<00:00, 137.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1014/1227 [=======================>......] - ETA: 1:45

    100%|██████████| 64/64 [00:00<00:00, 142.94it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1015/1227 [=======================>......] - ETA: 1:45

    100%|██████████| 64/64 [00:00<00:00, 139.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1016/1227 [=======================>......] - ETA: 1:44

    100%|██████████| 64/64 [00:00<00:00, 147.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1017/1227 [=======================>......] - ETA: 1:44

    100%|██████████| 64/64 [00:00<00:00, 131.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1018/1227 [=======================>......] - ETA: 1:43

    100%|██████████| 64/64 [00:00<00:00, 138.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1019/1227 [=======================>......] - ETA: 1:43

    100%|██████████| 64/64 [00:00<00:00, 138.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1020/1227 [=======================>......] - ETA: 1:42

    100%|██████████| 64/64 [00:00<00:00, 138.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1021/1227 [=======================>......] - ETA: 1:42

    100%|██████████| 64/64 [00:00<00:00, 138.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1022/1227 [=======================>......] - ETA: 1:41

    100%|██████████| 64/64 [00:00<00:00, 141.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1023/1227 [========================>.....] - ETA: 1:41

    100%|██████████| 64/64 [00:00<00:00, 133.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1024/1227 [========================>.....] - ETA: 1:40

    100%|██████████| 64/64 [00:00<00:00, 138.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1025/1227 [========================>.....] - ETA: 1:40

    100%|██████████| 64/64 [00:00<00:00, 136.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1026/1227 [========================>.....] - ETA: 1:39

    100%|██████████| 64/64 [00:00<00:00, 134.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1027/1227 [========================>.....] - ETA: 1:39

    100%|██████████| 64/64 [00:00<00:00, 137.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1028/1227 [========================>.....] - ETA: 1:38

    100%|██████████| 64/64 [00:00<00:00, 136.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1029/1227 [========================>.....] - ETA: 1:38

    100%|██████████| 64/64 [00:00<00:00, 142.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1030/1227 [========================>.....] - ETA: 1:37

    100%|██████████| 64/64 [00:00<00:00, 131.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1031/1227 [========================>.....] - ETA: 1:37

    100%|██████████| 64/64 [00:00<00:00, 137.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1032/1227 [========================>.....] - ETA: 1:36

    100%|██████████| 64/64 [00:00<00:00, 133.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1033/1227 [========================>.....] - ETA: 1:36

    100%|██████████| 64/64 [00:00<00:00, 141.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1034/1227 [========================>.....] - ETA: 1:35

    100%|██████████| 64/64 [00:00<00:00, 138.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1035/1227 [========================>.....] - ETA: 1:35

    100%|██████████| 64/64 [00:00<00:00, 144.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1036/1227 [========================>.....] - ETA: 1:34

    100%|██████████| 64/64 [00:00<00:00, 139.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1037/1227 [========================>.....] - ETA: 1:34

    100%|██████████| 64/64 [00:00<00:00, 135.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1038/1227 [========================>.....] - ETA: 1:33

    100%|██████████| 64/64 [00:00<00:00, 139.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1039/1227 [========================>.....] - ETA: 1:33

    100%|██████████| 64/64 [00:00<00:00, 127.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1040/1227 [========================>.....] - ETA: 1:32

    100%|██████████| 64/64 [00:00<00:00, 140.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1041/1227 [========================>.....] - ETA: 1:32

    100%|██████████| 64/64 [00:00<00:00, 141.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1042/1227 [========================>.....] - ETA: 1:31

    100%|██████████| 64/64 [00:00<00:00, 138.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1043/1227 [========================>.....] - ETA: 1:31

    100%|██████████| 64/64 [00:00<00:00, 129.21it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1044/1227 [========================>.....] - ETA: 1:30

    100%|██████████| 64/64 [00:00<00:00, 137.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1045/1227 [========================>.....] - ETA: 1:30

    100%|██████████| 64/64 [00:00<00:00, 140.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1046/1227 [========================>.....] - ETA: 1:29

    100%|██████████| 64/64 [00:00<00:00, 141.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1047/1227 [========================>.....] - ETA: 1:29

    100%|██████████| 64/64 [00:00<00:00, 143.57it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1048/1227 [========================>.....] - ETA: 1:28

    100%|██████████| 64/64 [00:00<00:00, 141.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1049/1227 [========================>.....] - ETA: 1:28

    100%|██████████| 64/64 [00:00<00:00, 138.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1050/1227 [========================>.....] - ETA: 1:27

    100%|██████████| 64/64 [00:00<00:00, 132.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1051/1227 [========================>.....] - ETA: 1:27

    100%|██████████| 64/64 [00:00<00:00, 137.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1052/1227 [========================>.....] - ETA: 1:26

    100%|██████████| 64/64 [00:00<00:00, 131.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1053/1227 [========================>.....] - ETA: 1:26

    100%|██████████| 64/64 [00:00<00:00, 133.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1054/1227 [========================>.....] - ETA: 1:25

    100%|██████████| 64/64 [00:00<00:00, 133.54it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1055/1227 [========================>.....] - ETA: 1:25

    100%|██████████| 64/64 [00:00<00:00, 137.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1056/1227 [========================>.....] - ETA: 1:24

    100%|██████████| 64/64 [00:00<00:00, 103.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1057/1227 [========================>.....] - ETA: 1:24

    100%|██████████| 64/64 [00:00<00:00, 163.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1058/1227 [========================>.....] - ETA: 1:23

    100%|██████████| 64/64 [00:00<00:00, 140.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1059/1227 [========================>.....] - ETA: 1:23

    100%|██████████| 64/64 [00:00<00:00, 136.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1060/1227 [========================>.....] - ETA: 1:22

    100%|██████████| 64/64 [00:00<00:00, 139.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1061/1227 [========================>.....] - ETA: 1:22

    100%|██████████| 64/64 [00:00<00:00, 136.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1062/1227 [========================>.....] - ETA: 1:21

    100%|██████████| 64/64 [00:00<00:00, 136.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1063/1227 [========================>.....] - ETA: 1:21

    100%|██████████| 64/64 [00:00<00:00, 139.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1064/1227 [=========================>....] - ETA: 1:20

    100%|██████████| 64/64 [00:00<00:00, 139.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1065/1227 [=========================>....] - ETA: 1:20

    100%|██████████| 64/64 [00:00<00:00, 142.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1066/1227 [=========================>....] - ETA: 1:19

    100%|██████████| 64/64 [00:00<00:00, 141.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1067/1227 [=========================>....] - ETA: 1:19

    100%|██████████| 64/64 [00:00<00:00, 144.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1068/1227 [=========================>....] - ETA: 1:18

    100%|██████████| 64/64 [00:00<00:00, 135.98it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1069/1227 [=========================>....] - ETA: 1:18

    100%|██████████| 64/64 [00:00<00:00, 133.79it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1070/1227 [=========================>....] - ETA: 1:17

    100%|██████████| 64/64 [00:00<00:00, 139.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1071/1227 [=========================>....] - ETA: 1:17

    100%|██████████| 64/64 [00:00<00:00, 143.38it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1072/1227 [=========================>....] - ETA: 1:16

    100%|██████████| 64/64 [00:00<00:00, 139.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1073/1227 [=========================>....] - ETA: 1:16

    100%|██████████| 64/64 [00:00<00:00, 134.25it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1074/1227 [=========================>....] - ETA: 1:15

    100%|██████████| 64/64 [00:00<00:00, 132.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1075/1227 [=========================>....] - ETA: 1:15

    100%|██████████| 64/64 [00:00<00:00, 132.09it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1076/1227 [=========================>....] - ETA: 1:14

    100%|██████████| 64/64 [00:00<00:00, 134.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1077/1227 [=========================>....] - ETA: 1:14

    100%|██████████| 64/64 [00:00<00:00, 140.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1078/1227 [=========================>....] - ETA: 1:13

    100%|██████████| 64/64 [00:00<00:00, 134.99it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1079/1227 [=========================>....] - ETA: 1:13

    100%|██████████| 64/64 [00:00<00:00, 142.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1080/1227 [=========================>....] - ETA: 1:12

    100%|██████████| 64/64 [00:00<00:00, 138.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1081/1227 [=========================>....] - ETA: 1:12

    100%|██████████| 64/64 [00:00<00:00, 134.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1082/1227 [=========================>....] - ETA: 1:11

    100%|██████████| 64/64 [00:00<00:00, 136.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1083/1227 [=========================>....] - ETA: 1:11

    100%|██████████| 64/64 [00:00<00:00, 141.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1084/1227 [=========================>....] - ETA: 1:10

    100%|██████████| 64/64 [00:00<00:00, 143.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1085/1227 [=========================>....] - ETA: 1:10

    100%|██████████| 64/64 [00:00<00:00, 136.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1086/1227 [=========================>....] - ETA: 1:09

    100%|██████████| 64/64 [00:00<00:00, 137.35it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1087/1227 [=========================>....] - ETA: 1:09

    100%|██████████| 64/64 [00:00<00:00, 131.31it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1088/1227 [=========================>....] - ETA: 1:08

    100%|██████████| 64/64 [00:00<00:00, 135.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1089/1227 [=========================>....] - ETA: 1:08

    100%|██████████| 64/64 [00:00<00:00, 129.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1090/1227 [=========================>....] - ETA: 1:07

    100%|██████████| 64/64 [00:00<00:00, 135.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1091/1227 [=========================>....] - ETA: 1:07

    100%|██████████| 64/64 [00:00<00:00, 135.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1092/1227 [=========================>....] - ETA: 1:06

    100%|██████████| 64/64 [00:00<00:00, 140.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1093/1227 [=========================>....] - ETA: 1:06

    100%|██████████| 64/64 [00:00<00:00, 132.46it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1094/1227 [=========================>....] - ETA: 1:05

    100%|██████████| 64/64 [00:00<00:00, 133.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1095/1227 [=========================>....] - ETA: 1:05

    100%|██████████| 64/64 [00:00<00:00, 133.08it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1096/1227 [=========================>....] - ETA: 1:04

    100%|██████████| 64/64 [00:00<00:00, 138.29it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1097/1227 [=========================>....] - ETA: 1:04

    100%|██████████| 64/64 [00:00<00:00, 137.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1098/1227 [=========================>....] - ETA: 1:03

    100%|██████████| 64/64 [00:00<00:00, 138.42it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1099/1227 [=========================>....] - ETA: 1:03

    100%|██████████| 64/64 [00:00<00:00, 136.15it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1100/1227 [=========================>....] - ETA: 1:02

    100%|██████████| 64/64 [00:00<00:00, 128.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1101/1227 [=========================>....] - ETA: 1:02

    100%|██████████| 64/64 [00:00<00:00, 135.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1102/1227 [=========================>....] - ETA: 1:01

    100%|██████████| 64/64 [00:00<00:00, 132.68it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1103/1227 [=========================>....] - ETA: 1:01

    100%|██████████| 64/64 [00:00<00:00, 135.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1104/1227 [=========================>....] - ETA: 1:00

    100%|██████████| 64/64 [00:00<00:00, 135.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1105/1227 [==========================>...] - ETA: 1:00

    100%|██████████| 64/64 [00:00<00:00, 132.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1106/1227 [==========================>...] - ETA: 59s 

    100%|██████████| 64/64 [00:00<00:00, 123.50it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1107/1227 [==========================>...] - ETA: 59s

    100%|██████████| 64/64 [00:00<00:00, 128.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1108/1227 [==========================>...] - ETA: 58s

    100%|██████████| 64/64 [00:00<00:00, 123.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1109/1227 [==========================>...] - ETA: 58s

    100%|██████████| 64/64 [00:00<00:00, 122.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1110/1227 [==========================>...] - ETA: 58s

    100%|██████████| 64/64 [00:00<00:00, 122.58it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1111/1227 [==========================>...] - ETA: 57s

    100%|██████████| 64/64 [00:00<00:00, 119.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1112/1227 [==========================>...] - ETA: 57s

    100%|██████████| 64/64 [00:00<00:00, 118.80it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1113/1227 [==========================>...] - ETA: 56s

    100%|██████████| 64/64 [00:00<00:00, 130.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1114/1227 [==========================>...] - ETA: 56s

    100%|██████████| 64/64 [00:00<00:00, 121.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1115/1227 [==========================>...] - ETA: 55s

    100%|██████████| 64/64 [00:00<00:00, 120.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1116/1227 [==========================>...] - ETA: 55s

    100%|██████████| 64/64 [00:00<00:00, 124.52it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1117/1227 [==========================>...] - ETA: 54s

    100%|██████████| 64/64 [00:00<00:00, 130.93it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1118/1227 [==========================>...] - ETA: 54s

    100%|██████████| 64/64 [00:00<00:00, 125.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1119/1227 [==========================>...] - ETA: 53s

    100%|██████████| 64/64 [00:00<00:00, 130.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1120/1227 [==========================>...] - ETA: 53s

    100%|██████████| 64/64 [00:00<00:00, 130.81it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1121/1227 [==========================>...] - ETA: 52s

    100%|██████████| 64/64 [00:00<00:00, 136.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1122/1227 [==========================>...] - ETA: 52s

    100%|██████████| 64/64 [00:00<00:00, 135.49it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1123/1227 [==========================>...] - ETA: 51s

    100%|██████████| 64/64 [00:00<00:00, 130.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1124/1227 [==========================>...] - ETA: 51s

    100%|██████████| 64/64 [00:00<00:00, 129.14it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1125/1227 [==========================>...] - ETA: 50s

    100%|██████████| 64/64 [00:00<00:00, 133.24it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1126/1227 [==========================>...] - ETA: 50s

    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1127/1227 [==========================>...] - ETA: 49s

    100%|██████████| 64/64 [00:00<00:00, 140.87it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1128/1227 [==========================>...] - ETA: 49s

    100%|██████████| 64/64 [00:00<00:00, 139.05it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1129/1227 [==========================>...] - ETA: 48s

    100%|██████████| 64/64 [00:00<00:00, 141.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1130/1227 [==========================>...] - ETA: 48s

    100%|██████████| 64/64 [00:00<00:00, 131.67it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1131/1227 [==========================>...] - ETA: 47s

    100%|██████████| 64/64 [00:00<00:00, 134.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1132/1227 [==========================>...] - ETA: 47s

    100%|██████████| 64/64 [00:00<00:00, 145.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1133/1227 [==========================>...] - ETA: 46s

    100%|██████████| 64/64 [00:00<00:00, 134.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1134/1227 [==========================>...] - ETA: 46s

    100%|██████████| 64/64 [00:00<00:00, 132.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1135/1227 [==========================>...] - ETA: 45s

    100%|██████████| 64/64 [00:00<00:00, 135.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1136/1227 [==========================>...] - ETA: 45s

    100%|██████████| 64/64 [00:00<00:00, 139.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1137/1227 [==========================>...] - ETA: 44s

    100%|██████████| 64/64 [00:00<00:00, 133.01it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1138/1227 [==========================>...] - ETA: 44s

    100%|██████████| 64/64 [00:00<00:00, 135.26it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1139/1227 [==========================>...] - ETA: 43s

    100%|██████████| 64/64 [00:00<00:00, 135.55it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1140/1227 [==========================>...] - ETA: 43s

    100%|██████████| 64/64 [00:00<00:00, 142.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1141/1227 [==========================>...] - ETA: 42s

    100%|██████████| 64/64 [00:00<00:00, 132.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1142/1227 [==========================>...] - ETA: 42s

    100%|██████████| 64/64 [00:00<00:00, 133.23it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1143/1227 [==========================>...] - ETA: 41s

    100%|██████████| 64/64 [00:00<00:00, 137.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1144/1227 [==========================>...] - ETA: 41s

    100%|██████████| 64/64 [00:00<00:00, 125.56it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1145/1227 [==========================>...] - ETA: 40s

    100%|██████████| 64/64 [00:00<00:00, 147.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1146/1227 [===========================>..] - ETA: 40s

    100%|██████████| 64/64 [00:00<00:00, 141.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1147/1227 [===========================>..] - ETA: 39s

    100%|██████████| 64/64 [00:00<00:00, 146.91it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1148/1227 [===========================>..] - ETA: 39s

    100%|██████████| 64/64 [00:00<00:00, 136.85it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1149/1227 [===========================>..] - ETA: 38s

    100%|██████████| 64/64 [00:00<00:00, 141.37it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1150/1227 [===========================>..] - ETA: 38s

    100%|██████████| 64/64 [00:00<00:00, 138.53it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1151/1227 [===========================>..] - ETA: 37s

    100%|██████████| 64/64 [00:00<00:00, 142.03it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1152/1227 [===========================>..] - ETA: 37s

    100%|██████████| 64/64 [00:00<00:00, 154.20it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1153/1227 [===========================>..] - ETA: 36s

    100%|██████████| 64/64 [00:00<00:00, 137.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1154/1227 [===========================>..] - ETA: 36s

    100%|██████████| 64/64 [00:00<00:00, 140.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1155/1227 [===========================>..] - ETA: 35s

    100%|██████████| 64/64 [00:00<00:00, 134.95it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1156/1227 [===========================>..] - ETA: 35s

    100%|██████████| 64/64 [00:00<00:00, 140.17it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1157/1227 [===========================>..] - ETA: 34s

    100%|██████████| 64/64 [00:00<00:00, 137.82it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1158/1227 [===========================>..] - ETA: 34s

    100%|██████████| 64/64 [00:00<00:00, 145.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1159/1227 [===========================>..] - ETA: 33s

    100%|██████████| 64/64 [00:00<00:00, 138.62it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1160/1227 [===========================>..] - ETA: 33s

    100%|██████████| 64/64 [00:00<00:00, 143.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1161/1227 [===========================>..] - ETA: 32s

    100%|██████████| 64/64 [00:00<00:00, 131.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1162/1227 [===========================>..] - ETA: 32s

    100%|██████████| 64/64 [00:00<00:00, 142.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1163/1227 [===========================>..] - ETA: 31s

    100%|██████████| 64/64 [00:00<00:00, 136.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1164/1227 [===========================>..] - ETA: 31s

    100%|██████████| 64/64 [00:00<00:00, 133.57it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1165/1227 [===========================>..] - ETA: 30s

    100%|██████████| 64/64 [00:00<00:00, 150.19it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1166/1227 [===========================>..] - ETA: 30s

    100%|██████████| 64/64 [00:00<00:00, 137.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1167/1227 [===========================>..] - ETA: 29s

    100%|██████████| 64/64 [00:00<00:00, 137.92it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1168/1227 [===========================>..] - ETA: 29s

    100%|██████████| 64/64 [00:00<00:00, 135.63it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1169/1227 [===========================>..] - ETA: 28s

    100%|██████████| 64/64 [00:00<00:00, 137.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1170/1227 [===========================>..] - ETA: 28s

    100%|██████████| 64/64 [00:00<00:00, 136.84it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1171/1227 [===========================>..] - ETA: 27s

    100%|██████████| 64/64 [00:00<00:00, 142.10it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1172/1227 [===========================>..] - ETA: 27s

    100%|██████████| 64/64 [00:00<00:00, 135.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1173/1227 [===========================>..] - ETA: 26s

    100%|██████████| 64/64 [00:00<00:00, 141.02it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1174/1227 [===========================>..] - ETA: 26s

    100%|██████████| 64/64 [00:00<00:00, 133.83it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1175/1227 [===========================>..] - ETA: 25s

    100%|██████████| 64/64 [00:00<00:00, 137.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1176/1227 [===========================>..] - ETA: 25s

    100%|██████████| 64/64 [00:00<00:00, 138.34it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1177/1227 [===========================>..] - ETA: 24s

    100%|██████████| 64/64 [00:00<00:00, 138.27it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1178/1227 [===========================>..] - ETA: 24s

    100%|██████████| 64/64 [00:00<00:00, 140.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1179/1227 [===========================>..] - ETA: 23s

    100%|██████████| 64/64 [00:00<00:00, 133.28it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1180/1227 [===========================>..] - ETA: 23s

    100%|██████████| 64/64 [00:00<00:00, 136.57it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1181/1227 [===========================>..] - ETA: 22s

    100%|██████████| 64/64 [00:00<00:00, 136.40it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1182/1227 [===========================>..] - ETA: 22s

    100%|██████████| 64/64 [00:00<00:00, 136.22it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1183/1227 [===========================>..] - ETA: 21s

    100%|██████████| 64/64 [00:00<00:00, 137.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1184/1227 [===========================>..] - ETA: 21s

    100%|██████████| 64/64 [00:00<00:00, 137.60it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1185/1227 [===========================>..] - ETA: 20s

    100%|██████████| 64/64 [00:00<00:00, 135.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1186/1227 [===========================>..] - ETA: 20s

    100%|██████████| 64/64 [00:00<00:00, 134.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1187/1227 [============================>.] - ETA: 19s

    100%|██████████| 64/64 [00:00<00:00, 130.97it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1188/1227 [============================>.] - ETA: 19s

    100%|██████████| 64/64 [00:00<00:00, 131.12it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1189/1227 [============================>.] - ETA: 18s

    100%|██████████| 64/64 [00:00<00:00, 138.18it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1190/1227 [============================>.] - ETA: 18s

    100%|██████████| 64/64 [00:00<00:00, 135.90it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1191/1227 [============================>.] - ETA: 17s

    100%|██████████| 64/64 [00:00<00:00, 135.66it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1192/1227 [============================>.] - ETA: 17s

    100%|██████████| 64/64 [00:00<00:00, 134.96it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1193/1227 [============================>.] - ETA: 16s

    100%|██████████| 64/64 [00:00<00:00, 133.43it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1194/1227 [============================>.] - ETA: 16s

    100%|██████████| 64/64 [00:00<00:00, 134.73it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1195/1227 [============================>.] - ETA: 15s

    100%|██████████| 64/64 [00:00<00:00, 140.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1196/1227 [============================>.] - ETA: 15s

    100%|██████████| 64/64 [00:00<00:00, 132.04it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1197/1227 [============================>.] - ETA: 14s

    100%|██████████| 64/64 [00:00<00:00, 134.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1198/1227 [============================>.] - ETA: 14s

    100%|██████████| 64/64 [00:00<00:00, 132.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1199/1227 [============================>.] - ETA: 13s

    100%|██████████| 64/64 [00:00<00:00, 136.74it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1200/1227 [============================>.] - ETA: 13s

    100%|██████████| 64/64 [00:00<00:00, 135.06it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1201/1227 [============================>.] - ETA: 12s

    100%|██████████| 64/64 [00:00<00:00, 136.16it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1202/1227 [============================>.] - ETA: 12s

    100%|██████████| 64/64 [00:00<00:00, 140.77it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1203/1227 [============================>.] - ETA: 11s

    100%|██████████| 64/64 [00:00<00:00, 133.32it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1204/1227 [============================>.] - ETA: 11s

    100%|██████████| 64/64 [00:00<00:00, 139.07it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1205/1227 [============================>.] - ETA: 10s

    100%|██████████| 64/64 [00:00<00:00, 130.36it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1206/1227 [============================>.] - ETA: 10s

    100%|██████████| 64/64 [00:00<00:00, 138.65it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1207/1227 [============================>.] - ETA: 10s

    100%|██████████| 64/64 [00:00<00:00, 138.89it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1208/1227 [============================>.] - ETA: 9s 

    100%|██████████| 64/64 [00:00<00:00, 145.86it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1209/1227 [============================>.] - ETA: 9s

    100%|██████████| 64/64 [00:00<00:00, 135.51it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1210/1227 [============================>.] - ETA: 8s

    100%|██████████| 64/64 [00:00<00:00, 138.47it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1211/1227 [============================>.] - ETA: 8s

    100%|██████████| 64/64 [00:00<00:00, 134.61it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1212/1227 [============================>.] - ETA: 7s

    100%|██████████| 64/64 [00:00<00:00, 140.70it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1213/1227 [============================>.] - ETA: 7s

    100%|██████████| 64/64 [00:00<00:00, 134.44it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1214/1227 [============================>.] - ETA: 6s

    100%|██████████| 64/64 [00:00<00:00, 136.48it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1215/1227 [============================>.] - ETA: 6s

    100%|██████████| 64/64 [00:00<00:00, 140.45it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1216/1227 [============================>.] - ETA: 5s

    100%|██████████| 64/64 [00:00<00:00, 132.75it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1217/1227 [============================>.] - ETA: 5s

    100%|██████████| 64/64 [00:00<00:00, 132.33it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1218/1227 [============================>.] - ETA: 4s

    100%|██████████| 64/64 [00:00<00:00, 129.88it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1219/1227 [============================>.] - ETA: 4s

    100%|██████████| 64/64 [00:00<00:00, 137.59it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1220/1227 [============================>.] - ETA: 3s

    100%|██████████| 64/64 [00:00<00:00, 130.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1221/1227 [============================>.] - ETA: 3s

    100%|██████████| 64/64 [00:00<00:00, 137.64it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1222/1227 [============================>.] - ETA: 2s

    100%|██████████| 64/64 [00:00<00:00, 138.71it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1223/1227 [============================>.] - ETA: 2s

    100%|██████████| 64/64 [00:00<00:00, 134.30it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1224/1227 [============================>.] - ETA: 1s

    100%|██████████| 64/64 [00:00<00:00, 130.39it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1225/1227 [============================>.] - ETA: 1s

    100%|██████████| 64/64 [00:00<00:00, 133.13it/s]
      0%|          | 0/64 [00:00<?, ?it/s]

    1226/1227 [============================>.] - ETA: 0s

    100%|██████████| 64/64 [00:00<00:00, 136.57it/s]
      0%|          | 0/17 [00:00<?, ?it/s]

    1227/1227 [============================>.] - ETA: 0s

    100%|██████████| 17/17 [00:00<00:00, 126.79it/s]
     31%|███▏      | 20/64 [00:00<00:00, 192.64it/s]

    1228/1227 [==============================] - 606s 493ms/step
    


```python
temp_pred.shape
```




    (78545, 6)




```python
submission_df = pivot_test_data
submission_df['any'] = temp_pred[:,0]
submission_df['epidural'] = temp_pred[:,1]
submission_df['intraparenchymal'] = temp_pred[:,2]
submission_df['intraventricular'] = temp_pred[:,3]
submission_df['subarachnoid'] = temp_pred[:,4]
submission_df['subdural'] = temp_pred[:,5]
```


```python
submission_df = submission_df.melt(id_vars=['fileName'])
submission_df['ID'] = submission_df.fileName + '_' + submission_df.variable
submission_df['Label'] = submission_df['value']
print(submission_df.head(20))
```

            fileName variable     value                ID     Label
    0   ID_000012eaf      any  0.077172  ID_000012eaf_any  0.077172
    1   ID_0000ca2f6      any  0.132851  ID_0000ca2f6_any  0.132851
    2   ID_000259ccf      any  0.001006  ID_000259ccf_any  0.001006
    3   ID_0002d438a      any  0.176040  ID_0002d438a_any  0.176040
    4   ID_00032d440      any  0.041040  ID_00032d440_any  0.041040
    5   ID_00044a417      any  0.069968  ID_00044a417_any  0.069968
    6   ID_0004cd66f      any  0.059623  ID_0004cd66f_any  0.059623
    7   ID_0005b2d86      any  0.063642  ID_0005b2d86_any  0.063642
    8   ID_0005db660      any  0.023117  ID_0005db660_any  0.023117
    9   ID_000624786      any  0.037962  ID_000624786_any  0.037962
    10  ID_0006441d0      any  0.048462  ID_0006441d0_any  0.048462
    11  ID_00067e05e      any  0.002257  ID_00067e05e_any  0.002257
    12  ID_000716c43      any  0.066924  ID_000716c43_any  0.066924
    13  ID_0007c5cb8      any  0.172517  ID_0007c5cb8_any  0.172517
    14  ID_00086a66f      any  0.045856  ID_00086a66f_any  0.045856
    15  ID_0008f134d      any  0.227791  ID_0008f134d_any  0.227791
    16  ID_000920cd1      any  0.057263  ID_000920cd1_any  0.057263
    17  ID_0009c4591      any  0.103415  ID_0009c4591_any  0.103415
    18  ID_000b8242c      any  0.137407  ID_000b8242c_any  0.137407
    19  ID_000dcad55      any  0.250025  ID_000dcad55_any  0.250025
    


```python
submission_df = submission_df.drop(['fileName','variable','value'],axis = 1)
print(submission_df.head(20))
```

                      ID     Label
    0   ID_000012eaf_any  0.077172
    1   ID_0000ca2f6_any  0.132851
    2   ID_000259ccf_any  0.001006
    3   ID_0002d438a_any  0.176040
    4   ID_00032d440_any  0.041040
    5   ID_00044a417_any  0.069968
    6   ID_0004cd66f_any  0.059623
    7   ID_0005b2d86_any  0.063642
    8   ID_0005db660_any  0.023117
    9   ID_000624786_any  0.037962
    10  ID_0006441d0_any  0.048462
    11  ID_00067e05e_any  0.002257
    12  ID_000716c43_any  0.066924
    13  ID_0007c5cb8_any  0.172517
    14  ID_00086a66f_any  0.045856
    15  ID_0008f134d_any  0.227791
    16  ID_000920cd1_any  0.057263
    17  ID_0009c4591_any  0.103415
    18  ID_000b8242c_any  0.137407
    19  ID_000dcad55_any  0.250025
    


```python
submission_df.to_csv('submission.csv', index=False)
```
