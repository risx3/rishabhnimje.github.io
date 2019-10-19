---
title: "Digit Recognizer"
date: 2019-10-19
tags: [Kaggle, Digit Recognition, Keras, Machine Learning, ]
excerpt: "Can you detect fraud from customer transactions?"
header:
  overlay_image: "/images/ieee-cis-fraud-detection/home-page.jpg"
  caption: "Photo by Arget on Unsplash"
mathjax: "true"
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from subprocess import check_output
from sklearn.metrics import confusion_matrix
import itertools
%matplotlib inline

# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

```

> /kaggle/input/digit-recognizer/train.csv<br>
> /kaggle/input/digit-recognizer/sample_submission.csv<br>
> /kaggle/input/digit-recognizer/test.csv<br>
    

## Load Data

```python
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.shape)
train.head()
```

> (42000, 785)


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
z_train = Counter(train['label'])
z_train
```




    Counter({1: 4684,
             0: 4132,
             4: 4072,
             7: 4401,
             3: 4351,
             5: 3795,
             8: 4063,
             9: 4188,
             2: 4177,
             6: 4137})




```python
sns.countplot(train['label'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcbe5108a58>




![png](digit-recognizer_files/digit-recognizer_4_1.png)



```python
print(test.shape)
test.head()
```

    (28000, 784)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>




```python
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels
x_test = test.values.astype('float32')
```


```python
%matplotlib inline
# preview the images first
plt.figure(figsize=(12,10))
x, y = 10, 4
for i in range(40):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
plt.show()
```


![png](digit-recognizer_files/digit-recognizer_7_0.png)



```python
x_train = x_train/255.0
x_test = x_test/255.0
```


```python
y_train
```




    array([1, 0, 1, ..., 7, 6, 9], dtype=int32)



## Printing the shape of the Datasets


```python
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```

    x_train shape: (42000, 784)
    42000 train samples
    28000 test samples
    

> ## Reshape


```python
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
X_test = x_test.reshape(x_test.shape[0], 28, 28,1)
```

## Implementing Keras


```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
batch_size = 64
num_classes = 10
epochs = 20
input_shape = (28, 28, 1)
```

    Using TensorFlow backend.
    


```python
# convert class vectors to binary class matrices One Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
```


```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 24, 24, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 12, 12, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 12, 12, 64)        18496     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 12, 12, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 6, 6, 128)         73856     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 6, 6, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4608)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               589952    
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 730,602
    Trainable params: 730,346
    Non-trainable params: 256
    _________________________________________________________________
    


```python
datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction],)
```

    Epoch 1/20
    590/590 [==============================] - 15s 26ms/step - loss: 0.3556 - accuracy: 0.8882 - val_loss: 0.0583 - val_accuracy: 0.9817
    Epoch 2/20
     10/590 [..............................] - ETA: 11s - loss: 0.1427 - accuracy: 0.9563

    /opt/conda/lib/python3.6/site-packages/keras/callbacks/callbacks.py:1042: RuntimeWarning: Reduce LR on plateau conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,val_accuracy,loss,accuracy,lr
      (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
    

    590/590 [==============================] - 11s 19ms/step - loss: 0.1105 - accuracy: 0.9662 - val_loss: 0.0560 - val_accuracy: 0.9838
    Epoch 3/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0859 - accuracy: 0.9731 - val_loss: 0.0278 - val_accuracy: 0.9917
    Epoch 4/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0708 - accuracy: 0.9780 - val_loss: 0.0526 - val_accuracy: 0.9826
    Epoch 5/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0600 - accuracy: 0.9818 - val_loss: 0.0292 - val_accuracy: 0.9912
    Epoch 6/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0571 - accuracy: 0.9823 - val_loss: 0.0280 - val_accuracy: 0.9910
    Epoch 7/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0514 - accuracy: 0.9849 - val_loss: 0.0279 - val_accuracy: 0.9900
    Epoch 8/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0528 - accuracy: 0.9835 - val_loss: 0.0203 - val_accuracy: 0.9933
    Epoch 9/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0449 - accuracy: 0.9859 - val_loss: 0.0371 - val_accuracy: 0.9883
    Epoch 10/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0432 - accuracy: 0.9871 - val_loss: 0.0231 - val_accuracy: 0.9924
    Epoch 11/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0445 - accuracy: 0.9864 - val_loss: 0.0284 - val_accuracy: 0.9914
    Epoch 12/20
    590/590 [==============================] - 12s 21ms/step - loss: 0.0429 - accuracy: 0.9867 - val_loss: 0.0312 - val_accuracy: 0.9902
    Epoch 13/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0386 - accuracy: 0.9882 - val_loss: 0.0243 - val_accuracy: 0.9938
    Epoch 14/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0372 - accuracy: 0.9888 - val_loss: 0.0227 - val_accuracy: 0.9936
    Epoch 15/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0363 - accuracy: 0.9889 - val_loss: 0.0196 - val_accuracy: 0.9940
    Epoch 16/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0355 - accuracy: 0.9895 - val_loss: 0.0172 - val_accuracy: 0.9938
    Epoch 17/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0353 - accuracy: 0.9887 - val_loss: 0.0220 - val_accuracy: 0.9931
    Epoch 18/20
    590/590 [==============================] - 11s 19ms/step - loss: 0.0355 - accuracy: 0.9892 - val_loss: 0.0170 - val_accuracy: 0.9945
    Epoch 19/20
    590/590 [==============================] - 12s 20ms/step - loss: 0.0341 - accuracy: 0.9898 - val_loss: 0.0172 - val_accuracy: 0.9950
    Epoch 20/20
    590/590 [==============================] - 16s 27ms/step - loss: 0.0334 - accuracy: 0.9905 - val_loss: 0.0169 - val_accuracy: 0.9955
    

## Basic Simple Plot And Evaluation


```python
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
```

    Final loss: 0.016873, final accuracy: 0.995476
    


```python
# Look at confusion matrix 
#Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))
```


![png](digit-recognizer_files/digit-recognizer_22_0.png)



```python
print(h.history.keys())
```

    dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy', 'lr'])
    


```python
accuracy = h.history['accuracy']
val_accuracy = h.history['val_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```


![png](digit-recognizer_files/digit-recognizer_24_0.png)



![png](digit-recognizer_files/digit-recognizer_24_1.png)



```python
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
```


![png](digit-recognizer_files/digit-recognizer_25_0.png)



```python
test_im = X_train[154]
plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
```




    <matplotlib.image.AxesImage at 0x7fcb51ebc748>




![png](digit-recognizer_files/digit-recognizer_26_1.png)



```python
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, output=layer_outputs)
activations = activation_model.predict(test_im.reshape(1,28,28,1))

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:3: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=Tensor("co..., outputs=[<tf.Tenso...)`
      This is separate from the ipykernel package so we can avoid doing imports until
    




    <matplotlib.image.AxesImage at 0x7fcb51e95da0>




![png](digit-recognizer_files/digit-recognizer_27_2.png)



```python
model.layers[:-1]# Droping The Last Dense Layer
```




    [<keras.layers.convolutional.Conv2D at 0x7fcbbf95c550>,
     <keras.layers.convolutional.Conv2D at 0x7fcbe4f702e8>,
     <keras.layers.pooling.MaxPooling2D at 0x7fcbbeeef9e8>,
     <keras.layers.core.Dropout at 0x7fcbbf95ca20>,
     <keras.layers.convolutional.Conv2D at 0x7fcbbeeefbe0>,
     <keras.layers.convolutional.Conv2D at 0x7fcbbf95c940>,
     <keras.layers.pooling.MaxPooling2D at 0x7fcbe6281f98>,
     <keras.layers.core.Dropout at 0x7fcbbee711d0>,
     <keras.layers.convolutional.Conv2D at 0x7fcbbee137b8>,
     <keras.layers.core.Dropout at 0x7fcbbee13518>,
     <keras.layers.core.Flatten at 0x7fcbbee4b240>,
     <keras.layers.core.Dense at 0x7fcbbee4bf98>,
     <keras.layers.normalization.BatchNormalization at 0x7fcbbee4bfd0>,
     <keras.layers.core.Dropout at 0x7fcbbed60d68>]




```python
layer_names = []
for layer in model.layers[:-1]:
    layer_names.append(layer.name) 
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('conv'):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:15: RuntimeWarning: invalid value encountered in true_divide
      from ipykernel import kernelapp as app
    


![png](digit-recognizer_files/digit-recognizer_29_1.png)



![png](digit-recognizer_files/digit-recognizer_29_2.png)



![png](digit-recognizer_files/digit-recognizer_29_3.png)



![png](digit-recognizer_files/digit-recognizer_29_4.png)



```python
layer_names = []
for layer in model.layers[:-1]:
    layer_names.append(layer.name) 
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('max'):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:15: RuntimeWarning: invalid value encountered in true_divide
      from ipykernel import kernelapp as app
    


![png](digit-recognizer_files/digit-recognizer_30_1.png)



![png](digit-recognizer_files/digit-recognizer_30_2.png)



```python
layer_names = []
for layer in model.layers[:-1]:
    layer_names.append(layer.name) 
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('drop'):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:15: RuntimeWarning: invalid value encountered in true_divide
      from ipykernel import kernelapp as app
    


![png](digit-recognizer_files/digit-recognizer_31_1.png)



![png](digit-recognizer_files/digit-recognizer_31_2.png)


## Classifcation Report


```python
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1)
Y_true_classes = np.argmax(Y_val, axis = 1)
```


```python
Y_pred_classes[:5], Y_true_classes[:5]
```




    (array([8, 1, 9, 9, 8]), array([8, 1, 9, 9, 8]))




```python
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(Y_true_classes, Y_pred_classes, target_names=target_names))
```

                  precision    recall  f1-score   support
    
         Class 0       1.00      1.00      1.00       408
         Class 1       1.00      1.00      1.00       471
         Class 2       1.00      0.99      1.00       420
         Class 3       0.99      1.00      1.00       506
         Class 4       0.99      1.00      1.00       397
         Class 5       0.99      0.99      0.99       339
         Class 6       1.00      1.00      1.00       402
         Class 7       1.00      1.00      1.00       438
         Class 8       0.99      0.99      0.99       403
         Class 9       1.00      1.00      1.00       416
    
        accuracy                           1.00      4200
       macro avg       1.00      1.00      1.00      4200
    weighted avg       1.00      1.00      1.00      4200
    
    


```python
predicted_classes = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("prediction.csv", index=False, header=True)
```


```python
model.save('my_model_1.h5')
json_string = model.to_json()
```


```python

```
