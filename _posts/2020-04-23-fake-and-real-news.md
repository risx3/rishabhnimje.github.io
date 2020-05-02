---
title: "Fake and Real News"
date: 2020-04-23
tags: [Kaggle, Keras, Machine Learning, Neural Network]
excerpt: "Classifying the news"
header:
  overlay_image: "/images/fake-and-real-news/home-page.jpg"
  caption: "Photo by Hayden Walker on Unsplash"
mathjax: "true"
---

## Overview

Can you use this data set to make an algorithm able to determine if an article is fake news or not ?

## Data Description

Fake.csv file contains a list of articles considered as "fake" news. True.csv contains a list of articles considered as "real" news. Both the files contain

1. The title of the article
2. The text of the article
3. The subject of the article
4. The date that this article was posted at

## Files

* Fake.csv
* True.csv

## So let’s begin here…

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

> Using TensorFlow backend.


> /kaggle/input/fake-and-real-news-dataset/Fake.csv<br>
> /kaggle/input/fake-and-real-news-dataset/True.csv
    
## Load Data

```python
real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
```


```python
real.head()
```

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
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
  </tbody>
</table>
</div>


```python
fake.head()
```

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
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Sends Out Embarrassing New Year’...</td>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drunk Bragging Trump Staffer Started Russian ...</td>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sheriff David Clarke Becomes An Internet Joke...</td>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>News</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>News</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis Just Called Out Donald Trump Dur...</td>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>News</td>
      <td>December 25, 2017</td>
    </tr>
  </tbody>
</table>
</div>

## Pre-process Data

Let's add another column in data set as 'category' where category will be 1 if news is real and 0 if news is fake.

```python
real['category']=1
fake['category']=0
```

Now we will concatenate both the datasets

```python
df = pd.concat([real,fake])
```

```python
df.head()
```

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
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

Check for NULL values

```python
df.isna().sum()
```

    title       0
    text        0
    subject     0
    date        0
    category    0
    dtype: int64

Total number of news


```python
df.title.count()
```
> 44898

Number of news grouped by Subject

```python
df.subject.value_counts()
```
    politicsNews       11272
    worldnews          10145
    News                9050
    politics            6841
    left-news           4459
    Government News     1570
    US_News              783
    Middle-east          778
    Name: subject, dtype: int64

We now concatenate Text, Title and Subject in Text.

```python
df['text'] = df['text'] + " " + df['title'] + " " + df['subject']
del df['title']
del df['subject']
del df['date']
```

```python
df.head()
```

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
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
stop = set(stopwords.words('english'))
pnc = list(punctuation)
stop.update(pnc)
```

```python
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)
```

```python
df.text = df.text.apply(stem_text)
```

Splitting dataset in train set and test set

```python
X_train,X_test,y_train,y_test = train_test_split(df.text,df.category)
```

```python
cv = CountVectorizer(min_df=0,max_df=1,ngram_range=(1,2))
cv_train = cv.fit_transform(X_train)
cv_test = cv.transform(X_test)

print('BOW_cv_train:',cv_train.shape)
print('BOW_cv_test:',cv_test.shape)
```

> BOW_cv_train: (33673, 1950850)<br>
> BOW_cv_test: (11225, 1950850)

## Defining Model

```python
model = Sequential()
model.add(Dense(units = 100 , activation = 'relu' , input_dim = cv_train.shape[1]))
model.add(Dense(units = 50 , activation = 'relu'))
model.add(Dense(units = 25 , activation = 'relu'))
model.add(Dense(units = 10 , activation = 'relu'))
model.add(Dense(units = 1 , activation = 'sigmoid'))
```

## Compile Model

```python
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
```

## Fit Model

```python
model.fit(cv_train,y_train , epochs = 5)
```
    Epoch 1/5
    33673/33673 [==============================] - 542s 16ms/step - loss: 0.6904 - accuracy: 0.5224
    Epoch 2/5
    33673/33673 [==============================] - 559s 17ms/step - loss: 0.1654 - accuracy: 0.9452
    Epoch 3/5
    33673/33673 [==============================] - 571s 17ms/step - loss: 0.0428 - accuracy: 0.9885
    Epoch 4/5
    33673/33673 [==============================] - 575s 17ms/step - loss: 0.0414 - accuracy: 0.9888
    Epoch 5/5
    33673/33673 [==============================] - 568s 17ms/step - loss: 0.0418 - accuracy: 0.9888

## Prediction

```python
pred = model.predict(cv_test)
for i in range(len(pred)):
    if(pred[i] > 0.5):
        pred[i] = 1
    else:
        pred[i] = 0
```

## Evaluate Model

### Accuracy

```python
accuracy_score(pred,y_test)
```

> 0.90271714922049

### Classification Report

```python
cv_report = classification_report(y_test,pred,target_names = ['0','1'])
print(cv_report)
```

                  precision    recall  f1-score   support
    
               0       0.92      0.89      0.91      5929
               1       0.88      0.91      0.90      5296
    
        accuracy                           0.90     11225
       macro avg       0.90      0.90      0.90     11225
    weighted avg       0.90      0.90      0.90     11225

### Confusion Matrix

```python
cm_cv = confusion_matrix(y_test,pred)
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])
cm_cv.index.name = 'Actual'
cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')
```

![png](/images/fake-and-real-news/notebook_42_1.png)
