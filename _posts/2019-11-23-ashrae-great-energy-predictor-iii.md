---
title: "ASHRAE - Great Energy Predictor III"
date: 2019-11-23
tags: [Kaggle, Energy Predictor, ASHRAE, Machine Learning, LightGBM]
excerpt: "How much energy will a building consume?"
header:
  overlay_image: "/images/ashrae-great-energy-predictor-iii/header.png"
  caption: "Photo by Federico Beccari on Unsplash"
mathjax: "true"
---

## Overview

Q: How much does it cost to cool a skyscraper in the summer?<br>
A: A lot! And not just in dollars, but in environmental impact.

Thankfully, significant investments are being made to improve building efficiencies to reduce costs and emissions. The question is, are the improvements working? That’s where you come in. Under pay-for-performance financing, the building owner makes payments based on the difference between their real energy consumption and what they would have used without any retrofits. The latter values have to come from a model. Current methods of estimation are fragmented and do not scale well. Some assume a specific meter type or don’t work with different building types.

In this project, we will develop accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

### About the Host

Founded in 1894, ASHRAE serves to advance the arts and sciences of heating, ventilation, air conditioning refrigeration and their allied fields. ASHRAE members represent building system design and industrial process professionals around the world. With over 54,000 members serving in 132 countries, ASHRAE supports research, standards writing, publishing and continuing education - shaping tomorrow’s built environment today.

## Data

Assessing the value of energy efficiency improvements can be challenging as there's no way to truly know how much energy a building would have used without the improvements. The best we can do is to build counterfactual models. Once a building is overhauled the new (lower) energy consumption is compared against modeled values for the original building to calculate the savings from the retrofit. More accurate models could support better market incentives and enable lower cost financing.

This challenges you to build these counterfactual models across four energy types based on historic usage rates and observed weather. The dataset includes three years of hourly meter readings from over one thousand buildings at several different sites around the world.

## Files

**train.csv**

1. building_id - Foreign key for the building metadata.
2. meter - The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}. Not every building has all meter types.
3. timestamp - When the measurement was taken
4. meter_reading - The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.

**building_meta.csv**

1. site_id - Foreign key for the weather files.
2. building_id - Foreign key for training.csv
3. primary_use - Indicator of the primary category of activities for the building based on EnergyStar property type definitions
4. square_feet - Gross floor area of the building
5. year_built - Year building was opened
6. floor_count - Number of floors of the building

**weather_[train/test].csv**

Weather data from a meteorological station as close as possible to the site.

1. site_id
2. air_temperature - Degrees Celsius
3. cloud_coverage - Portion of the sky covered in clouds, in oktas
4. dew_temperature - Degrees Celsius
5. precip_depth_1_hr - Millimeters
6. sea_level_pressure - Millibar/hectopascals
7. wind_direction - Compass direction (0-360)
8. wind_speed - Meters per second

**test.csv**

The submission files use row numbers for ID codes in order to save space on the file uploads. test.csv has no feature data; it exists so you can get your predictions into the correct order.

1. row_id - Row id for your submission file
2. building_id - Building id code
3. meter - The meter id code
4. timestamp - Timestamps for the test data period

## So lets begin with complete EDA...

```python
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/ashrae-energy-prediction/train.csv
    /kaggle/input/ashrae-energy-prediction/building_metadata.csv
    /kaggle/input/ashrae-energy-prediction/sample_submission.csv
    /kaggle/input/ashrae-energy-prediction/weather_test.csv
    /kaggle/input/ashrae-energy-prediction/weather_train.csv
    /kaggle/input/ashrae-energy-prediction/test.csv

## Load Data

```python
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
```

### Features that are likely predictive:

**Buildings**

* primary_use
* square_feet
* year_built
* floor_count (may be too sparse to use)

**Weather**

* time of day
* holiday
* weekend
* cloud_coverage + lags
* dew_temperature + lags
* precip_depth + lags
* sea_level_pressure + lags
* wind_direction + lags
* wind_speed + lags

**Train**

* max, mean, min, std of the specific building historically
* number of meters
* number of buildings at a siteid

```python
building_df
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
      <th>site_id</th>
      <th>building_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Education</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>Education</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>Education</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>Education</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>Education</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>15</td>
      <td>1444</td>
      <td>Entertainment/public assembly</td>
      <td>19619</td>
      <td>1914.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>15</td>
      <td>1445</td>
      <td>Education</td>
      <td>4298</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>15</td>
      <td>1446</td>
      <td>Entertainment/public assembly</td>
      <td>11265</td>
      <td>1997.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>15</td>
      <td>1447</td>
      <td>Lodging/residential</td>
      <td>29775</td>
      <td>2001.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>15</td>
      <td>1448</td>
      <td>Office</td>
      <td>92271</td>
      <td>2001.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1449 rows × 6 columns</p>
</div>

```python
weather_train
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
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2016-01-01 01:00:00</td>
      <td>24.4</td>
      <td>NaN</td>
      <td>21.1</td>
      <td>-1.0</td>
      <td>1020.2</td>
      <td>70.0</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2016-01-01 02:00:00</td>
      <td>22.8</td>
      <td>2.0</td>
      <td>21.1</td>
      <td>0.0</td>
      <td>1020.2</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2016-01-01 03:00:00</td>
      <td>21.1</td>
      <td>2.0</td>
      <td>20.6</td>
      <td>0.0</td>
      <td>1020.1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2016-01-01 04:00:00</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>-1.0</td>
      <td>1020.0</td>
      <td>250.0</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>139768</th>
      <td>15</td>
      <td>2016-12-31 19:00:00</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>-8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>180.0</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>139769</th>
      <td>15</td>
      <td>2016-12-31 20:00:00</td>
      <td>2.8</td>
      <td>2.0</td>
      <td>-8.9</td>
      <td>NaN</td>
      <td>1007.4</td>
      <td>180.0</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>139770</th>
      <td>15</td>
      <td>2016-12-31 21:00:00</td>
      <td>2.8</td>
      <td>NaN</td>
      <td>-7.2</td>
      <td>NaN</td>
      <td>1007.5</td>
      <td>180.0</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>139771</th>
      <td>15</td>
      <td>2016-12-31 22:00:00</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>-6.7</td>
      <td>NaN</td>
      <td>1008.0</td>
      <td>170.0</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>139772</th>
      <td>15</td>
      <td>2016-12-31 23:00:00</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
    </tr>
  </tbody>
</table>
<p>139773 rows × 9 columns</p>
</div>

```python
train
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
      <th>building_id</th>
      <th>meter</th>
      <th>timestamp</th>
      <th>meter_reading</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>2016-01-01 00:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20216095</th>
      <td>1444</td>
      <td>0</td>
      <td>2016-12-31 23:00:00</td>
      <td>8.750</td>
    </tr>
    <tr>
      <th>20216096</th>
      <td>1445</td>
      <td>0</td>
      <td>2016-12-31 23:00:00</td>
      <td>4.825</td>
    </tr>
    <tr>
      <th>20216097</th>
      <td>1446</td>
      <td>0</td>
      <td>2016-12-31 23:00:00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>20216098</th>
      <td>1447</td>
      <td>0</td>
      <td>2016-12-31 23:00:00</td>
      <td>159.575</td>
    </tr>
    <tr>
      <th>20216099</th>
      <td>1448</td>
      <td>0</td>
      <td>2016-12-31 23:00:00</td>
      <td>2.850</td>
    </tr>
  </tbody>
</table>
<p>20216100 rows × 4 columns</p>
</div>

```python
train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
```

```python
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month
```

```python
# looks like there may be some errors with some of the readings
train[train["site_id"] == 0].plot("timestamp", "meter_reading")
```

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_21_1.png)

```python
train[train["site_id"] == 2].plot("timestamp", "meter_reading")
```

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_22_1.png)

```python
train[["hour", "day", "weekend", "month"]]
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
      <th>hour</th>
      <th>day</th>
      <th>weekend</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20216095</th>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216096</th>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216097</th>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216098</th>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216099</th>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>20216100 rows × 4 columns</p>
</div>

```python
train = train.drop("timestamp", axis = 1)
```

```python
from sklearn.preprocessing import LabelEncoder
```

```python
le = LabelEncoder()
train["primary_use"] = le.fit_transform(train["primary_use"])
```

```python
categoricals = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]
```

```python
train
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
      <th>building_id</th>
      <th>meter</th>
      <th>meter_reading</th>
      <th>site_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>precip_depth_1_hr</th>
      <th>sea_level_pressure</th>
      <th>wind_direction</th>
      <th>wind_speed</th>
      <th>hour</th>
      <th>day</th>
      <th>weekend</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0.000</td>
      <td>0</td>
      <td>0</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>1019.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20216095</th>
      <td>1444</td>
      <td>0</td>
      <td>8.750</td>
      <td>15</td>
      <td>1</td>
      <td>19619</td>
      <td>1914.0</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216096</th>
      <td>1445</td>
      <td>0</td>
      <td>4.825</td>
      <td>15</td>
      <td>0</td>
      <td>4298</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216097</th>
      <td>1446</td>
      <td>0</td>
      <td>0.000</td>
      <td>15</td>
      <td>1</td>
      <td>11265</td>
      <td>1997.0</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216098</th>
      <td>1447</td>
      <td>0</td>
      <td>159.575</td>
      <td>15</td>
      <td>4</td>
      <td>29775</td>
      <td>2001.0</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216099</th>
      <td>1448</td>
      <td>0</td>
      <td>2.850</td>
      <td>15</td>
      <td>6</td>
      <td>92271</td>
      <td>2001.0</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>-1.0</td>
      <td>1008.5</td>
      <td>180.0</td>
      <td>8.8</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>20216100 rows × 19 columns</p>
</div>

```python
drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]
numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature"]
```

```python
train[categoricals + numericals]
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
      <th>building_id</th>
      <th>primary_use</th>
      <th>hour</th>
      <th>day</th>
      <th>weekend</th>
      <th>month</th>
      <th>meter</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20216095</th>
      <td>1444</td>
      <td>1</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
      <td>0</td>
      <td>19619</td>
      <td>1914.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
    </tr>
    <tr>
      <th>20216096</th>
      <td>1445</td>
      <td>0</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
      <td>0</td>
      <td>4298</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
    </tr>
    <tr>
      <th>20216097</th>
      <td>1446</td>
      <td>1</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
      <td>0</td>
      <td>11265</td>
      <td>1997.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
    </tr>
    <tr>
      <th>20216098</th>
      <td>1447</td>
      <td>4</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
      <td>0</td>
      <td>29775</td>
      <td>2001.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
    </tr>
    <tr>
      <th>20216099</th>
      <td>1448</td>
      <td>6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
      <td>0</td>
      <td>92271</td>
      <td>2001.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
    </tr>
  </tbody>
</table>
<p>20216100 rows × 12 columns</p>
</div>

```python
feat_cols = categoricals + numericals
```

```python
train["meter_reading"].value_counts()
```

    0.0000       1873976
    20.0000        23363
    2.9307         23181
    36.6000        22154
    8.7921         21787
                  ...   
    72.6357            1
    2977.7000          1
    55.4186            1
    2977.4500          1
    15.3563            1
    Name: meter_reading, Length: 1688175, dtype: int64


```python
import matplotlib.pyplot as plt
top_buildings = train.groupby("building_id")["meter_reading"].mean().sort_values(ascending = False).iloc[:100]
for value in top_buildings.index:
     train[train["building_id"] == value]["meter_reading"].rolling(window = 24).mean().plot()
     plt.show()
```

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_0.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_1.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_2.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_3.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_4.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_5.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_6.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_7.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_8.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_9.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_10.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_11.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_12.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_13.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_14.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_15.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_16.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_17.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_18.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_19.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_20.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_21.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_22.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_23.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_24.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_25.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_26.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_27.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_28.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_29.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_30.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_31.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_32.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_33.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_34.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_35.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_36.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_37.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_38.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_39.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_40.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_41.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_42.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_43.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_44.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_45.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_46.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_47.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_48.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_49.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_50.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_51.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_52.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_53.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_54.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_55.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_56.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_57.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_58.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_59.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_60.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_61.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_62.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_63.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_64.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_65.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_66.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_67.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_68.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_69.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_70.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_71.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_72.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_73.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_74.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_75.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_76.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_77.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_78.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_79.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_80.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_81.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_82.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_83.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_84.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_85.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_86.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_87.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_88.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_89.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_90.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_91.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_92.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_93.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_94.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_95.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_96.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_97.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_98.png)

![png](/images/ashrae-great-energy-predictor-iii/ashrae-great-energy-predictor-iii_33_99.png)

```python
target = np.log1p(train["meter_reading"])
del train["meter_reading"]
train = train.drop(drop_cols + ["site_id", "floor_count"], axis = 1)
```

```python
train
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
      <th>building_id</th>
      <th>meter</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
      <th>hour</th>
      <th>day</th>
      <th>weekend</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>25.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20216095</th>
      <td>1444</td>
      <td>0</td>
      <td>1</td>
      <td>19619</td>
      <td>1914.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216096</th>
      <td>1445</td>
      <td>0</td>
      <td>0</td>
      <td>4298</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216097</th>
      <td>1446</td>
      <td>0</td>
      <td>1</td>
      <td>11265</td>
      <td>1997.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216098</th>
      <td>1447</td>
      <td>0</td>
      <td>4</td>
      <td>29775</td>
      <td>2001.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>20216099</th>
      <td>1448</td>
      <td>0</td>
      <td>6</td>
      <td>92271</td>
      <td>2001.0</td>
      <td>1.7</td>
      <td>NaN</td>
      <td>-5.6</td>
      <td>23</td>
      <td>31</td>
      <td>5</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>20216100 rows × 12 columns</p>
</div>

```python
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist
```

```python
train, NAlist = reduce_mem_usage(train)
```

    Memory usage of properties dataframe is : 2005.0758361816406  MB
    ******************************
    Column:  building_id
    dtype before:  int64
    min for this col:  0
    max for this col:  1448
    dtype after:  uint16
    ******************************
    ******************************
    Column:  meter
    dtype before:  int64
    min for this col:  0
    max for this col:  3
    dtype after:  uint8
    ******************************
    ******************************
    Column:  primary_use
    dtype before:  int64
    min for this col:  0
    max for this col:  15
    dtype after:  uint8
    ******************************
    ******************************
    Column:  square_feet
    dtype before:  int64
    min for this col:  283
    max for this col:  875000
    dtype after:  uint32
    ******************************
    ******************************
    Column:  year_built
    dtype before:  float64
    min for this col:  1900.0
    max for this col:  2017.0
    dtype after:  uint16
    ******************************
    ******************************
    Column:  air_temperature
    dtype before:  float64
    min for this col:  -28.9
    max for this col:  47.2
    dtype after:  float32
    ******************************
    ******************************
    Column:  cloud_coverage
    dtype before:  float64
    min for this col:  0.0
    max for this col:  9.0
    dtype after:  uint8
    ******************************
    ******************************
    Column:  dew_temperature
    dtype before:  float64
    min for this col:  -35.0
    max for this col:  26.1
    dtype after:  float32
    ******************************
    ******************************
    Column:  hour
    dtype before:  int64
    min for this col:  0
    max for this col:  23
    dtype after:  uint8
    ******************************
    ******************************
    Column:  day
    dtype before:  int64
    min for this col:  1
    max for this col:  31
    dtype after:  uint8
    ******************************
    ******************************
    Column:  weekend
    dtype before:  int64
    min for this col:  0
    max for this col:  6
    dtype after:  uint8
    ******************************
    ******************************
    Column:  month
    dtype before:  int64
    min for this col:  1
    max for this col:  12
    dtype after:  uint8
    ******************************
    ___MEMORY USAGE AFTER COMPLETION:___
    Memory usage is:  597.6668357849121  MB
    This is  29.807692307692307 % of the initial size


## LightGBM

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = False, random_state = 42)
error = 0
models = []
for i, (train_index, val_index) in enumerate(kf.split(train)):
    if i + 1 < num_folds:
        continue
    print(train_index.max(), val_index.min())
    train_X = train[feat_cols].iloc[train_index]
    val_X = train[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y > 0)
    lgb_eval = lgb.Dataset(val_X, val_y > 0)
    params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq' : 5
            }
    gbm_class = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)

    lgb_train = lgb.Dataset(train_X[train_y > 0], train_y[train_y > 0])
    lgb_eval = lgb.Dataset(val_X[val_y > 0] , val_y[val_y > 0])
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq' : 5
            }
    gbm_regress = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)

    y_pred = (gbm_class.predict(val_X, num_iteration=gbm_class.best_iteration) > .5) *\
    (gbm_regress.predict(val_X, num_iteration=gbm_regress.best_iteration))
    error += np.sqrt(mean_squared_error(y_pred, (val_y)))/num_folds
    print(np.sqrt(mean_squared_error(y_pred, (val_y))))
    break
print(error)
```
    16172879 16172880
    Training until validation scores don't improve for 20 rounds
    [20]	training's binary_logloss: 0.179465	valid_1's binary_logloss: 0.179322
    [40]	training's binary_logloss: 0.157622	valid_1's binary_logloss: 0.175654
    [60]	training's binary_logloss: 0.146057	valid_1's binary_logloss: 0.172846
    [80]	training's binary_logloss: 0.137743	valid_1's binary_logloss: 0.173728
    Early stopping, best iteration is:
    [64]	training's binary_logloss: 0.144102	valid_1's binary_logloss: 0.172487
    Training until validation scores don't improve for 20 rounds
    [20]	training's rmse: 0.872481	valid_1's rmse: 0.933909
    [40]	training's rmse: 0.765797	valid_1's rmse: 0.861977
    [60]	training's rmse: 0.703914	valid_1's rmse: 0.808794
    [80]	training's rmse: 0.658989	valid_1's rmse: 0.784859
    [100]	training's rmse: 0.619144	valid_1's rmse: 0.767098
    [120]	training's rmse: 0.59556	valid_1's rmse: 0.758259
    [140]	training's rmse: 0.57568	valid_1's rmse: 0.746786
    [160]	training's rmse: 0.558876	valid_1's rmse: 0.722893
    [180]	training's rmse: 0.544507	valid_1's rmse: 0.716829
    [200]	training's rmse: 0.533321	valid_1's rmse: 0.710311
    [220]	training's rmse: 0.522582	valid_1's rmse: 0.705353
    [240]	training's rmse: 0.515471	valid_1's rmse: 0.703147
    [260]	training's rmse: 0.506647	valid_1's rmse: 0.701818
    [280]	training's rmse: 0.498045	valid_1's rmse: 0.699002
    [300]	training's rmse: 0.492483	valid_1's rmse: 0.69307
    [320]	training's rmse: 0.486364	valid_1's rmse: 0.690492
    [340]	training's rmse: 0.480546	valid_1's rmse: 0.692244
    Early stopping, best iteration is:
    [323]	training's rmse: 0.485635	valid_1's rmse: 0.690325
    1.3222083934671207
    0.26444167869342416

```python
sorted(zip(gbm_regress.feature_importance(), gbm_regress.feature_name()),reverse = True)
```

    [(2349, 'building_id'),
     (2111, 'square_feet'),
     (1085, 'meter'),
     (803, 'primary_use'),
     (767, 'year_built'),
     (740, 'month'),
     (680, 'hour'),
     (514, 'air_temperature'),
     (253, 'dew_temperature'),
     (205, 'weekend'),
     (158, 'day'),
     (25, 'cloud_coverage')]

```python
import gc
del train
del train_X, val_X, lgb_train, lgb_eval, train_y, val_y, y_pred, target
```

```python
gc.collect()
```

> 7550

```python
#preparing test data
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
del building_df
gc.collect()
```
> 0

```python
test
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
      <th>row_id</th>
      <th>building_id</th>
      <th>meter</th>
      <th>timestamp</th>
      <th>site_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>Education</td>
      <td>7432</td>
      <td>2008.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>Education</td>
      <td>2720</td>
      <td>2004.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>Education</td>
      <td>5376</td>
      <td>1991.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>Education</td>
      <td>23685</td>
      <td>2002.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>Education</td>
      <td>116607</td>
      <td>1975.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41697595</th>
      <td>41697595</td>
      <td>1444</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>Entertainment/public assembly</td>
      <td>19619</td>
      <td>1914.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41697596</th>
      <td>41697596</td>
      <td>1445</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>Education</td>
      <td>4298</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41697597</th>
      <td>41697597</td>
      <td>1446</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>Entertainment/public assembly</td>
      <td>11265</td>
      <td>1997.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41697598</th>
      <td>41697598</td>
      <td>1447</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>Lodging/residential</td>
      <td>29775</td>
      <td>2001.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41697599</th>
      <td>41697599</td>
      <td>1448</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>Office</td>
      <td>92271</td>
      <td>2001.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>41697600 rows × 9 columns</p>
</div>

```python
test["primary_use"] = le.transform(test["primary_use"])
```

```python
test, NAlist = reduce_mem_usage(test)
```

    Memory usage of properties dataframe is : 3181.2744140625  MB
    ******************************
    Column:  row_id
    dtype before:  int64
    min for this col:  0
    max for this col:  41697599
    dtype after:  uint32
    ******************************
    ******************************
    Column:  building_id
    dtype before:  int64
    min for this col:  0
    max for this col:  1448
    dtype after:  uint16
    ******************************
    ******************************
    Column:  meter
    dtype before:  int64
    min for this col:  0
    max for this col:  3
    dtype after:  uint8
    ******************************
    ******************************
    Column:  site_id
    dtype before:  int64
    min for this col:  0
    max for this col:  15
    dtype after:  uint8
    ******************************
    ******************************
    Column:  primary_use
    dtype before:  int64
    min for this col:  0
    max for this col:  15
    dtype after:  uint8
    ******************************
    ******************************
    Column:  square_feet
    dtype before:  int64
    min for this col:  283
    max for this col:  875000
    dtype after:  uint32
    ******************************
    ******************************
    Column:  year_built
    dtype before:  float64
    min for this col:  1900.0
    max for this col:  2017.0
    dtype after:  uint16
    ******************************
    ******************************
    Column:  floor_count
    dtype before:  float64
    min for this col:  1.0
    max for this col:  26.0
    dtype after:  uint8
    ******************************
    ___MEMORY USAGE AFTER COMPLETION:___
    Memory usage is:  1272.509765625  MB
    This is  40.0 % of the initial size

```python
test
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
      <th>row_id</th>
      <th>building_id</th>
      <th>meter</th>
      <th>timestamp</th>
      <th>site_id</th>
      <th>primary_use</th>
      <th>square_feet</th>
      <th>year_built</th>
      <th>floor_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>7432</td>
      <td>2008</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>2720</td>
      <td>2004</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>5376</td>
      <td>1991</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>23685</td>
      <td>2002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>0</td>
      <td>0</td>
      <td>116607</td>
      <td>1975</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41697595</th>
      <td>41697595</td>
      <td>1444</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>1</td>
      <td>19619</td>
      <td>1914</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41697596</th>
      <td>41697596</td>
      <td>1445</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>0</td>
      <td>4298</td>
      <td>1899</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41697597</th>
      <td>41697597</td>
      <td>1446</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>1</td>
      <td>11265</td>
      <td>1997</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41697598</th>
      <td>41697598</td>
      <td>1447</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>4</td>
      <td>29775</td>
      <td>2001</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41697599</th>
      <td>41697599</td>
      <td>1448</td>
      <td>0</td>
      <td>2018-05-09 07:00:00</td>
      <td>15</td>
      <td>6</td>
      <td>92271</td>
      <td>2001</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>41697600 rows × 9 columns</p>
</div>

```python
gc.collect()
```

> 0

```python
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
weather_test = weather_test.drop(drop_cols, axis = 1)
```

```python
weather_test
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
      <th>site_id</th>
      <th>timestamp</th>
      <th>air_temperature</th>
      <th>cloud_coverage</th>
      <th>dew_temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-01-01 00:00:00</td>
      <td>17.8</td>
      <td>4.0</td>
      <td>11.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2017-01-01 01:00:00</td>
      <td>17.8</td>
      <td>2.0</td>
      <td>12.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2017-01-01 02:00:00</td>
      <td>16.1</td>
      <td>0.0</td>
      <td>12.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2017-01-01 03:00:00</td>
      <td>17.2</td>
      <td>0.0</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2017-01-01 04:00:00</td>
      <td>16.7</td>
      <td>2.0</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>277238</th>
      <td>15</td>
      <td>2018-12-31 19:00:00</td>
      <td>3.3</td>
      <td>NaN</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>277239</th>
      <td>15</td>
      <td>2018-12-31 20:00:00</td>
      <td>2.8</td>
      <td>NaN</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>277240</th>
      <td>15</td>
      <td>2018-12-31 21:00:00</td>
      <td>2.8</td>
      <td>NaN</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>277241</th>
      <td>15</td>
      <td>2018-12-31 22:00:00</td>
      <td>2.8</td>
      <td>NaN</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>277242</th>
      <td>15</td>
      <td>2018-12-31 23:00:00</td>
      <td>3.3</td>
      <td>NaN</td>
      <td>2.2</td>
    </tr>
  </tbody>
</table>
<p>277243 rows × 5 columns</p>
</div>

```python
test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test
```

```python
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)
test["day"] = test["timestamp"].dt.day.astype(np.uint8)
test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)
test["month"] = test["timestamp"].dt.month.astype(np.uint8)
test = test[feat_cols]
```

```python
from tqdm import tqdm
i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    res.append(np.expm1((gbm_class.predict(test.iloc[i:i+step_size], num_iteration=gbm_class.best_iteration) > .5) *\
    (gbm_regress.predict(test.iloc[i:i+step_size], num_iteration=gbm_regress.best_iteration))))
    i+=step_size
```

> 100%|██████████| 834/834 [07:07<00:00,  1.95it/s]


```python
del test
res = np.concatenate(res)
```

```python
pd.DataFrame(res).describe()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.169760e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.046305e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.749480e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.467108e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.454626e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.412281e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.586173e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.592061e+07</td>
    </tr>
  </tbody>
</table>
</div>

```python
res.shape
```

> (41697600,)

## Submission

```python
sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")
sub["meter_reading"] = res
sub.to_csv("submission.csv", index = False)
```