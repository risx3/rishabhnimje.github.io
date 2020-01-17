---
title: "House Prices: Advanced Regression Techniques"
date: 2020-01-17
tags: [Kaggle, Regression, AMES Iowa, Machine Learning]
excerpt: "Predict sales prices and practice feature engineering, RFs, and gradient boosting"
header:
  overlay_image: "/images/house-prices/housesbanner.png"
  caption: ""
mathjax: "true"
---

## Overview

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home

### Acknowledgments

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.

## Files

1. train.csv - the training set
2. test.csv - the test set
3. data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here

## Data Fields

Here's a brief version of what you'll find in the data description file.<br>
<br>
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.<br>
MSSubClass: The building class<br>
MSZoning: The general zoning classification<br>
LotFrontage: Linear feet of street connected to property<br>
LotArea: Lot size in square feet<br>
Street: Type of road access<br>
Alley: Type of alley access<br>
LotShape: General shape of property<br>
LandContour: Flatness of the property<br>
Utilities: Type of utilities available<br>
LotConfig: Lot configuration<br>
LandSlope: Slope of property<br>
Neighborhood: Physical locations within Ames city limits<br>
Condition1: Proximity to main road or railroad<br>
Condition2: Proximity to main road or railroad (if a second is present)<br>
BldgType: Type of dwelling<br>
HouseStyle: Style of dwelling<br>
OverallQual: Overall material and finish quality<br>
OverallCond: Overall condition rating<br>
YearBuilt: Original construction date<br>
YearRemodAdd: Remodel date<br>
RoofStyle: Type of roof<br>
RoofMatl: Roof material<br>
Exterior1st: Exterior covering on house<br>
Exterior2nd: Exterior covering on house (if more than one material)<br>
MasVnrType: Masonry veneer type<br>
MasVnrArea: Masonry veneer area in square feet<br>
ExterQual: Exterior material quality<br>
ExterCond: Present condition of the material on the exterior<br>
Foundation: Type of foundation<br>
BsmtQual: Height of the basement<br>
BsmtCond: General condition of the basement<br>
BsmtExposure: Walkout or garden level basement walls<br>
BsmtFinType1: Quality of basement finished area<br>
BsmtFinSF1: Type 1 finished square feet<br>
BsmtFinType2: Quality of second finished area (if present)<br>
BsmtFinSF2: Type 2 finished square feet<br>
BsmtUnfSF: Unfinished square feet of basement area<br>
TotalBsmtSF: Total square feet of basement area<br>
Heating: Type of heating<br>
HeatingQC: Heating quality and condition<br>
CentralAir: Central air conditioning<br>
Electrical: Electrical system<br>
1stFlrSF: First Floor square feet<br>
2ndFlrSF: Second floor square feet<br>
LowQualFinSF: Low quality finished square feet (all floors)<br>
GrLivArea: Above grade (ground) living area square feet<br>
BsmtFullBath: Basement full bathrooms<br>
BsmtHalfBath: Basement half bathrooms<br>
FullBath: Full bathrooms above grade<br>
HalfBath: Half baths above grade<br>
Bedroom: Number of bedrooms above basement level<br>
Kitchen: Number of kitchens<br>
KitchenQual: Kitchen quality<br>
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)<br>
Functional: Home functionality rating<br>
Fireplaces: Number of fireplaces<br>
FireplaceQu: Fireplace quality<br>
GarageType: Garage location<br>
GarageYrBlt: Year garage was built<br>
GarageFinish: Interior finish of the garage<br>
GarageCars: Size of garage in car capacity<br>
GarageArea: Size of garage in square feet<br>
GarageQual: Garage quality<br>
GarageCond: Garage condition<br>
PavedDrive: Paved driveway<br>
WoodDeckSF: Wood deck area in square feet<br>
OpenPorchSF: Open porch area in square feet<br>
EnclosedPorch: Enclosed porch area in square feet<br>
3SsnPorch: Three season porch area in square feet<br>
ScreenPorch: Screen porch area in square feet<br>
PoolArea: Pool area in square feet<br>
PoolQC: Pool quality<br>
Fence: Fence quality<br>
MiscFeature: Miscellaneous feature not covered in other categories<br>
MiscVal: $Value of miscellaneous feature<br>
MoSold: Month Sold<br>
YrSold: Year Sold<br>
SaleType: Type of sale<br>
SaleCondition: Condition of sale<br>

## So lets begin with complete EDA...

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
%matplotlib inline
sns.set()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/house-prices-advanced-regression-techniques/train.csv
    /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
    /kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
    /kaggle/input/house-prices-advanced-regression-techniques/test.csv

```python

**Settings and switches**

**Here one can choose settings for optimal performance and runtime.**  
**For example, nr_cv sets the number of cross validations used in GridsearchCV, and**  
**min_val_corr is the minimum value for the correlation coefficient to the target (only features with larger correlation will be used).** 

# setting the number of cross validations used in the Model part
nr_cv = 5

# switch for using log values for SalePrice and features
use_logvals = 1

# target used for correlation
target = 'SalePrice_Log'

# only columns with correlation above this threshold value  
# are used for the ML Regressors in Part 3
min_val_corr = 0.4

# switch for dropping columns that are similar to others already used and show a high correlation to these
drop_similar = 1
```
**Some useful functions**

```python
def get_best_score(grid):

    best_score = np.sqrt(-grid.best_score_)
    print(best_score)
    print(grid.best_params_)
    print(grid.best_estimator_)

    return best_score
```

```python
def print_cols_large_corr(df, nr_c, targ) :
    corr = df.corr()
    corr_abs = corr.abs()
    print (corr_abs.nlargest(nr_c, targ)[targ])
```

```python
def plot_corr_matrix(df, nr_c, targ) :

    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 
                fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=cols.values, xticklabels=cols.values
               )
    plt.show()
```

### Load Data

```python
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
```

## Exploratory Data Analysis

```python
print(df_train.shape)
print(df_test.shape)
```

> (1460, 81)
> (1459, 80)

```python
print(df_train.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None

df train has 81 columns (79 features + id and target SalePrice) and 1460 entries (number of rows or house sales)<br> 
df test has 80 columns (79 features + id) and 1459 entries<br>
There is lots of info that is probably related to the SalePrice like the area, the neighborhood, the condition and quality.<br>
Maybe other features are not so important for predicting the target, also there might be a strong correlation for some of the features (like GarageCars and GarageArea).<br>
For some columns many values are missing: only 7 values for Pool QC in df train and 3 in df test.<br>

```python
df_train.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>

```python
df_train.describe()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>

## Distribution of SalePrice


```python
sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
```

> Skewness: 1.882876
> Kurtosis: 6.536282

![png](/images/house-prices/notebook_12_1.png)

As we see, the target variable SalePrice is not normally distributed.<br>
This can reduce the performance of the ML regression models because some assume normal distribution,<br>
see [sklearn info on preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html)<br>
Therfore we make a log transformation, the resulting distribution looks much better.<br>

```python
df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice_Log']);
# skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice_Log'].skew())
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())
# dropping old column
df_train.drop('SalePrice', axis= 1, inplace=True)
```
> Skewness: 0.121335
> Kurtosis: 0.809532

![png](/images/house-prices/notebook_13_1.png)

### Numerical and Categorical Features

```python
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))
```

> Number of Numerical features:  38
> Number of Categorical features:  43

```python
print(df_train[numerical_feats].columns)
print(df_train[categorical_feats].columns)
```

    Index(['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold', 'SalePrice_Log'],
          dtype='object')
    Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
           'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
           'SaleType', 'SaleCondition'],
          dtype='object')

```python
df_train[numerical_feats].head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>...</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>12.247694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>...</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>12.109011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>12.317167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>11.849398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>...</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>12.429216</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>

```python
df_train[categorical_feats].head()
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
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>...</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>...</td>
      <td>Detchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>...</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>

### Features with missing values

```python
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

### Filling Missing values
For a few columns there is lots of NaN entries.  
However, reading the data description we find this is not missing data:  
For PoolQC, NaN is not missing data but means no pool, likewise for Fence, FireplaceQu etc.

```python
# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    df_train[col].fillna('None',inplace=True)
    df_test[col].fillna('None',inplace=True)
```

```python
# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)
```

```python
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SalePrice_Log</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Heating</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RoofMatl</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

**Checking missing values in train data ?**

```python
df_train.isnull().sum().sum()
```

> 0

**Checking missing values in test data ?**

```python
df_test.isnull().sum().sum()
```

> 0

### Log transform

Like the target variable, also some of the feature values are not normally distributed and it is therefore better to use log values in df_train and df_test. Checking for skewness and kurtosis:

```python
for col in numerical_feats:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(df_train[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(df_train[col].kurt())  
         )
```

    Id              Skewness: 00.00     Kurtosis: -01.20
    MSSubClass      Skewness: 01.41     Kurtosis: 001.58
    LotFrontage     Skewness: 02.38     Kurtosis: 021.85
    LotArea         Skewness: 12.21     Kurtosis: 203.24
    OverallQual     Skewness: 00.22     Kurtosis: 000.10
    OverallCond     Skewness: 00.69     Kurtosis: 001.11
    YearBuilt       Skewness: -0.61     Kurtosis: -00.44
    YearRemodAdd    Skewness: -0.50     Kurtosis: -01.27
    MasVnrArea      Skewness: 02.68     Kurtosis: 010.15
    BsmtFinSF1      Skewness: 01.69     Kurtosis: 011.12
    BsmtFinSF2      Skewness: 04.26     Kurtosis: 020.11
    BsmtUnfSF       Skewness: 00.92     Kurtosis: 000.47
    TotalBsmtSF     Skewness: 01.52     Kurtosis: 013.25
    1stFlrSF        Skewness: 01.38     Kurtosis: 005.75
    2ndFlrSF        Skewness: 00.81     Kurtosis: -00.55
    LowQualFinSF    Skewness: 09.01     Kurtosis: 083.23
    GrLivArea       Skewness: 01.37     Kurtosis: 004.90
    BsmtFullBath    Skewness: 00.60     Kurtosis: -00.84
    BsmtHalfBath    Skewness: 04.10     Kurtosis: 016.40
    FullBath        Skewness: 00.04     Kurtosis: -00.86
    HalfBath        Skewness: 00.68     Kurtosis: -01.08
    BedroomAbvGr    Skewness: 00.21     Kurtosis: 002.23
    KitchenAbvGr    Skewness: 04.49     Kurtosis: 021.53
    TotRmsAbvGrd    Skewness: 00.68     Kurtosis: 000.88
    Fireplaces      Skewness: 00.65     Kurtosis: -00.22
    GarageYrBlt     Skewness: -0.67     Kurtosis: -00.27
    GarageCars      Skewness: -0.34     Kurtosis: 000.22
    GarageArea      Skewness: 00.18     Kurtosis: 000.92
    WoodDeckSF      Skewness: 01.54     Kurtosis: 002.99
    OpenPorchSF     Skewness: 02.36     Kurtosis: 008.49
    EnclosedPorch   Skewness: 03.09     Kurtosis: 010.43
    3SsnPorch       Skewness: 10.30     Kurtosis: 123.66
    ScreenPorch     Skewness: 04.12     Kurtosis: 018.44
    PoolArea        Skewness: 14.83     Kurtosis: 223.27
    MiscVal         Skewness: 24.48     Kurtosis: 701.00
    MoSold          Skewness: 00.21     Kurtosis: -00.40
    YrSold          Skewness: 00.10     Kurtosis: -01.19
    SalePrice_Log   Skewness: 00.12     Kurtosis: 000.81
    

```python
sns.distplot(df_train['GrLivArea']);
#skewness and kurtosis
print("Skewness: %f" % df_train['GrLivArea'].skew())
print("Kurtosis: %f" % df_train['GrLivArea'].kurt())
```

> Skewness: 1.366560
> Kurtosis: 4.895121

![png](/images/house-prices/notebook_29_1.png)

```python
sns.distplot(df_train['LotArea']);
#skewness and kurtosis
print("Skewness: %f" % df_train['LotArea'].skew())
print("Kurtosis: %f" % df_train['LotArea'].kurt())
```

> Skewness: 12.207688
> Kurtosis: 203.243271

![png](/images/house-prices/notebook_30_1.png)

```python
for df in [df_train, df_test]:
    df['GrLivArea_Log'] = np.log(df['GrLivArea'])
    df.drop('GrLivArea', inplace= True, axis = 1)
    df['LotArea_Log'] = np.log(df['LotArea'])
    df.drop('LotArea', inplace= True, axis = 1)

numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
```

```python
sns.distplot(df_train['GrLivArea_Log']);
#skewness and kurtosis
print("Skewness: %f" % df_train['GrLivArea_Log'].skew())
print("Kurtosis: %f" % df_train['GrLivArea_Log'].kurt())
```

> Skewness: -0.006995
> Kurtosis: 0.282603

![png](/images/house-prices/notebook_32_1.png)

```python
sns.distplot(df_train['LotArea_Log']);
#skewness and kurtosis
print("Skewness: %f" % df_train['LotArea_Log'].skew())
print("Kurtosis: %f" % df_train['LotArea_Log'].kurt())
```

> Skewness: -0.137994
> Kurtosis: 4.713358

![png](/images/house-prices/notebook_33_1.png)

## Relation of features to target (SalePrice_log)

### Plots of relation to target for all numerical features

```python
nr_rows = 12
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(numerical_feats)
li_not_plot = ['Id', 'SalePrice', 'SalePrice_Log']
li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]


for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(df_train[li_plot_num_feats[i]], df_train[target], ax = axs[r][c])
            stp = stats.pearsonr(df_train[li_plot_num_feats[i]], df_train[target])
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)

plt.tight_layout()
plt.show()
```

![png](/images/house-prices/notebook_38_0.png)

**Conclusion from EDA on numerical columns:**

We see that for some features like 'OverallQual' there is a strong linear correlation (0.79) to the target.  
For other features like 'MSSubClass' the correlation is very weak.  
For this kernel I decided to use only those features for prediction that have a correlation larger than a threshold value to SalePrice.  
This threshold value can be choosen in the global settings : min_val_corr  

With the default threshold for min_val_corr = 0.4, these features are dropped in Part 2, Data Wrangling:  
'Id', 'MSSubClass', 'LotArea', 'OverallCond', 'BsmtFinSF2', 'BsmtUnfSF',  'LowQualFinSF',  'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',
'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'

We also see that the entries for some of the numerical columns are in fact categorical values.  
For example, the numbers for 'OverallQual' and 'MSSubClass' represent a certain group for that feature ( see data description txt)

**Outliers**

```python
df_train = df_train.drop(df_train[(df_train['OverallQual']==10) & (df_train['SalePrice_Log']<12.3)].index)
df_train = df_train.drop(df_train[(df_train['GrLivArea_Log']>8.3) & (df_train['SalePrice_Log']<12.5)].index)
```

**Find columns with strong correlation to target**  
Only those with r > min_val_corr are used in the ML Regressors in Part 3  
The value for min_val_corr can be chosen in global settings

```python
corr = df_train.corr()
corr_abs = corr.abs()

nr_num_cols = len(numerical_feats)
ser_corr = corr_abs.nlargest(nr_num_cols, target)[target]

cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)
cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)
```

### List of numerical features and their correlation coefficient to target

```python
print(ser_corr)
print("List of numerical features with r above min_val_corr :")
print(cols_abv_corr_limit)
print("List of numerical features with r below min_val_corr :")
print(cols_bel_corr_limit)
```

    SalePrice_Log    1.000000
    OverallQual      0.821404
    GrLivArea_Log    0.737427
    GarageCars       0.681033
    GarageArea       0.656128
    TotalBsmtSF      0.647563
    1stFlrSF         0.620500
    FullBath         0.595899
    YearBuilt        0.587043
    YearRemodAdd     0.565992
    TotRmsAbvGrd     0.537702
    GarageYrBlt      0.500842
    Fireplaces       0.491998
    MasVnrArea       0.433353
    LotArea_Log      0.402814
    BsmtFinSF1       0.392283
    LotFrontage      0.352432
    WoodDeckSF       0.334250
    OpenPorchSF      0.325215
    2ndFlrSF         0.319953
    HalfBath         0.314186
    BsmtFullBath     0.237099
    BsmtUnfSF        0.221892
    BedroomAbvGr     0.209036
    EnclosedPorch    0.149029
    KitchenAbvGr     0.147534
    ScreenPorch      0.121245
    PoolArea         0.074338
    MSSubClass       0.073969
    MoSold           0.057064
    3SsnPorch        0.054914
    LowQualFinSF     0.037951
    YrSold           0.037151
    OverallCond      0.036821
    MiscVal          0.020012
    Id               0.017774
    BsmtHalfBath     0.005124
    BsmtFinSF2       0.004863
    Name: SalePrice_Log, dtype: float64
    List of numerical features with r above min_val_corr :
    ['SalePrice_Log', 'OverallQual', 'GrLivArea_Log', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd', 'GarageYrBlt', 'Fireplaces', 'MasVnrArea', 'LotArea_Log']
    List of numerical features with r below min_val_corr :
    ['BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF', '2ndFlrSF', 'HalfBath', 'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 'EnclosedPorch', 'KitchenAbvGr', 'ScreenPorch', 'PoolArea', 'MSSubClass', 'MoSold', '3SsnPorch', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MiscVal', 'Id', 'BsmtHalfBath', 'BsmtFinSF2']

### List of categorical features and their unique values

```python
for catg in list(categorical_feats) :
    print(df_train[catg].value_counts())
    print('#'*50)
```

    RL         1149
    RM          218
    FV           65
    RH           16
    C (all)      10
    Name: MSZoning, dtype: int64
    ##################################################
    Pave    1452
    Grvl       6
    Name: Street, dtype: int64
    ##################################################
    None    1367
    Grvl      50
    Pave      41
    Name: Alley, dtype: int64
    ##################################################
    Reg    925
    IR1    483
    IR2     41
    IR3      9
    Name: LotShape, dtype: int64
    ##################################################
    Lvl    1311
    Bnk      61
    HLS      50
    Low      36
    Name: LandContour, dtype: int64
    ##################################################
    AllPub    1457
    NoSeWa       1
    Name: Utilities, dtype: int64
    ##################################################
    Inside     1051
    Corner      262
    CulDSac      94
    FR2          47
    FR3           4
    Name: LotConfig, dtype: int64
    ##################################################
    Gtl    1380
    Mod      65
    Sev      13
    Name: LandSlope, dtype: int64
    ##################################################
    NAmes      225
    CollgCr    150
    OldTown    113
    Edwards     98
    Somerst     86
    Gilbert     79
    NridgHt     77
    Sawyer      74
    NWAmes      73
    SawyerW     59
    BrkSide     58
    Crawfor     51
    Mitchel     49
    NoRidge     41
    Timber      38
    IDOTRR      37
    ClearCr     28
    SWISU       25
    StoneBr     25
    MeadowV     17
    Blmngtn     17
    BrDale      16
    Veenker     11
    NPkVill      9
    Blueste      2
    Name: Neighborhood, dtype: int64
    ##################################################
    Norm      1260
    Feedr       80
    Artery      48
    RRAn        26
    PosN        18
    RRAe        11
    PosA         8
    RRNn         5
    RRNe         2
    Name: Condition1, dtype: int64
    ##################################################
    Norm      1444
    Feedr        6
    Artery       2
    RRNn         2
    RRAe         1
    RRAn         1
    PosN         1
    PosA         1
    Name: Condition2, dtype: int64
    ##################################################
    1Fam      1218
    TwnhsE     114
    Duplex      52
    Twnhs       43
    2fmCon      31
    Name: BldgType, dtype: int64
    ##################################################
    1Story    726
    2Story    443
    1.5Fin    154
    SLvl       65
    SFoyer     37
    1.5Unf     14
    2.5Unf     11
    2.5Fin      8
    Name: HouseStyle, dtype: int64
    ##################################################
    Gable      1141
    Hip         284
    Flat         13
    Gambrel      11
    Mansard       7
    Shed          2
    Name: RoofStyle, dtype: int64
    ##################################################
    CompShg    1433
    Tar&Grv      11
    WdShngl       6
    WdShake       5
    Roll          1
    Membran       1
    Metal         1
    Name: RoofMatl, dtype: int64
    ##################################################
    VinylSd    515
    HdBoard    222
    MetalSd    220
    Wd Sdng    206
    Plywood    108
    CemntBd     60
    BrkFace     50
    WdShing     26
    Stucco      24
    AsbShng     20
    Stone        2
    BrkComm      2
    CBlock       1
    AsphShn      1
    ImStucc      1
    Name: Exterior1st, dtype: int64
    ##################################################
    VinylSd    504
    MetalSd    214
    HdBoard    207
    Wd Sdng    197
    Plywood    142
    CmentBd     59
    Wd Shng     38
    BrkFace     25
    Stucco      25
    AsbShng     20
    ImStucc     10
    Brk Cmn      7
    Stone        5
    AsphShn      3
    CBlock       1
    Other        1
    Name: Exterior2nd, dtype: int64
    ##################################################
    None       872
    BrkFace    445
    Stone      126
    BrkCmn      15
    Name: MasVnrType, dtype: int64
    ##################################################
    TA    906
    Gd    488
    Ex     50
    Fa     14
    Name: ExterQual, dtype: int64
    ##################################################
    TA    1280
    Gd     146
    Fa      28
    Ex       3
    Po       1
    Name: ExterCond, dtype: int64
    ##################################################
    PConc     645
    CBlock    634
    BrkTil    146
    Slab       24
    Stone       6
    Wood        3
    Name: Foundation, dtype: int64
    ##################################################
    TA      649
    Gd      618
    Ex      119
    None     37
    Fa       35
    Name: BsmtQual, dtype: int64
    ##################################################
    TA      1309
    Gd        65
    Fa        45
    None      37
    Po         2
    Name: BsmtCond, dtype: int64
    ##################################################
    No      953
    Av      221
    Gd      132
    Mn      114
    None     38
    Name: BsmtExposure, dtype: int64
    ##################################################
    Unf     430
    GLQ     416
    ALQ     220
    BLQ     148
    Rec     133
    LwQ      74
    None     37
    Name: BsmtFinType1, dtype: int64
    ##################################################
    Unf     1254
    Rec       54
    LwQ       46
    None      38
    BLQ       33
    ALQ       19
    GLQ       14
    Name: BsmtFinType2, dtype: int64
    ##################################################
    GasA     1426
    GasW       18
    Grav        7
    Wall        4
    OthW        2
    Floor       1
    Name: Heating, dtype: int64
    ##################################################
    Ex    739
    TA    428
    Gd    241
    Fa     49
    Po      1
    Name: HeatingQC, dtype: int64
    ##################################################
    Y    1363
    N      95
    Name: CentralAir, dtype: int64
    ##################################################
    SBrkr    1332
    FuseA      94
    FuseF      27
    FuseP       3
    Mix         1
    None        1
    Name: Electrical, dtype: int64
    ##################################################
    TA    735
    Gd    586
    Ex     98
    Fa     39
    Name: KitchenQual, dtype: int64
    ##################################################
    Typ     1358
    Min2      34
    Min1      31
    Mod       15
    Maj1      14
    Maj2       5
    Sev        1
    Name: Functional, dtype: int64
    ##################################################
    None    690
    Gd      378
    TA      313
    Fa       33
    Ex       24
    Po       20
    Name: FireplaceQu, dtype: int64
    ##################################################
    Attchd     869
    Detchd     387
    BuiltIn     87
    None        81
    Basment     19
    CarPort      9
    2Types       6
    Name: GarageType, dtype: int64
    ##################################################
    Unf     605
    RFn     422
    Fin     350
    None     81
    Name: GarageFinish, dtype: int64
    ##################################################
    TA      1309
    None      81
    Fa        48
    Gd        14
    Po         3
    Ex         3
    Name: GarageQual, dtype: int64
    ##################################################
    TA      1324
    None      81
    Fa        35
    Gd         9
    Po         7
    Ex         2
    Name: GarageCond, dtype: int64
    ##################################################
    Y    1338
    N      90
    P      30
    Name: PavedDrive, dtype: int64
    ##################################################
    None    1452
    Gd         2
    Fa         2
    Ex         2
    Name: PoolQC, dtype: int64
    ##################################################
    None     1177
    MnPrv     157
    GdPrv      59
    GdWo       54
    MnWw       11
    Name: Fence, dtype: int64
    ##################################################
    None    1404
    Shed      49
    Othr       2
    Gar2       2
    TenC       1
    Name: MiscFeature, dtype: int64
    ##################################################
    WD       1267
    New       120
    COD        43
    ConLD       9
    ConLw       5
    ConLI       5
    CWD         4
    Oth         3
    Con         2
    Name: SaleType, dtype: int64
    ##################################################
    Normal     1198
    Partial     123
    Abnorml     101
    Family       20
    Alloca       12
    AdjLand       4
    Name: SaleCondition, dtype: int64
    ##################################################

### Relation to SalePrice for all categorical features

```python
li_cat_feats = list(categorical_feats)
nr_rows = 15
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=target, data=df_train, ax = axs[r][c])

plt.tight_layout()
plt.show()
```

![png](/images/house-prices/notebook_46_0.png)

**Conclusion from EDA on categorical columns:**

For many of the categorical there is no strong relation to the target.  
However, for some fetaures it is easy to find a strong relation.  
From the figures above these are : 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType'
Also for the categorical features, I use only those that show a strong relation to SalePrice. 
So the other columns are dropped when creating the ML dataframes in Part 2 :  
 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition' 


```python
catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 
                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]
```

### Correlation matrix 1

**Features with largest correlation to SalePrice_Log**  
all numerical features with correlation coefficient above threshold 


```python
nr_feats = len(cols_abv_corr_limit)
```


```python
plot_corr_matrix(df_train, nr_feats, target)
```


![png](/images/house-prices/notebook_51_0.png)


**Of those features with the largest correlation to SalePrice, some also are correlated strongly to each other.**
**To avoid failures of the ML regression models due to multicollinearity, these are dropped in part 2.**
**This is optional and controlled by the switch drop_similar (global settings)**

## Data Wrangling

**Drop all columns with only small correlation to SalePrice**
**Transform Categorical to numerical**
**Handling columns with missing data**
**Log values**
**Drop all columns with strong correlation to similar features**

Numerical columns : drop similar and low correlation
Categorical columns : Transform  to numerical

### Dropping all columns with weak correlation to SalePrice

```python
id_test = df_test['Id']

to_drop_num  = cols_bel_corr_limit
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [df_train, df_test]:
    df.drop(cols_to_drop, inplace= True, axis = 1)
```

### Convert categorical columns to numerical

For those categorcial features where the EDA with boxplots seem to show a strong dependence of the SalePrice on the category, we transform the columns to numerical.
To investigate the relation of the categories to SalePrice in more detail, we make violinplots for these features 
Also, we look at the mean of SalePrice as function of category.

```python
catg_list = catg_strong_corr.copy()
catg_list.remove('Neighborhood')

for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=df_train)
    plt.show()
```

![png](/images/house-prices/notebook_59_0.png)

![png](/images/house-prices/notebook_59_1.png)

![png](/images/house-prices/notebook_59_2.png)

![png](/images/house-prices/notebook_59_3.png)

![png](/images/house-prices/notebook_59_4.png)

![png](/images/house-prices/notebook_59_5.png)

![png](/images/house-prices/notebook_59_6.png)

![png](/images/house-prices/notebook_59_7.png)

![png](/images/house-prices/notebook_59_8.png)

```python
fig, ax = plt.subplots()
fig.set_size_inches(16, 5)
sns.violinplot(x='Neighborhood', y=target, data=df_train, ax=ax)
plt.xticks(rotation=45)
plt.show()
```

![png](/images/house-prices/notebook_60_0.png)

```python
for catg in catg_list :
    g = df_train.groupby(catg)[target].mean()
    print(g)
```

    MSZoning
    C (all)    11.118259
    FV         12.246616
    RH         11.749840
    RL         12.085939
    RM         11.692893
    Name: SalePrice_Log, dtype: float64
    Condition2
    Artery    11.570036
    Feedr     11.670631
    Norm      12.025925
    PosA      12.691580
    PosN      12.860999
    RRAe      12.154779
    RRAn      11.827043
    RRNn      11.435329
    Name: SalePrice_Log, dtype: float64
    MasVnrType
    BrkCmn     11.853239
    BrkFace    12.163630
    None       11.896884
    Stone      12.431016
    Name: SalePrice_Log, dtype: float64
    ExterQual
    Ex    12.792412
    Fa    11.304541
    Gd    12.311282
    TA    11.837985
    Name: SalePrice_Log, dtype: float64
    BsmtQual
    Ex      12.650235
    Fa      11.617600
    Gd      12.179882
    None    11.529680
    TA      11.810855
    Name: SalePrice_Log, dtype: float64
    CentralAir
    N    11.491858
    Y    12.061099
    Name: SalePrice_Log, dtype: float64
    Electrical
    FuseA    11.660315
    FuseF    11.539624
    FuseP    11.446808
    Mix      11.112448
    None     12.028739
    SBrkr    12.061474
    Name: SalePrice_Log, dtype: float64
    KitchenQual
    Ex    12.645425
    Fa    11.504581
    Gd    12.222337
    TA    11.810592
    Name: SalePrice_Log, dtype: float64
    SaleType
    COD      11.827437
    CWD      12.198344
    Con      12.483911
    ConLD    11.773000
    ConLI    12.044878
    ConLw    11.769706
    New      12.466114
    Oth      11.675295
    WD       11.991061
    Name: SalePrice_Log, dtype: float64


```python
# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 

# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']
```

```python
for df in [df_train, df_test]:

    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3

    df['NbHd_num'] = 1
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3

    df['Cond2_num'] = 1
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3

    df['Mas_num'] = 1
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2

    df['ExtQ_num'] = 1
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4
```

### Checking correlation to SalePrice for the new numerical columns


```python
new_col_num = ['MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']

nr_rows = 4
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(new_col_num):
            sns.regplot(df_train[new_col_num[i]], df_train[target], ax = axs[r][c])
            stp = stats.pearsonr(df_train[new_col_num[i]], df_train[target])
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)

plt.tight_layout()
plt.show()  
```

![png](/images/house-prices/notebook_65_0.png)

There are few columns with quite large correlation to SalePrice (NbHd_num, ExtQ_num, BsQ_num, KiQ_num).  
These will probably be useful for optimal performance of the Regressors in part 3.

**Dropping the converted categorical columns and the new numerical columns with weak correlation**
**columns and correlation before dropping**

```python
catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

corr1 = df_train.corr()
corr_abs_1 = corr1.abs()

nr_all_cols = len(df_train)
ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, target)[target]

print(ser_corr_1)
cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)


for df in [df_train, df_test] :
    df.drop(catg_cols_to_drop, inplace= True, axis = 1)
    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1)    
```

    SalePrice_Log    1.000000
    OverallQual      0.821404
    GrLivArea_Log    0.737427
    NbHd_num         0.696962
    ExtQ_num         0.682225
    GarageCars       0.681033
    KiQ_num          0.669989
    BsQ_num          0.661286
    GarageArea       0.656128
    TotalBsmtSF      0.647563
    1stFlrSF         0.620500
    FullBath         0.595899
    YearBuilt        0.587043
    YearRemodAdd     0.565992
    TotRmsAbvGrd     0.537702
    GarageYrBlt      0.500842
    Fireplaces       0.491998
    MasVnrArea       0.433353
    MSZ_num          0.409423
    LotArea_Log      0.402814
    CA_num           0.351598
    SlTy_num         0.337469
    Mas_num          0.313280
    Elc_num          0.304857
    Cond2_num        0.107610
    Name: SalePrice_Log, dtype: float64


**columns and correlation after dropping**

```python
corr2 = df_train.corr()
corr_abs_2 = corr2.abs()

nr_all_cols = len(df_train)
ser_corr_2 = corr_abs_2.nlargest(nr_all_cols, target)[target]

print(ser_corr_2)
```

    SalePrice_Log    1.000000
    OverallQual      0.821404
    GrLivArea_Log    0.737427
    NbHd_num         0.696962
    ExtQ_num         0.682225
    GarageCars       0.681033
    KiQ_num          0.669989
    BsQ_num          0.661286
    GarageArea       0.656128
    TotalBsmtSF      0.647563
    1stFlrSF         0.620500
    FullBath         0.595899
    YearBuilt        0.587043
    YearRemodAdd     0.565992
    TotRmsAbvGrd     0.537702
    GarageYrBlt      0.500842
    Fireplaces       0.491998
    MasVnrArea       0.433353
    MSZ_num          0.409423
    LotArea_Log      0.402814
    Name: SalePrice_Log, dtype: float64
    

**new dataframes**

```python
df_train.head()
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
      <th>MSZoning</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>FullBath</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>...</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>SalePrice_Log</th>
      <th>GrLivArea_Log</th>
      <th>LotArea_Log</th>
      <th>MSZ_num</th>
      <th>NbHd_num</th>
      <th>ExtQ_num</th>
      <th>BsQ_num</th>
      <th>KiQ_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>7</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>856</td>
      <td>856</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>548</td>
      <td>12.247694</td>
      <td>7.444249</td>
      <td>9.041922</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>1262</td>
      <td>1262</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>460</td>
      <td>12.109011</td>
      <td>7.140453</td>
      <td>9.169518</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>7</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>920</td>
      <td>920</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>608</td>
      <td>12.317167</td>
      <td>7.487734</td>
      <td>9.328123</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>7</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>756</td>
      <td>961</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>642</td>
      <td>11.849398</td>
      <td>7.448334</td>
      <td>9.164296</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>8</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>1145</td>
      <td>1145</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>836</td>
      <td>12.429216</td>
      <td>7.695303</td>
      <td>9.565214</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

```python
df_test.head()
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
      <th>MSZoning</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>FullBath</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GrLivArea_Log</th>
      <th>LotArea_Log</th>
      <th>MSZ_num</th>
      <th>NbHd_num</th>
      <th>ExtQ_num</th>
      <th>BsQ_num</th>
      <th>KiQ_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RH</td>
      <td>5</td>
      <td>1961</td>
      <td>1961</td>
      <td>0.0</td>
      <td>882.0</td>
      <td>896</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1961.0</td>
      <td>1.0</td>
      <td>730.0</td>
      <td>6.797940</td>
      <td>9.360655</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>108.0</td>
      <td>1329.0</td>
      <td>1329</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1958.0</td>
      <td>1.0</td>
      <td>312.0</td>
      <td>7.192182</td>
      <td>9.565704</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>0.0</td>
      <td>928.0</td>
      <td>928</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>1997.0</td>
      <td>2.0</td>
      <td>482.0</td>
      <td>7.395722</td>
      <td>9.534595</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>6</td>
      <td>1998</td>
      <td>1998</td>
      <td>20.0</td>
      <td>926.0</td>
      <td>926</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>1998.0</td>
      <td>2.0</td>
      <td>470.0</td>
      <td>7.380256</td>
      <td>9.208138</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>8</td>
      <td>1992</td>
      <td>1992</td>
      <td>0.0</td>
      <td>1280.0</td>
      <td>1280</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1992.0</td>
      <td>2.0</td>
      <td>506.0</td>
      <td>7.154615</td>
      <td>8.518193</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

**List of all features with strong correlation to SalePrice_Log**  
after dropping all coumns with weak correlation

```python
corr = df_train.corr()
corr_abs = corr.abs()

nr_all_cols = len(df_train)
print (corr_abs.nlargest(nr_all_cols, target)[target])
```

    SalePrice_Log    1.000000
    OverallQual      0.821404
    GrLivArea_Log    0.737427
    NbHd_num         0.696962
    ExtQ_num         0.682225
    GarageCars       0.681033
    KiQ_num          0.669989
    BsQ_num          0.661286
    GarageArea       0.656128
    TotalBsmtSF      0.647563
    1stFlrSF         0.620500
    FullBath         0.595899
    YearBuilt        0.587043
    YearRemodAdd     0.565992
    TotRmsAbvGrd     0.537702
    GarageYrBlt      0.500842
    Fireplaces       0.491998
    MasVnrArea       0.433353
    MSZ_num          0.409423
    LotArea_Log      0.402814
    Name: SalePrice_Log, dtype: float64


### Correlation Matrix 2 : All features with strong correlation to SalePrice

```python
nr_feats=len(df_train.columns)
plot_corr_matrix(df_train, nr_feats, target)
```

![png](/images/house-prices/notebook_78_0.png)

**Check for Multicollinearity**

Strong correlation of these features to other, similar features:

'GrLivArea_Log' and 'TotRmsAbvGrd'

'GarageCars' and 'GarageArea'

'TotalBsmtSF' and '1stFlrSF'

'YearBuilt' and 'GarageYrBlt'

**Of those features we drop the one that has smaller correlation coeffiecient to Target.**

```python
cols = corr_abs.nlargest(nr_all_cols, target)[target].index
cols = list(cols)

if drop_similar == 1 :
    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'] :
        if col in cols:
            cols.remove(col)
```

```python
cols = list(cols)
print(cols)
```

    ['SalePrice_Log', 'OverallQual', 'GrLivArea_Log', 'NbHd_num', 'ExtQ_num', 'GarageCars', 'KiQ_num', 'BsQ_num', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'MasVnrArea', 'MSZ_num', 'LotArea_Log']


**List of features used for the Regressors in Part 3**


```python
feats = cols.copy()
feats.remove('SalePrice_Log')

print(feats)
```

    ['OverallQual', 'GrLivArea_Log', 'NbHd_num', 'ExtQ_num', 'GarageCars', 'KiQ_num', 'BsQ_num', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'MasVnrArea', 'MSZ_num', 'LotArea_Log']


```python
df_train_ml = df_train[feats].copy()
df_test_ml  = df_test[feats].copy()

y = df_train[target]
```

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df_train_ml_sc = sc.fit_transform(df_train_ml)
df_test_ml_sc = sc.transform(df_test_ml)
```

```python
df_train_ml_sc = pd.DataFrame(df_train_ml_sc)
df_train_ml_sc.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.658506</td>
      <td>0.539624</td>
      <td>0.658963</td>
      <td>1.061109</td>
      <td>0.313159</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>-0.473766</td>
      <td>0.793546</td>
      <td>1.052959</td>
      <td>0.880362</td>
      <td>-0.952231</td>
      <td>0.521228</td>
      <td>0.438861</td>
      <td>-0.129585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.068293</td>
      <td>-0.380198</td>
      <td>0.658963</td>
      <td>-0.689001</td>
      <td>0.313159</td>
      <td>-0.770150</td>
      <td>0.648281</td>
      <td>0.504925</td>
      <td>0.793546</td>
      <td>0.158428</td>
      <td>-0.428115</td>
      <td>0.605965</td>
      <td>-0.574433</td>
      <td>0.438861</td>
      <td>0.118848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.658506</td>
      <td>0.671287</td>
      <td>0.658963</td>
      <td>1.061109</td>
      <td>0.313159</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>-0.319490</td>
      <td>0.793546</td>
      <td>0.986698</td>
      <td>0.831900</td>
      <td>0.605965</td>
      <td>0.331164</td>
      <td>0.438861</td>
      <td>0.427653</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.658506</td>
      <td>0.551993</td>
      <td>0.658963</td>
      <td>-0.689001</td>
      <td>1.652119</td>
      <td>0.741127</td>
      <td>-0.921808</td>
      <td>-0.714823</td>
      <td>-1.025620</td>
      <td>-1.862551</td>
      <td>-0.718888</td>
      <td>0.605965</td>
      <td>-0.574433</td>
      <td>0.438861</td>
      <td>0.108680</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.385305</td>
      <td>1.299759</td>
      <td>2.162512</td>
      <td>1.061109</td>
      <td>1.652119</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>0.222888</td>
      <td>0.793546</td>
      <td>0.953567</td>
      <td>0.734975</td>
      <td>0.605965</td>
      <td>1.382104</td>
      <td>0.438861</td>
      <td>0.889271</td>
    </tr>
  </tbody>
</table>
</div>


**Creating Datasets for ML algorithms**

```python
X = df_train_ml.copy()
y = df_train[target]
X_test = df_test_ml.copy()

X_sc = df_train_ml_sc.copy()
y_sc = df_train[target]
X_test_sc = df_test_ml_sc.copy()

X.info()
X_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1458 entries, 0 to 1459
    Data columns (total 15 columns):
    OverallQual      1458 non-null int64
    GrLivArea_Log    1458 non-null float64
    NbHd_num         1458 non-null int64
    ExtQ_num         1458 non-null int64
    GarageCars       1458 non-null int64
    KiQ_num          1458 non-null int64
    BsQ_num          1458 non-null int64
    TotalBsmtSF      1458 non-null int64
    FullBath         1458 non-null int64
    YearBuilt        1458 non-null int64
    YearRemodAdd     1458 non-null int64
    Fireplaces       1458 non-null int64
    MasVnrArea       1458 non-null float64
    MSZ_num          1458 non-null int64
    LotArea_Log      1458 non-null float64
    dtypes: float64(3), int64(12)
    memory usage: 182.2 KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 15 columns):
    OverallQual      1459 non-null int64
    GrLivArea_Log    1459 non-null float64
    NbHd_num         1459 non-null int64
    ExtQ_num         1459 non-null int64
    GarageCars       1459 non-null float64
    KiQ_num          1459 non-null int64
    BsQ_num          1459 non-null int64
    TotalBsmtSF      1459 non-null float64
    FullBath         1459 non-null int64
    YearBuilt        1459 non-null int64
    YearRemodAdd     1459 non-null int64
    Fireplaces       1459 non-null int64
    MasVnrArea       1459 non-null float64
    MSZ_num          1459 non-null int64
    LotArea_Log      1459 non-null float64
    dtypes: float64(5), int64(10)
    memory usage: 171.1 KB

```python
X.head()
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
      <th>OverallQual</th>
      <th>GrLivArea_Log</th>
      <th>NbHd_num</th>
      <th>ExtQ_num</th>
      <th>GarageCars</th>
      <th>KiQ_num</th>
      <th>BsQ_num</th>
      <th>TotalBsmtSF</th>
      <th>FullBath</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>Fireplaces</th>
      <th>MasVnrArea</th>
      <th>MSZ_num</th>
      <th>LotArea_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>7.444249</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>856</td>
      <td>2</td>
      <td>2003</td>
      <td>2003</td>
      <td>0</td>
      <td>196.0</td>
      <td>3</td>
      <td>9.041922</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>7.140453</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1262</td>
      <td>2</td>
      <td>1976</td>
      <td>1976</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>9.169518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>7.487734</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>920</td>
      <td>2</td>
      <td>2001</td>
      <td>2002</td>
      <td>1</td>
      <td>162.0</td>
      <td>3</td>
      <td>9.328123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>7.448334</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>756</td>
      <td>1</td>
      <td>1915</td>
      <td>1970</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>9.164296</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>7.695303</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>1145</td>
      <td>2</td>
      <td>2000</td>
      <td>2000</td>
      <td>1</td>
      <td>350.0</td>
      <td>3</td>
      <td>9.565214</td>
    </tr>
  </tbody>
</table>
</div>

```python
X_sc.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.658506</td>
      <td>0.539624</td>
      <td>0.658963</td>
      <td>1.061109</td>
      <td>0.313159</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>-0.473766</td>
      <td>0.793546</td>
      <td>1.052959</td>
      <td>0.880362</td>
      <td>-0.952231</td>
      <td>0.521228</td>
      <td>0.438861</td>
      <td>-0.129585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.068293</td>
      <td>-0.380198</td>
      <td>0.658963</td>
      <td>-0.689001</td>
      <td>0.313159</td>
      <td>-0.770150</td>
      <td>0.648281</td>
      <td>0.504925</td>
      <td>0.793546</td>
      <td>0.158428</td>
      <td>-0.428115</td>
      <td>0.605965</td>
      <td>-0.574433</td>
      <td>0.438861</td>
      <td>0.118848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.658506</td>
      <td>0.671287</td>
      <td>0.658963</td>
      <td>1.061109</td>
      <td>0.313159</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>-0.319490</td>
      <td>0.793546</td>
      <td>0.986698</td>
      <td>0.831900</td>
      <td>0.605965</td>
      <td>0.331164</td>
      <td>0.438861</td>
      <td>0.427653</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.658506</td>
      <td>0.551993</td>
      <td>0.658963</td>
      <td>-0.689001</td>
      <td>1.652119</td>
      <td>0.741127</td>
      <td>-0.921808</td>
      <td>-0.714823</td>
      <td>-1.025620</td>
      <td>-1.862551</td>
      <td>-0.718888</td>
      <td>0.605965</td>
      <td>-0.574433</td>
      <td>0.438861</td>
      <td>0.108680</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.385305</td>
      <td>1.299759</td>
      <td>2.162512</td>
      <td>1.061109</td>
      <td>1.652119</td>
      <td>0.741127</td>
      <td>0.648281</td>
      <td>0.222888</td>
      <td>0.793546</td>
      <td>0.953567</td>
      <td>0.734975</td>
      <td>0.605965</td>
      <td>1.382104</td>
      <td>0.438861</td>
      <td>0.889271</td>
    </tr>
  </tbody>
</table>
</div>

# Scikit-learn basic regression models and comparison of results

**Test simple sklearn models and compare by metrics**

**We test the following Regressors from scikit-learn:**  
Linear Regression
Stochastic Gradient Descent  
DecisionTreeRegressor  
RandomForestRegressor  
SVR

**Model tuning and selection with GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV
score_calc = 'neg_mean_squared_error'
```

### Linear Regression


```python
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)
grid_linear.fit(X, y)

sc_linear = get_best_score(grid_linear)
```

> Fitting 5 folds for each of 8 candidates, totalling 40 fits
> 0.1362343506167217

```python
linreg_sc = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_linear_sc = GridSearchCV(linreg_sc, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)
grid_linear_sc.fit(X_sc, y)

sc_linear_sc = get_best_score(grid_linear_sc)
```

> Fitting 5 folds for each of 8 candidates, totalling 40 fits
> 0.13623435061672204

```python
linregr_all = LinearRegression()
linregr_all.fit(X, y)
pred_linreg_all = linregr_all.predict(X_test)
pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()
```

```python
sub_linreg = pd.DataFrame()
sub_linreg['Id'] = id_test
sub_linreg['SalePrice'] = pred_linreg_all
```

### Stochastic Gradient Descent Regressor

Linear model fitted by minimizing a regularized empirical loss with SGD. SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net).

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()
parameters = {'max_iter' :[10000], 'alpha':[1e-05], 'epsilon':[1e-02], 'fit_intercept' : [True]  }
grid_sgd = GridSearchCV(sgd, parameters, cv=nr_cv, verbose=1, scoring = score_calc)
grid_sgd.fit(X_sc, y_sc)

sc_sgd = get_best_score(grid_sgd)

pred_sgd = grid_sgd.predict(X_test_sc)
```

> Fitting 5 folds for each of 1 candidates, totalling 5 fits
> 0.13740875157743374

### Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,
               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],
                'presort': [False,True] , 'random_state': [5] }

grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)
grid_dtree.fit(X, y)

sc_dtree = get_best_score(grid_dtree)

pred_dtree = grid_dtree.predict(X_test)
```

> Fitting 5 folds for each of 480 candidates, totalling 2400 fits
> 0.18299182249476628

```python
dtree_pred = grid_dtree.predict(X_test)
sub_dtree = pd.DataFrame()
sub_dtree['Id'] = id_test
sub_dtree['SalePrice'] = dtree_pred
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)
grid_rf.fit(X, y)

sc_rf = get_best_score(grid_rf)
```

> Fitting 5 folds for each of 8 candidates, totalling 40 fits
> 0.1465978663015509

```python
pred_rf = grid_rf.predict(X_test)

sub_rf = pd.DataFrame()
sub_rf['Id'] = id_test
sub_rf['SalePrice'] = pred_rf 

if use_logvals == 1:
    sub_rf['SalePrice'] = np.exp(sub_rf['SalePrice']) 

sub_rf.to_csv('rf.csv',index=False)
```

```python
sub_rf.head(10)
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
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>121404.964212</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>130824.396900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>183372.764889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>183944.210608</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>198272.459357</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1466</td>
      <td>182039.290710</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1467</td>
      <td>164671.143500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1468</td>
      <td>175829.325089</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1469</td>
      <td>180844.256443</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1470</td>
      <td>121240.046457</td>
    </tr>
  </tbody>
</table>
</div>


### KNN Regressor

```python
from sklearn.neighbors import KNeighborsRegressor

param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,
              'weights' : ['uniform','distance'] ,
              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}

grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)
grid_knn.fit(X_sc, y_sc)

sc_knn = get_best_score(grid_knn)
```

> Fitting 5 folds for each of 42 candidates, totalling 210 fits
> 0.15615217437688825

```python
pred_knn = grid_knn.predict(X_test_sc)

sub_knn = pd.DataFrame()
sub_knn['Id'] = id_test
sub_knn['SalePrice'] = pred_knn

if use_logvals == 1:
    sub_knn['SalePrice'] = np.exp(sub_knn['SalePrice']) 

sub_knn.to_csv('knn.csv',index=False)
```

### Comparison plot: RMSE of all models

```python
list_scores = [sc_linear, sc_sgd, sc_dtree, sc_rf, sc_knn]
list_regressors = ['Linear','SGD','DTr','RF','KNN']
```

```python
fig, ax = plt.subplots()
fig.set_size_inches(10,7)
sns.barplot(x=list_regressors, y=list_scores, ax=ax)
plt.ylabel('RMSE')
plt.show()
```

![png](/images/house-prices/notebook_115_0.png)

### Correlation of model results

```python
predictions = {'Linear': pred_linreg_all, 'SGD': pred_sgd, 'DTr': pred_dtree, 'RF': pred_rf,
               'KNN': pred_knn}
df_predictions = pd.DataFrame(data=predictions) 
df_predictions.corr()
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
      <th>Linear</th>
      <th>SGD</th>
      <th>DTr</th>
      <th>RF</th>
      <th>KNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear</th>
      <td>1.000000</td>
      <td>0.999844</td>
      <td>0.937594</td>
      <td>0.979555</td>
      <td>0.964423</td>
    </tr>
    <tr>
      <th>SGD</th>
      <td>0.999844</td>
      <td>1.000000</td>
      <td>0.937412</td>
      <td>0.979134</td>
      <td>0.963465</td>
    </tr>
    <tr>
      <th>DTr</th>
      <td>0.937594</td>
      <td>0.937412</td>
      <td>1.000000</td>
      <td>0.961966</td>
      <td>0.922761</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>0.979555</td>
      <td>0.979134</td>
      <td>0.961966</td>
      <td>1.000000</td>
      <td>0.962788</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.964423</td>
      <td>0.963465</td>
      <td>0.922761</td>
      <td>0.962788</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
plt.figure(figsize=(5, 5))
sns.set(font_scale=1.25)
sns.heatmap(df_predictions.corr(), linewidths=1.5, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=df_predictions.columns , xticklabels=df_predictions.columns
            )
plt.show()
```

![png](/images/house-prices/notebook_118_0.png)

Only for Random Forest and Decision Tree, the results are less correlated with the other Regressors.

## Submission

```python
sub_mean = pd.DataFrame()
sub_mean['Id'] = id_test
sub_mean['SalePrice'] = np.round( ( pred_rf + pred_sgd) / 2.0 ) 
sub_mean['SalePrice'] = sub_mean['SalePrice'].astype(float)
sub_mean.to_csv('mean.csv',index=False)
```