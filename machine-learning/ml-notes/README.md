---
description: Notes for Machine Learning from small topics to topic
---

# ML Notes

**`Statistical Jargon for Machine Learning and Exploratory data analysis`**

## Measures of Central Tendency

### `Mean`

Mean is the average of the data, and the important property of the mean is that it includes all values in the data. Thus it can be affected by the outliers. Presence of outliers can give unreliable results. &#x20;

```
data.mean()
```

#### Trimmed Mean

We know that the mean value is affected by the extreme observations, so let us obtain the trimmed mean.

```
import scipy
import numpy as np
import pandas as pd
from scipy import stats

num = pd.DataFrame(data.select_dtypes(include = np.number))
print(scipy.stats.trim_mean(num, proportiontocut = 0.20))
```

### `Median`

The median value is the middlemost value of the data, it divides the data into two equal halves. Thus, it is a positional value, Unlike mean, the median value is not affected by extreme values.The median value is at times used to impute the missing values in the data

```
data.median()
```

### `Mode`

Mode of the data is the value which has the highest frequency. It is a measure that can be used for categorical variables. Example: gender

```
data.select_dtypes(include=object).mode()
```

### `Partition Values (Quantiles)`

Partition values are the values which divide the data into equal parts, We have seen that the median divides the data into two equal halves. Similarly, in order to divide the data into four parts we use quartiles, for ten parts we use deciles.&#x20;

```
# the first quartile
data.quantile(0.25)

# the second quartile
data.quantile(0.5)

# the third quartile
data.quantile(0.75)
```

## Measure of Dispersion

The measure of dispersion indicates the variability within a set of measures, less variability implies more data is closer to the central observations. More variability implies more data is spread further away from the center. The most popular variability measures are the range, interquartile range (IQR), variance, and standard deviation.

### `Range`

The range represents the difference between the largest and the smallest points in the data, In spite of being very easy to define and calculate, it is not a reliable measure. It is highly affected by extreme values.

```
num = pd.DataFrame(data.select_dtypes(include = np.number))
print(num.max() - num.min())
```

### `Variance`

Variance measure the dispersion of the data from the mean, It is an average of the sum of squares of difference between an observation and the mean. Thus the variance is always positive, The variance is based on all the observations.

```
print(round(data.var(),2))
```

Variance is the sum of squares of the difference, so the unit of variance is the squared unit of the observations. Hence it is viable to take the square root of the variance to get an answer back in the original unit of measurement

### `Standard Deviation`

Standard deviation is the positive square root of the variance, It has the same unit as that of the observations. . Like variance, it also measures the variability of the observations from the mean.&#x20;

```
data.std()
```

### `Coefficient of Variation`

The coefficient of variation is a statistical measure of dispersion of data points around the mean. It is a unit free measure and is always expressed in percentage, We compare the coefficient of variation in order to decide which of the two sets of observations has more spread.

```
from scipy.stats import variation  
 
list(scipy.stats.variation(num,nan_policy='omit'))
```

### &#x20;`Interquartile Range (IQR)`

The interquartile range is another useful measure of spread based on the quartiles, It is the difference between the third quartile and the first quartile. The IQR gives the range of middle 50% of the data. It also helps in identifying the outliers.

```
print(num.quantile(0.75)-num.quantile(0.25))
```

## Distribution of the data

The distribution is a summary of the frequency of values taken by a variable. The distribution of the data gives information on the shape and spread of the data. On plotting the histogram or a frequency curve for a variable, we are actually looking at how the data is distributed over its range.

```
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
data.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)
plt.show()
```

## Shape of the data

### `Skewness`

Skewness helps us to study the shape of the data. It represents how much a distribution differs from a normal distribution, either to the left or to the right. The value of the skewness can be either positive, negative or zero.

```
data.skew()
```

| Value of Skewness | Interpretation                 | ?                    |
| ----------------- | ------------------------------ | -------------------- |
| S < 0             | Negatively Skewed Distribution | mean < median < mode |
| S = 0             | Distribution is not Skewed     | mean = median = mode |
| S > 0             | Positively Skewed Distribution | mean > median > mode |

### `Kurtosis`

Kurtosis measures the peakedness of the distribution. In other words, kurtosis is a statistical measure that defines how the tails of the distribution differ from the normal distribution. Kurtosis identifies whether the tails of a given distribution contain extreme values.

```
data.kurt()
```

## Z Score

Another way to detect outliers are using the z-score. Z-score of a value is the difference between that value and the mean, divided by the standard deviation. Z-score of 0 indicates the value is the same as the mean. Z-score is positive or negative means the value is above or below the mean and by how many standard deviations. Z-score greater than 3 or less than -3, indicates an outlier value.&#x20;

```
outlier = scipy.stats.zscore(data.column_name)

outlier[(outlier < -3) | (outlier > 3)]
```

## Correlation **and Covariance**

**Correlation** shows whether pairs of variables are related to each other. If there is correlation, it shows how strong the correlation is. It takes values between -1 to +1, where values close to +1 represents strong positive correlation while values close to -1 represents strong negative correlation.&#x20;

```
data.corr()

import seaborn as sns
sns.heatmap(data.corr(), annot=True)
```

**Covariance** is the relationship between a pair of random variables where change in one variable causes change in another variable. It can take any value between -infinity to +infinity, where the negative value represents the negative relationship whereas a positive value represents the positive relationship.

```
data.cov()
```





## **Exploratory data analysis (EDA)**

### Univariate Analysis

Uni(single) variate analysis is the simplest form of statistical analysis.The main purpose of this type of analysis is to understand each variable in the data using various statistical and visualization techniques. It helps to study the pattern in each variable. The Univariate analysis contains various techniques for numerical as well as a categorical variable.&#x20;

* Numerical Variable  -> Summary Statistic, Histogram, Density, Box plot
* Categorical Variable ->  Summary Statistic, frequency table, Bar Plot

### Bivariate Analysis

It helps us to understand relationship between two variables.&#x20;

* Quantitative - Quantitative  -> Line, Scatter plot, heatmap
* Quantitative - Categorical -> Bar, Kde, Box, Violin plot
* Categorical - Categorical -> Crosstab, Stacked bar chart

### Multivariate Analysis

* Pair plot, grouped box plot, heatmap, Scatter plot.

**Visualization:**

This file contains graph/plots for all the above Analysis.

{% file src="../../.gitbook/assets/Visualization_Plots.pdf" %}



&#x20;                                                  **                                                   `Data PreProcessing`**

So before moving in, we check datatype of variables and if some column are not correct datatype, we shall fix them first.  We can check by following code.&#x20;

```
data.dtypes()


data['column X'] = data['column X'].astype('int')
```

## Missing Values

* Standard Missing Values:  Can be detected using Python NaN or Space&#x20;
* Non-Standard Missing Values: Can't be detected i.e ( ?, - , NA)

```
# returns Standard Missing Values and % of missing data

missing_value = pd.DataFrame({
    'Missing Value': data.isnull().sum(),
    'Percentage': (data.isnull().sum() / len(data))*100
})

missing_value.sort_values(by='Percentage', ascending=False)
```

To handle non-standard missing values, first we have to identified them and then replace with **nan** and later impute them as standard missing values.

```
data.replace('?', np.nan, inplace=True)
```

### **Dealing Missing Values**

![](<../../.gitbook/assets/image (12).png>)

If variable is not very important predictor for target variable and has around 60-70% of missing values, we should consider dropping the variable/column.                                  In categorical variable, the missing value can be replaced by the most frequent class of the variable.&#x20;

```
data.dropna(inplace=True) # drop all the rows with missing values
```

But there can be loss of huge data due to above action. So for that we can impute missing values using mean, mode, median. &#x20;

For numeric variable, missing values can be replaced by the mean, and if there is presence of outliers in the data then median can be used.

```
# for categorical variable

import numpy as np
data.column.replace(np.nan, data.column.mode()[0],inplace=True )


data.column.fillna(np.mean, inplace=True)
```

## Non-Numeric Data

There will be numerical and categorical data in the dataset, but most of the algorithms are designed to work with the numerical data only. So to convert categorical data into numeric we use data encoding.&#x20;

### One-hot encoding&#x20;

For categorical feature that has n class, for that n dummy variables are created, Each category/class converted into one column with values 0 and 1, depending on the absence and presence of the category in the observation.

```
# using pandas
data = pd.get_dummies(columns=['column1'], drop_first=True, data=dataframe)


# using sklearn
from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder()

df_encode = pd.DataFrame(encode.fit_transform(data[['column']]).toarray(), columns = ['class1', 'class2', 
                                                                                         'class3'])
data= pd.concat([data, df_encode], axis=1)
```

### Label encoding&#x20;

The Label encoder considers the levels in a categorical variable by alphabetic order for encoding. And for n class labels will be 0 to n-1.&#x20;

```
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

data['New_Label'] = labelencoder.fit_transform(data.Label)
```

### Ordinal encoding&#x20;

The ordinal encoder from sklearn encodes the categorical variable with values between 0 and (n-1). We can pass the order of the categories to preserve the order present in the ordinal categorical variable. For example:  ‘Average’ is labeled as 0, ‘Good’ as 1, and ‘Excellent’ as 2.

```
# ordinal encoding for feature that has size

from sklearn.preprocessing import OrdinalEncoder
orderencoding = OrdinalEncoder(categories = [["Small", "Medium", "High"]])

data['new_column'] = orderencoding.fit_transform(data['column'].values.reshape(-1,1))
```

### Frequency encoding&#x20;

If a categorical variable contains too many levels, then using one-hot encoding will increase the number of features drastically. Frequency encoding replaces each label of the categorical variable by the percentage of observations within that category.&#x20;

```
encoding = data.groupby('column').size()

encoding = encoding/len(data)

data['new_column'] = data.column.map(encoding)*100
```

The method fails when two or more categories have the same number of observations. In such a scenario, different labels will have the same frequency.&#x20;

### Target encoding

Target encoding is a technique used in a classification problem to convert a categorical variable to a numeric variable. Encodes each level of categorical variable with its corresponding target mean. The dimensionality of the data remains the same as that of not encoded data. It is also known as **mean encoding.**

```
--
```

## Outliers&#x20;

Discovery and Treatment

### Using Z-score

Z-scores can quantify the unusualness of observation when data follow the Gaussian/normal distribution and we check if data is normal or not using shaprio test and if p\_value is less than 0.05 that mean data is normally distributed.                  Z-score of a value is the difference between that value and the mean, divided by the standard deviation. If the z-score greater than 3 or less than -3, indicates an outlier value.

```
# calculating z score

import scipy
from scipy import stats

z_scores = scipy.stats.zscore(data["col1"])
```

A standard cut-off value for finding outliers are z-scores of +/-3

```
# removing outlier using z-score

data["col1"][~((z_scores < -3) |(z_scores > 3))]
```

```
# detecting and removing outlier for full dataset using Z

for i in df_numeric.columns:
    thresold = 3
    mean = data[i].mean()
    std = data[i].std()

    outliers = []
    
    for value in data[i]:
        zscore = (value-mean)/std
        if abs(zscore) > thresold:
            outliers.append(value)      

    print('The count of Outliers in the column {0} is {1}'.format(i,len(outliers)))
    
    
    
    for i in df_numeric.columns:   
        z_scores = stats.zscore(data[i])
        
    data=data.loc[~(( z_scores < -3) |(z_scores > 3))]
    print("column",i," done")
```

### Using IQR

The interquartile range is the middle 50% of the dataset, It ranges between the third and the first quartile. We use the interquartile range, first quartile, and third quartile to identify the outliers.&#x20;

```
# For full dataset outlier detection and removal using IQR

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1


data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
```

The outlier is a point which falls below **Q1 – 1.5×IQR** or above **Q3 + 1.5×IQR**

## Feature scaling

The data may contain the variables of the different numerical scales or magnitudes. We scale the variable to get all the variables in the same range or common scale. With this, we can avoid a problem in which some features come to dominate solely because they tend to have larger values than others.&#x20;

Feature scaling is also known as data normalization**.** Since the features have various ranges, it becomes a necessary step in data preprocessing while using machine learning algorithms.

For example:- In a dataset that has variables age and income. The age of a person is measured in years which can take values between 18 to 65 (retirement age) and the income of a person is in thousands So it is necessary to bring the two features on the same scale to assign appropriate weights

### Standardization or _Z-score normalization_

Standardization transforms the data such that the data has mean 0 and unit variance. The procedure involves subtracting the mean from observation and then dividing by the standard deviation. This method does not change the shape of the original distribution.

![](<../../.gitbook/assets/image (14).png>)

```
from sklearn.preprocessing import StandardScaler
standard_scale = StandardScaler() #instantiate

data['col'] = standard_scale.fit_transform(data[['col']]) #fit



# apply on whole dataset at once [dont' consider Y]
data = data.apply(lambda x: (x - x.mean())/x.std())
```

### Min-max normalization

Performs linear transformation on the original data and after normalization, all values will be between 0 and 1. And correlation does not change b/w variables after min-max normalization.&#x20;

![](<../../.gitbook/assets/image (13).png>)

```
from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()

data['co1'] = min_max.fit_transform(data[['col1']])
```

**Normalization** is a good technique to use when you do not know the distribution of your data or when you know the distribution is not Gaussian (a bell curve) or not normally distributed. **Standardization** assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to be true, but the technique is more effective if your attribute distribution is Gaussian.

## Data/Feature Transformation

### Log transformation

It reduces the skewness in the distribution of the original data, and makes it more interpretable and the arithmetic mean of the log-transformed data is the geometric mean of the original data. If the data values are increasing at an exponential rate, then log transformation can transform the values such that the values will increase linearly. It cannot be used on a categorical variable after dummy encoding since ln(0) is undefined. Also if a variable takes zero or negative values, logarithmic transform cannot be used on it.

```
data['col'] = np.log(data['col'])
```

### Square root transformation

Values of a variable are replaced with its square root, to reduce right skewness, we may use square root transformation and It can be applied even when the variable takes a zero value.

```
```

### Reciprocal transformation

Values of a variable are replaced with its reciprocal. It can not be applied only when the variable takes zero values, However, can be applied to negative values. Example: population per area (population density) transforms to area per person

```
```

### Exponential transformation

It is the inverse transformation of the log transformation. It is used to convert the log-transformed values to their original units.

```
data['col'] = np.exp(data['col'])
```

### Box cox transformation

The Box-Cox transformation can only be used on positive variables, it is a generalized form of logarithmic transformation.

```
```

## Feature Selection.

Feature selection is the process of including the significant features in the model, to understand the above methods let X1, X2, ..., Xk be k predictor variables and Y be the response variable.

### Forward selection method

* Start with a null model (with no predictors)
* Obtain the correlation between Y and each variable. The variable with the highest correlation gets added to the model (say Xm). Build a model Y \~ Xm
* Obtain the correlation between Y and remaining (k-1) variables. The next variable (say Xp) is included, which has the highest correlation with Y after removing Xm
* Build a model Y \~ X m + Xp. If Xp is significant include it in the model else discard
* Repeat steps (3) and (4) until reaching the stopping rule or running out of variables

### Backward elimination method

* Start with a full model (model with all k predictors)
* Remove the variable which is least significant (variable with the largest p-value)
* Fit a new model with remaining (k-1) regressors
* The next variable (say Xp) is removed if it is least significant
* Repeat steps (3) and (4) until reaching the stopping rule or all variables are significant.

### Stepwise method

It is a combination of forward selection and backward elimination method.

* Start with a null model (with no predictors)
* At each step add or remove variable based on its corresponding p-value
* Stop when no variable can be added or removed justifiably

### Recursive feature elimination (RFE)

It is an instance of backward feature elimination.

* Train a full model&#x20;
* Create subsets for features&#x20;
* Set the subset size&#x20;
* Compute the ranking criteria for each feature subset&#x20;
* Remove the feature subset that has the least ranking
