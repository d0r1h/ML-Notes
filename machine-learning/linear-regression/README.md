---
description: >-
  Linear regression is a linear approach to modelling the relationship between a
  scalar response(Y) and one or more explanatory variables(X)
---

# Linear Regression

Linear Regression is a simple and powerful model approach for predicting a numeric response from a set of one or more independent variables. It models the relationship between a dependent variable **y**  and one or more independent variables **X**. In Linear regression the goal is to find a line (or hyperplane) that best fits the data points by minimizing the error between predicted and actual values. Line in case of simple and hyperplane in case of multi linear regression.&#x20;

Dependent and Independent Variable

* **y** -> variable we wish to predict, i.e. dependent, Response, or Target variable                           &#x20;
* **X** -> variable used to predict Y, independent, predictor variable

**In Linear Regression there are two approaches for the training :-**&#x20;

1. Direct closed form equation (OLS)&#x20;
2. Gradient Descent (Iterative)&#x20;

We'll see these both in detailed ....&#x20;

**Based on the type of input data this algorithms can be applied in three different forms**&#x20;

1. Simple Linear Regression
2. Multi Linear Regression
3. Polynomial Regression&#x20;

#### Simple Linear Regression&#x20;

A simple linear regression model (also called bi variate regression) has one independent variable x that linear relationship with dependent variable Y. Let’s understand Simple Linear Regression with example,&#x20;

You have dataset of 10k student which consists of their CGPA and Package they got in the placement, so here is input feature is CGPA and target variable is package, X = CGPA and y = Package&#x20;

So equation of simple linear regression would

$$
y = mx + c
$$

Here **y** is the target or predicted variable and **X** is input data, and **m/c** are the model parameters which model learn from the data and in starting both are randomly initialized.  Here the model is a linear function of the input feature x. A linear model makes predictions by simply computing a weighted sum of input features and a constant called bias or intercept term.

**m** - weight or slope and **c** -  intercept or bias&#x20;

For training a Linear Regression model we need to find the values of m and c (or model parameter ) such that they reduce th Error /  MSE (mean squared Error) i.e, difference between actual value of y and predicted value of y^.  Error term/ Residual represents the distance of the observed value from the value predicted by the regression line.   **e = y - y^** for each observation.

So as mentioned earlier, we have a dataset of 10k (Training 9k and Validation set 1k), we need to find the feature weight for all these 9k students, which minimizes the error. &#x20;

To calculate the value of feature weights or slope (**m**) and intercept (**c**)&#x20;

$$
m = \frac{\sum_{i=1}^{m} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{m} (x_i - \bar{x})^2}
$$

$$
\newline
c = \bar{y} - m\bar{x}
$$

Here x\_bar and y\_bar are the avg value of  cgpa and package.&#x20;

Equation of MSE for linear regression model :-&#x20;

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

For every true value y and predicted value y\_bar for the same data, we calculate the error.&#x20;

Code for implementing Simple Linear Regression from scratch

```py
'''
Implementation of Simple Linear Regression from stratch Justing using Math 
Pandas - for reading data 
Sklearn - for splitting data into train and validation set 
'''

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('placement.csv')

print(data.head())

# Assigning data into variable X, and y 

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Spliting data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(data.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


class SimpelLinearRegression:
    '''
    Implementation of Simple Linear Regression (OLS)
    To make predictions using this we need to implement y = mx + c equation
    y and x are coming from data but we need to caculate m and c
    '''
    
    def __init__(self):

        self.m = None
        self.c = None

    def fit(self, X_train, y_train):
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):

            # for calcuation of m (Feature weight or slope)
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
            
        self.m = num/den 
        self.c = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        return (self.m * X_test) + self.c
    
lr = SimpelLinearRegression()

lr.fit(X_train, y_train)

print(f'coffecient_ : {lr.m}')
print(f'intercept_ : {lr.c}')

# Let's make predictions on the test set 

pred  = lr.predict(X_test)
```

#### Multi Linear Regression&#x20;

When we have more than one input features, Multi Linear Regression comes into picture, which is mostly the case in real world. In simple linear regression we try to fit the optimal line, but in multi we try to fit the plane(3D) or hyperplane (nD).&#x20;

The formula for multiple linear regression is expressed as:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon
= X\beta + \varepsilon
$$

* y is the predicted value of the dependent variable&#x20;
* β<sub>0</sub> is the y-intercept
* β<sub>1</sub> and β<sub>p</sub> are the regression coefficients(or weights) for each independent variable (x<sub>1</sub> to x<sub>p</sub>)
* ε represents the error term

In multiple linear regression, the β coefficients can be calculated using the following formula:

$$
\hat{\beta} = (X^{T}X)^{-1}X^{T}y
$$

Where:

* X is the design matrix that includes all the independent variables, with a column of ones added for the intercept.
* y is the vector of observed values of the dependent variable.
* X<sup>T</sup> is the transpose of the design matrix X.
* (X<sup>T</sup>X)<sup>-1</sup> is the inverse of the matrix product X<sup>T</sup>X.

#### Calculation Steps

1. **Construct the Design Matrix (X)**: Include all independent variables and a column for the intercept.
2. **Compute Transpose (X**<sup>**T**</sup>**)**: Calculate the transpose of the design matrix.
3. **Calculate Matrix Product**: Find the product of X<sup>T</sup> and X.
4. **Inverse Calculation**: Compute the inverse of the product from the previous step.
5. **Multiply with X**<sup>**T**</sup>**Y**: Multiply the inverse from step 4 with X<sup>T</sup> and then with the dependent variable vector Y.

This process yields the estimated coefficients β, which describe the relationship between the independent variables and the dependent variable in the model.

| Aspect                    | Simple Linear Regression               | Multiple Linear Regression                       |
| ------------------------- | -------------------------------------- | ------------------------------------------------ |
| **Equation**              | Y' = b0 + b1X + ε                      | Y' = b0 + b1X1 + b2X2 + ... + bkXk + ε           |
| **Number of Variables**   | One independent variable (X)           | Multiple independent variables (X1, X2, ..., Xk) |
| **Slope Calculation**     | b1 = (Σ(X - X̄)(Y - Ȳ)) / (Σ(X - X̄)²) | β = (X'X)^{-1}X'Y                                |
| **Intercept Calculation** | b0 = Ȳ - b1X̄                          | Part of the β coefficients calculation           |
|                           |                                        |                                                  |

```py
import numpy as np

class MultipleLinearRegression:
    '''
    Implementation of Multiple Linear Regression
    '''
    
    def __init__(self): 
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, X_test):
        X_train = np.insert(X_train, 0 , 1, axis=1) # insert 1 at the 0th index as column 

        # calculating the coefficient 
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
```



Above both discussed, Simple and Multi Linear Regression are closed forms solutions also known as OLS, and it solves a system of linear equations. &#x20;

<br>

Ordinary Least Squares (OLS) linear regression does not use an iterative learning process (like weight updations and all). It computes the optimal slope(s) and intercept in one shot, using a closed-form mathematical formula derived from calculus. It's like solving “Given these points, draw the best straight line.” You compute the answer mathematically.&#x20;

<br>

And reason for that is :-&#x20;

<br>

For linear regression with features XXX and target yyy, OLS finds parameter vector β\betaβ by solving:

<br>

β=(XTX)−1XTy.

This formula comes from setting the derivative of the MSE loss to zero and solving the resulting linear equations.

<br>

So the learning process is literally:

1. Compute matrix XTXX^T XXTX
2. Invert it (or use numerical equivalent)
3. Compute (XTX)−1XTy(X^T X)^{-1} X^T y(XTX)−1XTy
4. Done. Those are your slopes & intercept.

It’s deterministic and exact (assuming perfect numerical stability).

When dataset is huge (very large n or p)

Modern libraries sometimes avoid the closed-form OLS because:

* Matrix inversion is expensive
* Memory cost high: storing XTXX^T XXTX\
  <br>

In such cases the model may use:

* Gradient Descent
* Stochastic Gradient Descent (SGD)
* Mini-batch GD
* Coordinate descent (for Lasso)

### Measures of Variation <a href="#id-4.2-measures-of-variation" id="id-4.2-measures-of-variation"></a>

**SSE:** It is defined as the sum of the squared difference between the actual value and the predicted value.

**SSR:** It is defined as the sum of the squared difference between the predicted value and the mean of the dependent variable.

**SST**: It is the sum of the squared difference between the actual value and the mean of the dependent variable. And SST (Total variation) is the sum of SSR and SSE.

**Coefficient of Determination (R-Squared):** The coefficient of determination explains the percentage of variation in the dependent variable that the independent variables explain collectively.&#x20;

**SEE:** The SEE is the measure of the variability of actual values around the prediction line. The smaller the value of SEE better is the model.

### Assumptions of Linear Regression

Before building a model

* The dependent variable must be numeric
* Predictors must not show multicollinearity

After building a model

* Linear relationship between dependent and independent variables
* Independence of observations should exist (Absence of Autocorrelation)
* The error terms should be homoscedastic
* The error terms must follow the normal distribution

#### Multicollinearity

Multicollinearity arises when the independent variables have high correlation among each other. Multicollinearity may be introduced if there exists an empirical relationship among variables such as income = expenditure + saving. Also, the confidence interval obtained for β’s is wider since the SE(β) becomes large. These are the following ways to detect multicollinearity.&#x20;

* Is there multicollinearity present&#x20;
  * Determinant of correlation matrix
  * Condition Number (CN)
* Which variables are involved in multicollinearity
  * Correlation Matrix
  * Variance Inflation Factor (VIF)

Determinant of the correlation matrix: Let D be the determinant of the correlation matrix. Then 0 < D < 1. `D=0 High multicollinearity | D=1 No multicollinearity.`

Condition Number (CN):&#x20;

| Value           | Interpretation             |
| --------------- | -------------------------- |
| CN > 1000       | Severe multicollinearity   |
| 100 < CN < 1000 | Moderate multicollinearity |
| 100 < CN        | No multicollinearity       |

Variance Inflation Factor (VIF):

| Value       | Interpretation       |
| ----------- | -------------------- |
| VIF > 5     | High correlation     |
| 5 > VIF >1  | Moderate correlation |
| VIF = 1     | No correlation       |

#### Linear relationship

To check if there linear relationship between dependent and independent variables, we should plot a scatter plot of residuals vs predictors and if the scatter plot shows no pattern indicates that the variable has linear relationship with the response.&#x20;

#### Autocorrelation

Assumption of Auto/self-correlation is violated when residuals are correlated within themselves, i.e. they are serially correlated. It does not impact the regression coefficients but the associated standard errors are reduced. This reduction in standard error leads to a reduction in associated p-value and It incorrectly concludes that a predictor is statistically significant.

It occurs If the relationship between the target and predictor variables is non-linear and is incorrectly considered linear. To test whether the error terms are autocorrelated, we perform Durbin-Watson test. When we run OLS model we get D values in the statistic summary.&#x20;

| Value      |                           |
| ---------- | ------------------------- |
| 0 < d <2   |  Positive autocorrelation |
| d = 2      | No autocorrelation        |
| 2 < d < 4  | Negative autocorrelation  |

#### Homoscedasticity

If the residuals have constant variance across different values of the predicted values, then it is known as `Homoskedasticity`. The absence of homoskedasticity is known as, heteroskedasticity. One of the assumptions of linear regression is that heteroskedasticity should not be present. Let us study two different tests to check the presence of heteroskedasticity.&#x20;

#### Normality test of Error Terms

Normality tests are used to determine if a data set is well-modeled by a normal distribution. And it can be done using the following methods.

* Quantile-Quantile Plot
* Shapiro-Wilk Test



### Model evaluation metrics

#### **R square**

The R2 value gives the percentage of variation in the response variable explained by the predictor variable&#x73;**.** If the values of R2 = 0.87, it implies that 87% of variation in the response variable is explained by the predictor variables.&#x20;

#### **Adjusted R square**

Adjusted R2 gives the percentage of variation explained by independent variables that actually affect the dependent variabl&#x65;**.**&#x20;

#### **F test for significance**

To check the significance of the regression model we use the F tes&#x74;**,** which is similar to ANOVA for regression. And if p\_value is less than alpha(level of significance) then implies that the model is significant.

### Interaction effect



### Model Performance Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)



[https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics](https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics)&#x20;
