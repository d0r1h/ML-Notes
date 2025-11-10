---
description: >-
  In Statistic, linear regression is a linear approach to modelling the
  relationship between a scalar response(Y) and one or more explanatory
  variables(X)
---

# Linear Regression

Linear Regression is a simple and powerful model for predicting a numeric response from a set of one or more independent variables.&#x20;





Dependent and Independent Variable

* **Y** -> variable we wish to predict, i.e. dependent, Response, or Target variable                           &#x20;
* **x** -> variable used to predict Y, independent, predictor variable

## Regression Analysis

Regression Analysis allows us to examine which independent variables have more impact on the dependent variable and investigates and models the relationship between variables. It determines which variables can be ignored, which ones are most important, and how they influence each other.&#x20;

### Simple Linear Regression

A simple linear regression model(also called bivariate regression) has one independent variable x that linear relationship with dependent variable Y.

![](<../.gitbook/assets/image (16).png>)

here,  **a** – Intercept,  **b** – Slope, and  **ϵ** – Residual (error)

Intercept:                                                                 &#x20;

Slope:&#x20;

Error term/ Residual represents the distance of the observed value from the value predicted by the regression line.   **e = y - y^** for each observation.

### Multiple Linear Regression

It has multiple&#x20;

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
