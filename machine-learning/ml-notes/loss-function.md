---
description: A long list of curated Loss functions used in ML/DL
---

# Loss Function

All the algorithms in machine learning rely on minimizing or maximizing a function, which we call “objective function”. The group of functions that are minimized is called “loss functions”. A loss function is a measure of how good a prediction model does in terms of being able to predict the expected outcome. Loss function measures how far an estimated value is from its true value.

#### **Regression Loss \[**[**Code**](https://nbviewer.org/github/groverpr/Machine-Learning/blob/master/notebooks/05_Loss_Functions.ipynb)**]** <br>

1. Mean Square Loss, Quadratic loss, L2 loss&#x20;

[Mean Square Error (MSE)](https://medium.freecodecamp.org/machine-learning-mean-squared-error-regression-line-c7dde9a26b93) is the most commonly used regression loss function. MSE is the sum of squared distances between our target variable and predicted values.

2. Mean Absolute Error, L1 loss

[Mean Absolute Error](https://medium.com/@ewuramaminka/mean-absolute-error-mae-sample-calculation-6eed6743838a) (MAE) is another loss function used for regression models. MAE is the sum of absolute differences between our target and predicted variables. So it measures the average magnitude of errors in a set of predictions, without considering their directions.

3. Huber Loss, Smooth MAE

[Huber loss](https://en.wikipedia.org/wiki/Huber_loss) is less sensitive to outliers in data than the squared error loss. It’s also differentiable at 0. It’s an absolute error, which becomes quadratic when an error is small. How small that error has to be to make it quadratic depends on a hyperparameter, 𝛿 (delta), which can be tuned.&#x20;

4. Log-Cosh Loss&#x20;

Log-cosh is another function used in regression tasks that’s smoother than L2. Log-cosh is the logarithm of the hyperbolic cosine of the prediction error.

In short, using the squared error is easier to solve, but using the absolute error is more robust to outliers.MAE loss is useful if the training data is corrupted with outliers (i.e. we erroneously receive unrealistically huge negative/positive values in our training environment, but not our testing environment).

#### Classification Loss



![](<../../.gitbook/assets/unknown (4).png>)





