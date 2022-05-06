---
description: >-
  Optimization is the problem of finding a set of inputs to an objective
  function that results in a maximum or minimum function evaluation.
---

# Optimization

In optimization, the main aim to find the weights that reduce the loss. &#x20;

![](<../../.gitbook/assets/image (61).png>)

* Prediction Evaluation: Process of evaluating how effectively the constructed model performs predictions.&#x20;
* Model Validation: Using test data to validate the model built using train data
* Fine Tuning: Maximizing the performance of a constructed model

## Prediction Evaluation

To construct a model with high prediction efficacy it is important to consider the prediction errors, and that is Bias and Variance.&#x20;

### **Bias:**

### **Variance:**

****

If the model is too simple it will have a `high bias and low variance`**,** Such a model will give not perfectly accurate predictions, but the predictions will be consistent. The model will not be flexible enough to learn from the majority of given data, this is termed as **underfitting.**

If the model is too complex it will have a `low bias and high variance`, Such a model will give accurate predictions but inconsistently. The high variance indicates it will have a much better fit on the train data compared to the test data, this is termed **overfitting**. OR If a model performs very well on the training data but does not perform well on the testing data, it is said to have high generalization error and high generalization error implies overfitting. And it can be reduced by avoiding overfitting in the model.&#x20;

## Model validation

The model validation methods use test data to validate the model built using train data.

### **k-fold Cross Validation:**&#x20;

Each observation is used exactly k times for training and exactly once for testing.

* Partition the dataset into ‘k’ subsets
* Consider one subset as the test set and the remaining subsets as the train set
* Measure the model performance
* Repeat this until all k subsets are considered as the test set
* The total error is obtained by summing up the errors for all the k runs

### **Leave one out Cross Validation (LOOCV):**&#x20;

It is a special case of k - fold cross-validation method. Instead of subsetting**,** the data, at every run one observation is considered as the test set. For n observations, there are n runs and the total error is the sum of errors for n runs.&#x20;

## Gradient Descent

Gradient Descent is an optimization technique/algorithm which is used to minimize the cost function. A cost function tells how good the model performance at the making predictions for a given set of parameters. It's also called Loss function or Error function. For linear regression cost function given by

![](<../../.gitbook/assets/image (17).png>)

It is an iterative method that converges to the optimum solution, It takes large steps when it is away from the solution and takes smaller steps closer to the optimal solution and the estimates of the parameter are updated at every iteration.

* What should be the step size or learning rate?

The gradient descent technique has a hyperparameter called learning rate, α. It specifies the jumps the algorithm takes to move towards the optimal solution. For very large α, the algorithm may skip the optimal solution and converges to a suboptimal solution and For very small α, the algorithm is more precise, however computationally expensive thus, it is important to choose an appropriate learning rate.

![](<../../.gitbook/assets/image (18).png>)

* How did we know the value of the intercept is to be increased.

It is determined by the derivative of the cost function and it is always a positive number and parameters are updated as`-` `New Parameter = Old parameter - (learning rate * derivative)` The gradient descent computes the derivative of the cost function at each iteration. This derivative value determines the increase/decrease in the parameter.&#x20;

Following steps are involved in GD:

* Start with some initial set of parameters
* Compute the cost function
* The derivative of the cost function (delta; δ) is calculated
* Update the parameters based on learning rate α and derivative δ
* Repeat the procedure until the derivative of the _cost function is zero._

### Batch gradient descent

The batch gradient descent computes the cost function with respect to the parameter for the entire data and also known as vanilla gradient descent. In spite of being computationally expensive, it is efficient and gradually converges to the optimal solution.&#x20;

### Stochastic gradient descent

For data with many samples and many features, the batch gradient descent is slow. The SGD works efficiently for large data as it works with only a single observation at each iteration, i.e. this one sample is used to calculate the derivative. The advantage of SGD is that we can add more data to the train set. The estimated new parameters are based on the recent estimates and previous and it is especially useful in presence of clustered data.

### Mini batch gradient descent

It is a combination of both Batch gradient descent and SGD, Like in SGD where one sample is considered, mini-batch uses a group of samples and as in batch GD, all the sample are considered to obtain the cost function hence it works faster than batch gradient descent and SGD.



{% tabs %}
{% tab title="First Tab" %}
{% embed url="https://www.youtube.com/watch?v=sDv4f4s2SB8" %}
{% endtab %}

{% tab title="Second Tab" %}

{% endtab %}
{% endtabs %}

## Regularization

Regularization refers to the modifications we make to a learning algorithm, that help in reducing its generalization error but not its training error it adds a penalty term to the cost function such that the model with higher variance receives a larger penalty and chooses a model with smaller parameter values (i.e.shrinked coefficients) that has less error.

Regularization converges the beta coefficients of the linear regression model towards zero, this is known as **shrinkage.** And for linear regression, the goal is to minimize the error, where _penalty = λ(regularization parameter) \* w(weight)_&#x20;

![](<../../.gitbook/assets/image (19).png>)

λ can take any values from 0 to infinity and if λ = 0, then there is no difference between a model with regularization and without regularization. And the best value for λ is determined by trying different values, the value that leads to least cross-validation error is chosen.

### Ridge regression

Ridge regression uses squared L-2 norm regularization i.e it adds a squared L-2 penalty, Here, the w is the L2 norm.&#x20;

### Lasso regression

Here, the w is the L1 norm

### Elastic net regression

It is a combination of ridge and lasso regression

* If there are many interactions present or it is important to consider all the predictors in the model, ridge regression is used&#x20;
* If the dataset contains some useless independent variables that can be eliminated from the model, lasso regression is used&#x20;
* If the dataset contains too many variables where it is practically impossible to determine whether to use ridge or lasso regression, elastic-net regression is used.

## Grid Search

Some parameters do not learn from the model, they are preset by the user. Such parameters are called hyperparameters such as, in the gradient descent, the learning rate (α) is a hyperparameter also the parameter lambda (λ) in regularization is a hyperparameter.

The grid search is the process of tuning the hyperparameters to obtain the optimum values of the hyperparameters and ‘**GridSearchCV**’ method to tune the hyperparameters.

...&#x20;
