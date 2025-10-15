---
description: >-
  In machine learning, classification refers to a predictive modeling problem
  where a class label is predicted for a given example of input data.
---

# Classification

In classification, the target variable has categories. In the example, Cold, Warm and Hot are the categories of the target variable. These categories are called the class labels.

1. Binary Classification
2. Multiclass Classification&#x20;

Cross-Industry Standard Process for Data Mining (**CRISP-DM**) is a standard process used for data mining. It breaks into 6 phases:

* Business Understanding
* Data Understanding
* Data Preparation
* Modeling
* Evaluation
* Deployment

Odds vs Probability

![](<../../.gitbook/assets/image (20).png>)

![](<../../.gitbook/assets/image (21).png>)

![](<../../.gitbook/assets/image (23).png>)

Odds Ratio:- Odds ratio can be used to determine the impact of a feature on the target variable.  In our example odds ratio of Email being spam or not being spam

![](<../../.gitbook/assets/image (25).png>)

## Logistic Regression

Logistic regression is a binary classification algorithm, It predicts the probability of occurrence of a class label. Based on these probabilities the data points are labeled and a threshold (or cut-off; commonly a threshold of 0.5 is used) is fixed.

In the email example if probability > threshold then email is spam.&#x20;

### Sigmoid Function

The sigmoid function is a mathematical function which is S-shaped and is given by and it exists between 0 to 1.

<div align="center"><img src="../../.gitbook/assets/image (27).png" alt=""></div>

**Assumptions of Logistic Regression**

* Independence of error, whereby all sample group outcomes are separate from each other (i.e., there are no duplicate responses)
* Linearity in the logit for any continuous independent variables
* Absence of multicollinearity
* lack of strongly influential outliers

**Significance of coefficients**

In logistic regression, the significance of the coefficients is determined by the Wald statistic and by the likelihood ratio test, and to test the significance of the model, the likelihood ratio test is used.

For β(coefficients) to be significant, β > 0.  H0: β = 0 against H1: β ≠ 0 and Failing to reject H0 implies that the parameter β is not significant.&#x20;

Wald Test:

LRT(likelihood ratio test):

## Model Evaluation Metrics

* Training Error: Number of misclassification on the training set also know as Also known as re-substitution or apparent error.&#x20;
* Generalization error: Number of misclassification on the test set.

### Deviance

* Null model: A model without any predictors
* Saturated model: A model with exactly n samples (n predictors), that fits the data perfectly
* Full model: A model fitted with all the variables in the data
* Fitted model: A model with at least one predictor variable

Deviance is analogous to the sum of squares in the linear regression, and it is a measure of goodness of fit for logistic regression.

![](<../../.gitbook/assets/image (28).png>)

where a saturated model is a model assumed to have the perfect fit and If the saturated model is not available use the fitted model.&#x20;

* Null deviance: The difference between the log likelihood of the null model and the saturated model.
* Model deviance: The difference between the log likelihood of the null model and the fitted model

Smaller values indicate a better fit, and to check for the significance of k predictors, subtract the model deviance from the null deviance and access it on Xk.

### AIC

The Akaike Information Criteria (AIC) is a relative measure of model evaluation for a given dataset, and it is given by&#x20;

![](<../../.gitbook/assets/image (29).png>)

The AIC gives a trade-off between the model accuracy and model complexity, i.e. it prevents us from overfitting.

### Pseudo R2

There are various pseudo R2s developed that are similar on the scale, i.e. on \[0,1], and work exactly the same with higher values indicating a better fit.

**McFadden R2**: If comparing two models on the same data, we consider the model which has higher value is considered to be better. The pseudo R2 in the python output is the McFadden R2.

![](<../../.gitbook/assets/image (30).png>)

**Cox-Snell R2**: It is similar to the McFadden R2, the likelihood is the product of probability N observations of the dataset. Thus the Nth square root of the provides an estimate of each target value. The R2 Cox-Snell can be greater than 1, and for a model with likelihood 1, i.e if predictions are perfect, then the denominator becomes.&#x20;

![](<../../.gitbook/assets/image (33).png>)

**Nagelkerke R2**: It is based on Cox-Snell R2 , it scales the values so that the maximum is 1. If the full model predicts the outcome perfectly, i.e it has likelihood = 1, then R2 Nagelkerke = 1 and similarly, if the likelihood of null model is equal to that of full model then R2 Nagelkerke = 0.

![](<../../.gitbook/assets/image (34).png>)

```
# Logistic Regression using statsmodels

import statsmodels.api as sm

X = df.feature
y = df.target

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, df_target, random_state = 10, test_size = 0.2)

logreg = sm.Logit(y_train, X_train).fit()

print('AIC:', logreg.aic)

df_odds = pd.DataFrame(np.exp(logreg.params), columns= ['Odds']) 

# predictions on the test set

y_pred_prob = logreg.predict(X_test)
y_pred_prob.head()

y_pred = [ 0 if x < 0.5 else 1 for x in y_pred_prob]
y_pred[0:5]
```

## Model Performance Measures

When we have an imbalanced dataset using Precision and Recall as a performance measure is the best idea, not accuracy because the overwhelming number of examples from the majority class (or classes) will overwhelm the number of examples in the minority class, meaning that even unskillful models can achieve accuracy scores of 90 percent, or 99 percent, depending on how severe the class imbalance happens to be.

**Precision** quantifies the number of positive class predictions that actually belong to the positive class. **Recall** quantifies the number of positive class predictions made out of all positive examples in the dataset. **F-Measure** provides a single score that balances both the concerns of precision and recall in one number.

The confusion matrix provides more insight into not only the performance of a predictive model, but also which classes are being predicted correctly, which incorrectly, and what type of errors are being made.

When making a prediction for a binary or two-class classification problem, there are two types of errors that we could make.

* **False Positive**. Predict an event when there was no event.
* **False Negative**. Predict no event when in fact there was an event.

### Confusion matrix

```
               | Positive Prediction | Negative Prediction
Positive Class | True Positive (TP)  | False Negative (FN) 
Negative Class | False Positive (FP) | True Negative (TN)

```

The performance measure for the classification problem, It is a table used to compare predicted and actual values of the target variable. There are few other performance evaluation metrics.

Maximizing precision will minimize the number of false positives, whereas maximizing the recall will minimize the number of false negatives. Sometimes, we want excellent predictions of the positive class. We want high precision and high recall.

**Precision**: Appropriate when **minimizing false positives** is the focus.                    **Recall**: Appropriate when **minimizing false negatives** is the focus.

```python
def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
    sns.heatmap(conf_matrix, annot = True, fmt = 'd')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()
    
    
   
# call function to plot matrix
plot_confusion_matrix(model_name)

TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]                
```

* **Accuracy**: Accuracy is the fraction of predictions that our model got correct, and the higher the value of accuracy better is the model. Accuracy is not always a reliable metric and the scenario when accuracy is not a reliable metric is called the accuracy paradox



* **Precision**: Precision is the proportion of positive cases that were correctly predicted. It is calculated as the ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.&#x20;

```
precision = TP / (TP+FP)
```

* **Recall**: Recall is the proportion of actual positive cases that were correctly predicted and it is also sometimes called True Positive Rate (TPR) or Sensitivity. Unlike precision that only comments on the correct positive predictions out of all positive predictions, recall provides an indication of missed positive predictions.

```
recall = TP / (TP+FN)
```

* **Specificity**: Specificity is the proportion of actual negative cases that were correctly predicted and the higher the value better is the model.

```
specificity = TN / (TN+FP)
```

* **F1 score**: F1 score is the harmonic mean of precision and recall values for a classification model, it is a good measure if we want to find a balance between precision and recall or if there is an uneven distribution of classes (either positive or negative class has way more actual instances than the other). Higher the F1 score better the model. F-Measure provides a way to combine both precision and recall into a single measure that captures both properties.

![mathsisfun.com](<../../.gitbook/assets/image (53).png>)

```
f1_score = 2*((precision*recall)/(precision+recall))
```

```
# correct prediction
accuracy = (TN+TP) / (TN+FP+FN+TP)


classification_report(y_test, y_pred)
```

* **False Positive Rate**: False Positive Rate (FPR) is the proportion of actual negative cases that were predicted positive (incorrectly). Lower the value of FPR better is the model.&#x20;



* **Kappa**: Reliability is the degree to which an assessment tool produces consistent results, Inter-rater reliability is used to measure the degree to which different raters agree while assessing the same thing. The kappa statistics is used to test inter-rater reliability.&#x20;

![](<../../.gitbook/assets/image (39).png>)

| Kappa        | Interpretation           |
| ------------ | ------------------------ |
| < 0          | No agreement             |
| 0-0.2        | Slight agreement         |
| 0.2-0.4      | Fair agreement           |
| 0.4-0.6      | Moderate agreement       |
| 0.6-0.8      | Substantial agreement    |
| 0.8-1        | Almost perfect agreement |

```
kappa = cohen_kappa_score(y_test, y_pred)
```

Calculation of po(relative observed agreement between raters):  Accuracy&#x20;

Calculation of pe(hypothetical probability of chance agreement):

![](<../../.gitbook/assets/image (40).png>)

![](<../../.gitbook/assets/image (41).png>)



### Cross entropy

Cross entropy is the loss function commonly used in classification problems, As the prediction goes closer to the actual value the cross-entropy decreases.&#x20;

![](<../../.gitbook/assets/image (46).png>)

### ROC

Receiver operating characteristics (ROC) curve is the plot of TPR(x-axis) against the FPR values obtained at all possible threshold values, The TPR and FPR values change with different threshold values.&#x20;

The true positive rate is calculated as the number of true positives divided by the sum of the number of true positives and the number of false negatives. It describes how good the model is at predicting the positive class when the actual outcome is positive. The true positive rate is also referred to as **sensitivity**.

`True Positive Rate = True Positives / (True Positives + False Negatives)`

The false positive rate is calculated as the number of false positives divided by the sum of the number of false positives and the number of true negatives. It is also called the false alarm rate as it summarizes how often a positive class is predicted when the actual outcome is negative.

`False Positive Rate = False Positives / (False Positives + True Negatives)`

Area under the ROC curve (**AUC**) is the measure of separability between the classes of target variables. AUC increases as the separation between the classes increases. Higher the AUC better the model.&#x20;

## Identify the Best Cut-off Value

### **Youden’s index for Cut-off**

Sensitivity and specificity represent the total number of correctly identifies samples (true positives and true negatives). Youden’s index is the classification cut-off probability for which the (Sensitivity + Specificity -1) value is maximized. Higher the value of Youden’s index better the model.

```
# select the cut-off(Threshold) probability for which the (TPR - FPR) is maximum

y_pred_prob = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

youdens_table = pd.DataFrame({'TPR': tpr,
                             'FPR': fpr,
                             'Threshold': thresholds})
youdens_table['Difference'] = youdens_table.TPR - youdens_table.FPR
youdens_table = youdens_table.sort_values('Difference', ascending = False).reset_index(drop = True)
youdens_table.head()                             
```

### Cost-based Method

We can use the cost-based method to calculate the optimal value of the cut-off. In this method, we find the optimal value of the cut-off for which the total **cost is minimum**. The total cost is given by the formula:

&#x20;**`total_cost = FN x C_1 + FP x C_2`**

Where,\
&#x20;C\_1: It is the cost of false negatives\
&#x20;C\_2: It is the cost of false positives

The cost values can be decided using business knowledge.

```
def calculate_total_cost(actual_value, predicted_value, cost_FN, cost_FP):
    cm = confusion_matrix(actual_value, predicted_value)
    cm_array = np.array(cm)
    return cm_array[1,0] * cost_FN + cm_array[0,1] * cost_FP
    
df_total_cost = pd.DataFrame(columns = ['cut-off', 'total_cost'])

i = 0    
for cut_off in range(10, 100):
    total_cost = calculate_total_cost(y_test,  y_pred_prob.map(lambda x: 1 if x > (cut_off/100) else 0), 3.5, 2
    df_total_cost.loc[i] = [(cut_off/100), total_cost] 
    i += 1
    

df_total_cost.sort_values('total_cost', ascending = True).head(10)
```

## Imbalanced Data

Data is imbalanced if there are more records of one class compared to other classes, it may lead to the accuracy paradox and In reality, datasets always have some degree of imbalance.&#x20;

{% hint style="info" %}
Example of Imbalanced Data
{% endhint %}

* Upsample minority class
* Down-sample majority class
* Change the performance metric
* Try synthetic sampling approach
* Use different algorithm

## Naïve Bayes

Conditional probability is the likelihood of an event given that another event has occurred, Bayes theorem provides a way to updated the probability based on the new information and It is completely based on conditional probability.&#x20;

![](<../../.gitbook/assets/image (42).png>)

**Posterior Probability**: In context with a classification problem the posterior probability is the conditional probability of a class label taking value t given that the predictor takes value x. For example, consider the example of labeling an email as spam or ham. The conditional probability that it is a spam message given the word appears in it, i.e. P(spam | word) is the posterior probability.&#x20;

**Prior probability**: Prior probability is the probability of an event computed from the data at hand. For example, consider the example of labeling an email as spam or ham. The probability the email is spam, i.e. P(spam) is the prior probability Likewise P(ham) is also a prior probability.&#x20;

**Likelihood:** In context with a classification problem the Likelihood is the conditional probability of a predictor taking value x given that its class label is t. For Example, consider the example of labeling an email as spam or ham. The conditional probability that the word appears in spam, i.e. P(word | spam) is the likelihood.

**Evidence**: It is the probability that the predictor takes value x and also known as marginal probability. For Example, consider the example of labeling an email as spam or ham. The probability that the word appears in a message, i.e. P(word) is the evidence.

A **Naïve Bayes classifier** uses the Bayes’ theorem for classification, It is an eager learning algorithm. Since it does not wait for test data to learn, it can classify the new instance faster.&#x20;

Assumptions of Naïve Bayes

* The predictors are independent of each other
* All the predictors have an equal effect on the outcome

{% hint style="info" %}
Example of Spam/Ham using Naive Bayes
{% endhint %}

Laplace smoothing metho&#x64;**:** To solve the zero probability problem we use Laplace smoothing method. Add α to every count so the count is never zero and α > 0. Generally, α = 1.&#x20;

Naïve Bayes Classifier available in the scikit learn library:

* Gaussian Naïve Bayes: predictors are continuous and normally distributed
* Multinomial Naïve Bayes: used in text(document classification problem
* Bernoulli Naïve Bayes: predictors are boolean.

Naïve Bayes used in Spam Filtering, Sentiment Analysis, and Recommendation System.

It is easy to implement in the case of text analytics problems, can be used for multiple class prediction problems, performs better for categorical data than numeric data.

Fails to find the relationship among features, May not perform when the data has more predictor, and the assumption of independence among features may not always hold good.

## KNN(K - Nearest Neighbours)

The proximity measures find the distance between two instances, depending upon the data types, we choose the proximity measure.&#x20;

* **Similarity measures:** A similarity measure for two objects, will return the value 0 if the objects are unlike, and the value 1 if the objects are alike.
* **Dissimilarity measure**: A dissimilarity measure for two objects, will return the value 1 if the objects are unlike, and the value 0 if the objects are alike.

### Distance measures

For Numeric data&#x20;

* Euclidean distance
* Manhattan distance
* Minkowski distance
* Chebyshev's distance

For String data

* Chebyshev's distance
* Edit distance
* Longest Common Sequence
* Hamming distance

The K - Nearest Neighbour (KNN) algorithm classifies the data based on the similarity measur&#x65;**,** K specifies the number of nearest neighbors to be considered, does not require the data to be trained and KNN does not return the model.

* It is Instance based learning algorithm: uses training instances to make predictions.
* Lazy learning algorithm: does not require a model to be trained
* Non-Parametric algorithm: no assumptions are made about the functional form of the problem being solved

Selecting an apt K is challenging. To overcome this, **weighted KNN** is used, Weights are assigned to each instance. Generally the weights are the inverse of the distance. The weights are higher for the points which are nearer to the new instance and The weights are lower for the points which are away from the new instance.

**KNN algorithm - Procedure**

1. **C**hoose the distance measure and value of K
2. Compute the distance between the point whose label is to be identified (say x) and other data points.
3. Sort the distance in ascending order.
4. Chose the k data points which have shorted distance and note their corresponding labels. Then the label which has the highest frequency will be assigned to the point x.

{% hint style="info" %}
For even number of class, labels consider K to be odd and for odd number of class labels consider K to be even

Example of KNN
{% endhint %}

Application of KNN:

* Image classification&#x20;
* Handwriting recognition&#x20;
* Predict credit rating of customers&#x20;
* Replace missing values



Entropy

Entropy is the measure of information for classification problem, i.e. it measures the heterogeneity of a feature and calculated as:

![](<../../.gitbook/assets/image (43).png>)

A lower entropy is always preferred and it is always non-negative.

For example if target variable has two class and distribution between them is equal that is class A has 50% of data and class B has 50% of data, so in that case entropy will be 1.&#x20;

Information Gain: It is the decrease in entropy at a node. To construct the decision tree, the feature with the highest information gain is chosen as root node.  Information gain is always positive.&#x20;

After the split of data, the purity of data will be higher as a result the entropy will always be lower. Thus, the information gain is always positive.

## Decision Trees

Decision tree is a classifier that results in flowchart-like structure with nodes and edges.  Branch/sub-tree: a subsection of the entire decision tree. Root Node: no incoming edge and zero or more outgoing edges. Internal Node: exactly one incoming edge and zero or more outgoing edges. Leaf/Terminal Node: exactly one incoming edge and no outgoing edge.

A decision tree is built from top to bottom. That is we begin with the root node, While constructing a decision tree we try to achieve pure nodes. A node is considered to be pure when all the data points belong to the same class. This purity of nodes is determined using the entropy value.

To select the root node, from k features, select the feature with the highest information gain and split the data on this feature, at the next node, from (k-1) features, select the feature with the highest information gain and again split the data on this feature, continue the process till you exhaust all features

If you need to choose between attributes with same information gain, then the first predictor found from left to right in a data set is considered, some decision tree algorithm implementations might consider each of the variables with same information gain at a time and check which model performs better. This rule applies to all parent nodes.&#x20;

Decision trees are prone to overfitting, and overfitting occurs when the decision tree uses all of the data samples in the decision tree, resulting in a perfect fit. An overfitted tree may have leaf nodes that contain only one sample, ie. singleton node and overfitted tree are generally complicated and long decision chains. An overfitted tree has low training error and a high generalization error, hence can not be generalized for new data.&#x20;

An approach to handle overfitting is **pruning.** Pruning is a technique that removes the branches of a tree that provide little power to classify instances, thus reduces the size of the tree.&#x20;

* Pre-Pruning: The decision tree stops growing before the tree completely grown
* Post-Pruning: The decision tree is allowed to grow completely and then prune

**Hyperparameters**: Pre-pruning can be done by specifying the following hyperparameters.

* max\_depth: maximum length of the decision allowed to grow
* min\_samples\_split: minimum samples required to split an internal node
* max\_leaf\_nodes: maximum number of leaf nodes the decision tree can have
* max\_feature\_size: maximum number of features to be considered to while splitting a node
* min\_samples\_leaf: minimum samples required to be at the leaf node

**Algorithms**:&#x20;

* ID3 Algorithm: only categorical data
* C4.5 Algorithm: both categorical, numeric data and missing values '?'
* C5.0 Algorithm: Work faster and more memory efficient than C4.5
* CART

**Measures of Purity of a node**

Entrop&#x79;**:**&#x20;

Gini Index:&#x20;

![](<../../.gitbook/assets/image (44).png>)

Classification error: For samples belonging to one class, the classification error is 0 and for equally distributed samples, the classification error is 0.5

![](<../../.gitbook/assets/image (45).png>)

## Ensemble Learning

Ensemble learning algorithms combine multiple models into one predictive model. Decisions from several weak learners are combined to increase the model performance.&#x20;

### Bagging

Designed to improve the stability (small change in datasets change the model) and accuracy of classification and regression models, It reduces variance errors and helps to avoid overfitting. Uses sampling with replacement to generate multiple samples of a given size. Sample may contain repeat data points For classification bagging is used with voting to decide the class of input while for regression average or median values are calculated. **In bagging base learners learn is parallel.**

![](<../../.gitbook/assets/image (52).png>)

For example: Random Forest&#x20;

### Boosting

Boosting is an ensemble of weak learners that learn sequentially. In each iteration weights of the samples are adjusted, such that the misclassified samples have a higher weight, therefore higher chance of getting selected to train the next classifier. Boosting reduces bias and variance. It enhances the efficiency of weak classifiers and both precision and recall can be enhanced through boosting algorithms. **In bosting base learners learn sequentially.**

For example: Adaboost,  Gradient Boosting, XGBoost

### Stacking

For example: Voting classifier

## Random Forest Classifier

Random Forest consists of several independent decision trees that operate as an ensemble, It is an ensemble learning algorithm based on bagging. Train decision tree models on bootstrap samples where variables are selected at random. The aggregate output from these tree is considered as the final output.&#x20;

![](<../../.gitbook/assets/image (48) (1).png>)

**Hyperparameter**

* n\_estimators: number of decision tree built for random forest&#x20;

**Feature Importance**: A technique that assigns a score to independent features based on its importance in predicting the target variable.

* Gini importance: Also known as mean decrease impurity, It is the average total decrease in the node impurity weighted by the probability of reaching it. The average is taken over all the trees of the random forest.
* Mean decrease in accuracy: Measure the decrease in accuracy on the out-of-bag data. Basically, the idea is to measure the decrease in accuracy on OOB data when you randomly permute the values for that feature and If the decrease is low, then the feature is not important, and vice-versa.

## Adaboost

## Gradient Boosting

## XGBoost

## External Link to refer

{% embed url="https://machinelearningmastery.com/types-of-classification-in-machine-learning/" %}

{% embed url="https://machinelearningmastery.com/bayes-theorem-for-machine-learning/#:~:text=The%20Bayes%20Theorem%20assumes%20that,dependent%20upon%20all%20other%20variables.&text=This%20means%20that%20we%20calculate,class)%20%2F%20P(data)" %}

