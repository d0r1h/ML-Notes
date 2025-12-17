---
description: Ensemble Method
---

# Random Forest

Random Forest is a bagging technique which uses n decision tress, and uses voting classifier to get the final results, with row sampling with replacement.&#x20;

Reason to use RF over decision tree is decision tree with max depth gives a Low bias (Get well with training data) and High variance (bad results on the test set) model, using RF it makes the low variance.

With low Bias Decision tree model tend to achieve very low training error, because of splitting data into till max depth and high variance model get's very sensitive to small changes to the test set, leading to overfitting.&#x20;

RF uses voting method for classification and Average methods for regression.&#x20;
