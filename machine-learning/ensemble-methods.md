# Ensemble Methods

* Voting Classifier -> use different classifiers -> Hard voting -> Final prediction
* Pasting&#x20;
* Boosting : XgBoost & Adaboost

Ensemble learning combines several learners (models) to improve overall performance, increasing predictiveness and accuracy in machine learning and predictive modeling. Technically speaking, the power of ensemble models is simple: they can combine thousands of smaller learners trained on subsets of the original data.&#x20;

This can lead to interesting observations, like:

* The variance of the general model decreases significantly thanks to bagging.
* The bias also decreases due to boosting.
* And overall predictive power improves because of stacking.  <br>

1. **Sequential ensemble methods:** learners are generated sequentially. These methods use the dependency between base learners. A popular example of sequential ensemble algorithms is AdaBoost.&#x20;
2. **Parallel ensemble methods:** learners are generated in parallel. The base learners are created independently to study and exploit the effects related to their independence and reduce error by averaging the results. An example of implementing this approach is Random Forests.
   1. **Homogeneous** ensemble methods typically use a single type of base learning algorithm, diversifying the training data by weighting samples. Ensemble algorithms that use bagging, like Decision Tree Classifiers.
   2. **Heterogeneous** ensembles, on the other hand, consist of members with different base learning algorithms that can be combined and used simultaneously to form the predictive model. Bagged and Boosted decision Trees like XGBoost.

### Bagging&#x20;

Bagging, short for Bootstrap Aggregating, is a technique that reduces overall variance by combining multiple models. It works by creating multiple subsets of the original dataset through random sampling with replacement, a process known as bootstrapping. A separate model is then trained on each of these subsets. When making predictions, bagging combines the outputs of all these models.

For classification problems, such as in Random Forests, it typically uses majority voting to determine the final prediction.&#x20;

For regression problems, it usually averages the predictions of all models.

![](<../.gitbook/assets/unknown (11).png>)



### Boosting&#x20;

This technique matches weak learners that have poor predictive power and do slightly better than random guessing to a specific weighted subset of the original dataset. Higher weights are given to subsets that were misclassified earlier.



![](<../.gitbook/assets/unknown (12).png>)



**References :-** <br>

* [https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
* [https://www.kaggle.com/code/alexisbcook/xgboost](https://www.kaggle.com/code/alexisbcook/xgboost)
* [https://www.kaggle.com/discussions/questions-and-answers/160932](https://www.kaggle.com/discussions/questions-and-answers/160932)
* [https://www.kaggle.com/code/dansbecker/xgboost](https://www.kaggle.com/code/dansbecker/xgboost)
* [https://ai.plainenglish.io/xgboost-regression-in-depth-cb2b3f623281](https://ai.plainenglish.io/xgboost-regression-in-depth-cb2b3f623281)
* [https://www.igmguru.com/blog/xgboost](https://www.igmguru.com/blog/xgboost)
* [https://dzone.com/articles/xgboost-deep-dive](https://dzone.com/articles/xgboost-deep-dive)
* [https://medium.com/@ml.enesguler/understanding-xgboost-from-basics-to-advanced-insights-d88536d87038](https://medium.com/@ml.enesguler/understanding-xgboost-from-basics-to-advanced-insights-d88536d87038)
* [https://www.mygreatlearning.com/blog/generalized-linear-models/](https://www.mygreatlearning.com/blog/generalized-linear-models/)
* [https://scikit-learn.org/stable/modules/ensemble.html#bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging)
* [https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)
* [https://www.ijnrd.org/papers/IJNRD2411095.pdf](https://www.ijnrd.org/papers/IJNRD2411095.pdf)&#x20;
* [https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/)
* [https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/)
* [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9893798](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9893798)
