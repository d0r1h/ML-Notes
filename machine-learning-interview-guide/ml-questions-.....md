---
description: Machine Learning & AI questions
---

# ML Questions ....

* **What are Loss Functions and Cost Functions? Explain the key difference between them?**

The loss function is to capture the difference between the actual and predicted values for a single record whereas cost functions aggregate the difference for the entire training dataset. The Most commonly used loss functions are Mean-squared error and Hinge loss.&#x20;

Mean-Squared Error(MSE): In simple words, we can say how our model predicted values against the actual values.  MSE = √(predicted value - actual value)2&#x20;

Hinge loss: It is used to train the machine learning classifier, which is L(y) = max(0,1- yy)&#x20;

Where y = -1 or 1 indicates two classes and y represents the output form of the classifier. The most common cost function represents the total cost as the sum of the fixed costs and the variable costs in the equation y = mx + b

\
**Collected from twitter :-**

1. How do you handle an imbalanced dataset and avoid the majority class completely overtaking the other?&#x20;
2. How do you deal with out-of-distribution samples in a classification problem?
3. How would you design a system that minimizes Type II errors?
4. Sometimes, the validation loss of your model is consistently lower than your training loss. Why could this be happening, and how can you fix it?
5. Explain what you would expect to see if we use a learning rate that's too large to train a neural network.
6. What are the advantages and disadvantages of using a single sample (batch = 1) on every iteration when training a neural network?
7. What's the process you follow to determine how many layers and nodes per layer you need for the neural network you are designing.
8. What do you think about directly using the model's accuracy as the loss function to train a neural network?
9. Batch normalization is a popular technique when training deep neural networks. Why do you think this is the case? What benefits does it provide?
10. Why do you think combining a few weak learners into an ensemble would give us better results than any of the individual models alone?
11. Early stopping is a popular regularization technique. How does it work, and what are some of the triggers that you think could be useful?
12. Explain how one-hot-encoding the features of your dataset works. What are some disadvantages of using it?&#x20;
13. Explain how Dropout works as a regularizer of your deep learning model.
14. How would you go about reducing the dimensionality of a dataset?
15. Can you achieve translation invariance when processing images using a fully connected neural network? Why is this the case?
16. Why are deep neural networks usually more powerful than shallower but wider networks?

Neural Network

17. What's the best way to initialize the weights of a neural network?&#x20;
18. What criteria would you follow to choose between mean squared error (MSE) and mean absolute error (MAE) as the loss function of your neural network?
19. &#x20;How do you determine the number of hidden layers that will best solve your problem(in a neural network)?
20. How do you decide it is a good time to introduce a scheduler to decrease your learning rate?
21. For how long should you keep training your neural network?
22. What would you expect to see as you vary the batch size used to train your model?
23. What would you expect to see as you vary the learning rate used to train your model?
24. How would you explain the loss of your classification model getting worse with every epoch but the accuracy improving?
25. Batch normalization and dropout are two different techniques helpful to combat overfitting. How are they different?
26. What would you expect to happen as you increase the number of layers and nodes per layer of your network?
27. Assume you are using Batch gradient descent, what advantage would you get from shuffling your training data.
28. Are feature engineering and feature extraction still need when applying deep learning?
29. What are the advantages of RNN over the fully connected network when working with text data?
30. Machine learning model error analysis and evaluating with different matrices and testing with different optimization algorithms.&#x20;
31. Tradeoff Bias/variance, Precision/Recall, and Under-fitting/over-fitting&#x20;
32. How to deal with an Imbalanced dataset in the context of the classification problem.
33. Distance Materials in Machine learning (KNN)
34. Regularization and optimization&#x20;





**Q. What is the difference between supervised and unsupervised learning?**

The most distinguishing difference between supervised and unsupervised learning is if we have access to ground truth data during training. This means, can we tell our model when it is right or wrong or not.

_Supervised learning:_ Supervised models learn to predict a certain output from the input data, based on many example input/output pairs. A typical example is a classification - is this an image of a car or not. We train a model using lots of examples of cars and not-cars.

_Unsupervised learning:_ Here we don't have ground truth, so the model needs to find patterns in the data by itself. Typical examples, Clustering, Dimensionality reduction, Anomaly detection, Autoencoders

**Q. Choose one method for unsupervised learning and explain its idea.**

One of the most fascinating unsupervised learning methods is the Autoencoder, The Autoencoder is a neural network that tries to output the same data as its input. The trick is that the layers in between containing fewer neurons than the input and the output. Therefore, the net is "forced" to learn an efficient (compressed) representation of the input data. This means that the net will learn features that capture the structure of the data in the encoder part. After training, the encoder part can be used to compute these features on arbitrary input and use them in a supervised method.

K-Means Clustering is one method of Unsupervised Learning. On specifying the number of clusters ‘K’, the algorithm splits the data points into the specified number of clusters. The data points within a cluster are more ‘similar’ to each other than to points outside. For example, A child asked to separate pictures of animals into 2 groups may separate by color (black/non-black) or by size (big/small), or by shape (animals/birds). And the problem with k-means clustering is determining the optimum number of clusters. K-means splits the data into the number of clusters determined by us beforehand. Finding the optimum number of clusters is a challenging task, we can’t just look at the data and determine how many partitions we should have.

Clustering with DBSCAN. It measures regions of high density based on core instances within a neighborhood of a given distance away. It separates dense regions (many instances) from sparse regions (blank or anomalies) to separate clusters (which can be fed into a predictive model).

K-means measures the distance to the centroid of a cluster, so it’s not as good with irregularly shaped clusters/ or clusters close together irregular boundaries and you have to prespecify the number of k clusters, K means is better, however, with clusters with clearly defined boundaries and clusters with anomalies still separated from other clusters than DBSCAN.&#x20;

Principal Component Analysis (PCA) is one of the unsupervised learning methods. Using PCA, we can reduce the feature dimension and understand the data pattern easily by visualizing a few dimensions. In most Machine learning problems, we have a large number of features, high dimension makes it harder to visualize the data and there might be a case that these features are correlated. With PCA, we create new feature sets which are the linear combinations of original features and are independent of each other -> get rid of correlated features.

Also, PCA can also be used in supervised learning problems as a preprocessing step. Reducing features might be helpful to reduce overfitting in supervised learning problems when we use PCA as a preprocessing step.<br>

**Q. Why is deep learning called "deep"? Why is it so popular nowadays? Why/when is it better than traditional ML methods?**

Deep Learning is built on the architecture of multiple layers and state of the art DL models may be hundreds of such layers deep. Each layer of the DL network extracts/learns aspects of the input data which in most cases is unstructured data like images, text or speech. Deep Learning has worked exceptionally well in image recognition tasks (e.g. AlexNet), speech recognition, language translation etc. The diff. between DL vs. classical ML methods, feature extraction i.e. selecting/encoding key features of data is not required for DL.

**Q. What are common challenges encountered during the training of a deep neural network? How can you solve them?**&#x20;

Overfitting, Underfitting, Lack of training data, Vanishing gradients, Exploding gradients, Dead ReLUs, Network architecture design, Hyperparameter tuning

Solving the above problems:-&#x20;

Overfitting Your model performs well during training, but poorly during tests.&#x20;

Possible solutions:  Reduce the size of your model, Add more data, Increase dropout, Stop the training early, Add regularization to your loss, Decrease batch size

Underfitting Your model performs poorly both during training and test. Possible solutions: - - - - - - Increase the size of your model - Add more data, Train for a longer time, Start with a pre-trained network

**Q. Lack of training data Deep learning algorithms are hungry for data compared to classical ML methods.**&#x20;

Possible solutions: Get more data, Use data augmentation, Use a pre-trained network and fine-tune for your problem - Try transfer learning

**Q. Vanishing Gradient During training the gradients in the first layers become small or 0. Learning is slow or the net doesn't learn at all.**&#x20;

Possible solutions: - Use ReLU, which doesn't saturate in the positive direction - Add residual/skip connections - Batch normalization

**Q. Exploding Gradients Gradients become too big, training is unstable.**

Possible solutions: -  Decrease the learning rate - Use saturating activation functions, like sigmoid or tanh - Gradient clipping - Batch normalization

**Q. Dead ReLUs When using ReLU as an activation function, a large gradient can knock off the weight of certain neurons so that they output 0 for all input data. They become useless.**

Decrease learning rate - Use Leaky ReLU, ELU or some other variant

**Q. What is the idea of backpropagation?**

Backpropagation is an algorithm used to train neural networks. It adjusts the weights of the network in a way that minimizes the difference between the predictions of the net and the ground truth labels - the loss function.

The math behind it is essentially taking every weight in the network and computing the partial derivative of the loss function for it. We can then adapt this weight according to the computed gradient and the learning rate - this is called Gradient Descend.

A naive implementation of this will be super slow - here comes backpropagation. We start from the last layer where we can directly compute the loss and the effect of every weight on it, so we can do the adaptation easily. Then we use the chain rule to move back to the previous layer and reuse the computations we did just before that to compute the gradients and update the weights. We continue until we reach the first trainable layer. Backpropagation avoids the redundant computation of the intermediate terms and therefore is very efficient. It is a great example of dynamic programming!

**Q. What is the learning rate? How do you choose it?**&#x20;

During the training of a neural network, we minimize a loss function using gradient descent. In each iteration, we adapt the weights by changing them a little in the direction of the gradient. How large this change is defined by the learning rate.

Imagine you are on the top of a mountain and want to get down to the lodge in the valley. The learning rate defines the speed at which you will be going down. A high speed means that you may breeze past the lodge and need to go back. Going too slow will cost you lots of time.

It is similar to the learning rate... If it is too high, the optimization will be unstable jumping around the optimum. It if is too low, the optimization will take too long to converge.&#x20;

So how to find the best learning rate:-&#x20;

As usual, hyperparameter tuning will require some trial and error. You will try different approaches on your training dataset and check how good it was on the validation set. You can use the following strategies:&#x20;

\- Parameter search - Schedule - Adaptive learning rate

Parameter search The easiest way is to just try different values and see how it goes. You should select them in an exponential manner: 1, 0.1, 0.01, 0.001... you get the idea...&#x20;

If the loss goes down, but slowly - increase&#x20;

If the loss starts oscillating - decrease&#x20;

Schedule, The schedule allows you to apply a decay of the learning rate. You start with a high rate to get close to the optimum fast, but decrease it after that to be more precise. The LR can be reduced based on the epoch or based on other criteria, like hitting a plateau.

Momentum, you can also apply momentum. Just like a ball rolling down a hill, the optimization will accelerate if it is steadily going down. Momentum will also help the optimizer jump over small local minima and find a better optimum.

Adaptive Learning Rate, there are also optimizers that adapt the learning rate automatically based on the gradients. They introduce other hyperparameters (e.g. the forgetting factor), but they are usually easier to tune.&#x20;

Prominent examples: - Adagrad - RMSProp - Adam



#### Classification

**What's the trade-off between bias and variance & Overfitting and Underfitting?**

Bias → error between average model predictions and the ground truth&#x20;

Variance → variability in the model predictions, how much a model can adjust on the given dataset.&#x20;

Underfitting → Error is high on training data itself and for test data too, accuracy going down for the training data. \[High Bias & Low Variance]&#x20;

Overfitting → Error is low on training data but very high on test data, accuracy going down for the test data. \[Low Bias & High Variance]&#x20;

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand, if our model has a large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.

Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.

A good model will give low bias and low variance&#x20;

Handling overfitting :

1. Cross-validation
2. Regularization&#x20;
3. Early Stopping
4. Pruning (in decision tree-based models)
5. Dropout (deep learning)

Handling Underfitting:

1. Get more training data
2. Augmentation (vision and nlp)
3. Increase the size or number of parameters in the model (no of neuron)&#x20;
4. Increasing the training time, until cost function is minimised.

**Why do ensembles typically have higher scores than individual models?**

An ensemble is the combination of multiple models to create a single prediction. The key idea for making better predictions is that the models should make different errors. That way the errors of one model will be compensated by the right guesses of the other models and thus the score of the ensemble will be higher.&#x20;

We need diverse models for creating an ensemble. Diversity can be achieved by:&#x20;

1. Using different ML algorithms. For example, you can combine logistic regression, k-nearest neighbors, and decision trees.&#x20;
2. Using different subsets of the data for training. This is called bagging.&#x20;
3. Giving a different weight to each of the samples of the training set. If this is done iteratively, weighting the samples according to the errors of the ensemble, it’s called boosting. Many winning solutions to data science competitions are ensembles. However, in real-life machine learning projects, engineers need to find a balance between execution time and accuracy.&#x20;

**What is an imbalanced dataset? Can you list some ways to deal with it?**

An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.

There are different options to deal with imbalanced datasets

* Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
* Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
* Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

**Precision, Recall, and F1-score**

Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances Precision = true positive / (true positive + false positive)

Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Recall = true positive / (true positive + false negative)

It is the weighted average of precision and recall. It considers both false positives and false negatives into account. It is used to measure the model’s performance.

F1-Score = 2 \* (precision \* recall) / (precision + recall)

**What’s the difference between Type I and Type II errors?**

Type I error is a false positive, while Type II error is a false negative. Briefly stated, a Type I error means claiming something has happened when it hasn’t, while a Type II error means that you claim nothing is happening when in fact something is. A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.

**What's the difference between boosting and bagging?**

Boosting and bagging are similar, in that they are both ensemble techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training.

**Explain Correlation and Covariance?**

Correlation is used for measuring and also for estimating the quantitative relationship between two variables. Correlation measures how strongly two variables are related. Examples like income and expenditure, demand and supply, etc.&#x20;

Covariance is a simple way to measure the correlation between two variables. The problem with covariance is that they are hard to compare without normalization.

**Can logistic regression be used for more than 2 classes?**

Yes, using one vs rest method.&#x20;

**How do you check the Normality of a dataset?**

Visually, we can use plots. A few of the normality checks are as follows:&#x20;

1. Shapiro-Wilk Test&#x20;
2. Anderson-Darling Test&#x20;
3. Martinez-Iglewicz Test
4. Kolmogorov-Smirnov Test&#x20;
5. D’Agostino Skewness Test

**How to Handle Outlier Values?**

An Outlier is an observation in the dataset that is far away from other observations in the dataset. Tools used to discover outliers are:

* Box plot&#x20;
* Z-score&#x20;
* Scatter plot, etc.&#x20;

Typically, we need to follow three simple strategies to handle outliers:&#x20;

* We can drop them.
* We can mark them as outliers and include them as a feature.&#x20;
* Likewise, we can transform the feature to reduce the effect of the outlier.

**What is Cross-Validation?**

Cross-validation is a method of splitting all your data into three parts: training, testing, and validation data. Data is split into k subsets, and the model has trained on k-1of those datasets. The last subset is held for testing. This is done for each of the subsets. This is k-fold cross-validation. Finally, the scores from all the k-folds are averaged to produce the final score.

**What are Support Vectors in SVM?**

A Support Vector Machine (SVM) is an algorithm that tries to fit a line (or plane or hyperplane) between the different classes that maximizes the distance from the line to the points of the classes. In this way, it tries to find a robust separation between the classes. The Support Vectors are the points of the edge of the dividing hyperplane as in the below figure.

**What are Different Kernels in SVM?**&#x20;

There are six types of kernels in SVM:

* Linear kernel - used when data is linearly separable.&#x20;
* Polynomial kernel - When you have discrete data that has no natural notion of smoothness.&#x20;
* Radial basis kernel - Create a decision boundary able to do a much better job of separating two classes than the linear kernel.&#x20;
* Sigmoid kernel - used as an activation function for neural networks.

**What is ‘Naive’ in a Naive Bayes?**&#x20;

The Naive Bayes method is a supervised learning algorithm, it is naive since it makes assumptions by applying Bayes’ theorem that all attributes are independent of each other.

Bayes’ theorem states the following relationship, given class variable y and dependent vector x1  through xn:

P(yi | x1,..., xn) =P(yi)P(x1,..., xn | yi)(P(x1,..., xn)

**When to use a Label Encoding vs. One Hot Encoding?**



#### Cluster Analysis<br>

1\. Cluster analysis does not classify variables as dependent or independent. - True,

2\. Cluster analysis is the obverse of factor analysis in that it reduces the number of objects, not the number of variables, by grouping them into a much smaller number of clusters. (True,

3\. If cluster analysis is used as a general data reduction tool, subsequent multivariate analysis can be conducted on the clusters rather than on the individual observations.(True

4\. The dendrogram is read from right to left. (False,

5\. Clustering should be done on samples of 300 or more. False,

6\. In cluster analysis, objects with larger distances between them are more similar to each other than are those at smaller distances. (False,

7\. The average linkage method of hierarchical clustering is preferred to the single and complete linkage methods.(True,

8\. The centroid method is a variance method of hierarchical clustering in which the distance between two clusters is the distance between their centroids (means for all the variables). (True,

9\. Nonhierarchical clustering is faster than hierarchical methods. (True,

10\. It is helpful to profile the clusters in terms of variables that were not used for clustering. (True,

11\. One method of assessing the reliability and validity of clustering is to use different methods of clustering and compare the results.(True,

12\. To reduce the number of variables, a large set of variables can often be replaced by the set of cluster components. (True,

13\. Which method of analysis does not classify variables as dependent or independent?

a. regression analysis b. discriminant analysis c. analysis of variance **d. cluster analysis**

**Q. How big should your batch size be? How do you choose it?**

There are reasons to use both larger and smaller batch sizes and you need to find the right \*balance\* for your dataset.

Why use larger batch sizes?&#x20;

Computing the gradients on more data leads to less noise that can be caused by outliers.&#x20;

Increase training speed, by avoiding the optimization jumping in different directions. Reduce oscillation of the loss function.

Why use \*smaller\* batch sizes?&#x20;

Your training data likely doesn't sample the problem space perfectly. You actually want some noise to avoid overfitting.&#x20;

Smaller batches act as a regularization mechanism.&#x20;

You can't fit all the data in the GPU memory.

**Q. What is the problem with unbalanced datasets? Can you give an example? How do you deal with it?**&#x20;

Real-world datasets are often imbalanced - some of the classes appear much more often in your data than others. The problem? Your ML model will likely learn to only predict the dominant classes.&#x20;

Example:&#x20;

We will be dealing with a ML model to detect traffic lights for a self-driving car. Traffic lights are small so you will have many more parts of the image that are not traffic lights. Furthermore, yellow lights are much rarer than green or red.

The problem is, Imagine we train a model to classify the color of the traffic light. A typical distribution will be:&#x20;

red- 56%&#x20;

yellow- 3%&#x20;

green- 41%&#x20;

So, your model can get to 97% accuracy just by learning to distinguish red from green. How can we deal with this?

Evaluation measures:  First, you need to start using a different evaluation measure other than the accuracy:&#x20;

\- Precision per class&#x20;

\- Recall per class -

\- F1 score per class&#x20;

I also like to look at the confusion matrix to get an overview. Always look at examples from the data as well!

In the traffic lights example above, we will see a very poor recall for yellow(most real examples were not recognized), while precision will likely be high. At the same time, the precision of green and red will be lower (yellow will be classified as green or red). The best thing you can do is to collect more data of the underrepresented classes. This may be hard or even impossible…

Balance your data: The idea is to resample your dataset so it is better balanced.

Undersampling - throw away some examples of the dominant classes, Even better, you can use some unsupervised clustering method and throw out only samples from the big clusters. The problem of course is that you are throwing out valuable data.

Oversampling - This is more difficult. You can just repeat the sample, but it won't work very well. You can use methods like SMOTE (Synthetic Minority Oversampling Technique) to generate new samples interpolating between existing ones. This may not be easy for complex images. If you are dealing with images, you can use data augmentation techniques to create new samples by modifying the existing ones (rotation, flipping, skewing, color filters...) You can also use GANs or simulations the synthesize completely new images.

Adapting your loss: Another strategy is to modify your loss function to penalize misclassification of the underrepresented classes more than the dominant ones. For the above example, we can set them like this(proportionally  to the distribution)

&#x20; Red    1.8

Yellow 33.3

Green  2.4

**Q9. 𝘞𝘩𝘢𝘵 𝘪𝘴 𝘵𝘩𝘦 𝘱-𝘷𝘢𝘭𝘶𝘦 and 𝘏𝘰𝘸 𝘸𝘰𝘶𝘭𝘥 𝘺𝘰𝘶 𝘦𝘹𝘱𝘭𝘢𝘪𝘯 𝘵𝘩𝘦 𝘱-𝘷𝘢𝘭𝘶𝘦 𝘵𝘰 𝘢 𝘯𝘰𝘯-𝘵𝘦𝘤𝘩𝘯𝘪𝘤𝘢𝘭 𝘴𝘵𝘢𝘬𝘦𝘩𝘰𝘭𝘥𝘦𝘳?**

The p-value is the probability of the observed statistic or more extreme given that the null hypothesis. And In statistics, you are looking for evidence that the null hypothesis, your baseline assumption, may not be valid. P-value is the probability of getting the observed value or more extreme given that the baseline model is assumed to be true. So, suppose that the baseline model is that the mean is 6’3’’ as the average height for basketball players. If you observe a value that’s 5’2’’, it’s in the extreme which is suggestive that the baseline model may not be correct.

**Q. 𝘈𝘯 𝘦𝘹𝘱𝘦𝘳𝘪𝘮𝘦𝘯𝘵 𝘴𝘩𝘰𝘸𝘦𝘥 𝘵𝘩𝘢𝘵 𝘢 𝘯𝘦𝘸 𝘷𝘦𝘳𝘴𝘪𝘰𝘯 𝘰𝘧 𝘢𝘯 𝘦𝘮𝘢𝘪𝘭 𝘤𝘢𝘮𝘱𝘢𝘪𝘨𝘯 𝘱𝘳𝘰𝘥𝘶𝘤𝘦𝘥 𝘢 𝘭𝘪𝘧𝘵 𝘪𝘯 𝘵𝘩𝘦 𝘶𝘴𝘦𝘳 𝘤𝘰𝘯𝘷𝘦𝘳𝘴𝘪𝘰𝘯. 𝘖𝘯𝘤𝘦 𝘵𝘩𝘦 𝘤𝘢𝘮𝘱𝘢𝘪𝘨𝘯 𝘸𝘢𝘴 𝘭𝘢𝘶𝘯𝘤𝘩𝘦𝘥 𝘰𝘯 100% 𝘰𝘧 𝘵𝘩𝘦 𝘶𝘴𝘦𝘳𝘴, 𝘵𝘩𝘦 𝘭𝘪𝘧𝘵 𝘴𝘩𝘪𝘧𝘵𝘦𝘥 𝘪𝘯 𝘵𝘩𝘦 𝘯𝘦𝘨𝘢𝘵𝘪𝘷𝘦 𝘥𝘪𝘳𝘦𝘤𝘵𝘪𝘰𝘯. 𝘞𝘩𝘺?**

To approach this question, the first thing to do is frame the question with clarifying questions. For instance, was the experimentation run on a subset of the population or slice in time that did not generalize well to the rest of the population?&#x20;

If the experimentation ran only in the U.S., then rolled out globally, what worked in the U.S. may not generalize well to other markets given cultural and language differences. Hence, the overall lift could shift in the direction. This is classic Simpson’s Paradox.

The other aspect is the timing of the campaign. Perhaps, the experimentation ran during a holiday that encourages spendings. But, post-launch could be a non-holiday period which could be different conditions for users. Once you call out a potential hypothesis to the interviewer, the next step is to briefly summarize how you would investigate the analysis and provide a recommendation.

You can say something along the lines of, “I will pull conversion data by GEO and look to see how the directions might differ at the global, continental, and country levels. The shift in the conversions will suggest that the treatment effects have different implications from one region to another. Therefore, this is suggestive that perhaps a different marketing strategy should be applied. Or, we should roll back the marketing campaign to only the subset of the regions like the U.S. that has been performing well.”

**What is Clustering?**

Clustering is the process of grouping a set of objects into a number of groups. Objects should be similar to one another within the same cluster and dissimilar to those in other clusters. A few types of clustering are:&#x20;

1. Hierarchical clustering&#x20;
2. K means clustering&#x20;
3. Density-based clustering&#x20;
4. Fuzzy clustering, etc.

**How can you select K for K-means Clustering?**

There are two kinds of methods that include direct methods and statistical testing methods:&#x20;

* Direct methods: It contains elbow and silhouette&#x20;
* Statistical testing methods: It has gap statistics.&#x20;

The silhouette is the most frequently used while determining the optimal value of k.



#### Deep Learning&#x20;



**Explain activation functions in deep learning**&#x20;

1. The sigmoid function is used for binary classification. The probabilities sum needs to be 1.&#x20;
2. The softmax function is used for multi-classification. The probabilities sum will be 1.
3. Hyperbolic Tangent Function (Tanh)
4. Rectified Linear Unit Function (Relu)&#x20;

<figure><img src="../.gitbook/assets/unknown (2).png" alt=""><figcaption></figcaption></figure>



**What is data normalization and why do we need it**

Data normalization is a very important preprocessing step, used to rescale values to fit in a specific range to assure better convergence during backpropagation. In general, it boils down to subtracting the mean of each data point and dividing by its standard deviation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function (if a higher-magnitude feature changes by 1%, then that change is pretty big, but for smaller features, it's quite insignificant). The data normalization makes all features weighted equally.

**Why do we use convolutions for images rather than just FC layers.**

Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as its own filter/feature detector.

**Why do we have max-pooling in classification CNNs?**

Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNN more translation in-variance.

**What is the significance of Residual Networks?**

The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

**What is batch normalization and why does it work?**

Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

**What is data augmentation? Can you give some examples? \[Image - Computer Vision]**&#x20;

Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or is changed in a known way.

Computer vision is one of the fields where data augmentation is very useful. There are many modifications that we can do to images:

* Resize&#x20;
* Horizontal or vertical flip&#x20;
* Rotate&#x20;
* Add noise&#x20;
* Deform&#x20;
* Modify colors Each problem needs a customized data augmentation pipeline

For NLP, we can use the back-translation technique for data augmentation.&#x20;

**What is a vanishing and exploding gradient?**

As we add more and more hidden layers, backpropagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

**What are dropouts**

Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where nature produces offspring by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

Epoch vs. Batch vs. Iteration.

* Epoch: one forward pass and one backward pass of all the training examples
* Batch: examples processed together in one pass (forward and backward)
* Iteration: number of training examples / Batch size

**What are Parametric and Non-Parametric Models?**

* Parametric models will have limited parameters and to predict new data, you only need to know the parameter of the model.&#x20;
* Non-Parametric models have no limits in taking a number of parameters, allowing for more flexibility and to predict new data. You need to know the state of the data and model parameters.

**What are the advantages of transfer learning?**

1. Better initial model: In other methods of learning, you must create a model from scratch. Transfer learning is a better starting point because it allows us to perform tasks at a higher level without having to know the details of the starting model.&#x20;
2. Higher learning rate: Because the problem has already been taught for a similar task, transfer learning allows for a faster learning rate during training.&#x20;
3. Higher accuracy after training: Transfer learning allows a deep learning model to converge at a higher performance level, resulting in more accurate output, thanks to a better starting point and higher learning rate.<br>

**Why would you use many small convolutional kernels such as 3x3 rather than a few large ones? \[**[**Kaggle Link**](https://www.kaggle.com/discussions/general/461216)**]**



