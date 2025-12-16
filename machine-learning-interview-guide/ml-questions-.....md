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





