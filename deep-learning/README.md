# Deep Learning

> **Deep Learning** is a sub field of machine learning concerned with algorithms inspired by the structure and function of the brain called **artificial neural networks.**

![](../.gitbook/assets/deeplearning.webp)

**Feed Forward Neural Network:**

![](<../.gitbook/assets/image (59).png>)

This is a two layer neural network, One is hidden layer(having 3 neurons) and one is output layer(having 2 neurons). Feedforward refers to a unidirectional flow of information (and no lateral/intra-layer connections) from input to output.&#x20;

Predicted output is compared to the actual output during the training process, and the difference (loss) determines how much the weights should be tweaked. Tweaking the gradients and thereby the weights is done through **backpropagation**

* Output Layer: Represents the output of neural network ( each node correspond to the class)&#x20;
* Hidden Layer: Represents the intermediary nodes. it takes set of weighted input and produces output an activation function.
* Input Layer: Represents the dimension of the input vector, one node for each dimension. (In above diagram 3 inputs are there i.e. 3 nodes in input layer)

While training a neural network, keep these points in mind:

* Decide the structure of network
* &#x20;

Training Deep Learning isn't walk in the park, following problems can you can run into while training a DNN

1. You may faced Vanishing gradients or exploding gradient&#x20;
2. Model with million parameters would be risky to overfit.
3. Training maybe extremely slow.
4. You might not have enough training data for large network.

![Deep Neural Network](../.gitbook/assets/Deep-Neural-Network-architecture.png)

**Error or Loss Function :**

In general, error/loss is the difference between actual vs predicted values. And the goal is to minimize the loss while training a neural network, and we can calculate the error using Loss function and there are different kind of loss function, one should choose based on problem at hand. Loss function are different for Regression and Classification.&#x20;

{% hint style="info" %}
Check Loss Function section under Machine Learning for more in depth details. &#x20;
{% endhint %}

**Backpropagation**&#x20;

Backpropagation is used while training a feed forward neural network, it helps in efficiently calculating the gradient of loss function w.r.t weights and that helps in minimizing the loss by updating weights.&#x20;

At the end of each forward pass, we have a loss(difference between actual and predicted outcome). The core of backprop is a partial derivatives of loss w.r.t weights which tells us how quickly the loss changes for any change in weights. Backprop follows the rule of chain rule of derivatives, i.e. the loss can be computed for each and every weight in the network.&#x20;

To understand math behind backprop follow these two videos:

{% tabs %}
{% tab title="First Tab" %}
{% embed url="https://www.youtube.com/watch?v=mH9GBJ6og5A" %}
{% endtab %}

{% tab title="Second Tab" %}
{% embed url="https://www.youtube.com/watch?v=Ilg3gGewQ5U" %}
{% endtab %}
{% endtabs %}

**What if ?**

Initialize weights to 0, this makes your model equivalent to linear model and when you set weights to 0, the derivatives w.r.t to loss function is same for every w in every layer thus, all the weights have the same values in the subsequent iteration. This makes the hidden units symmetric and continues for all the n iterations you run. Thus setting weights to zero makes your network no better than a linear model.

Initializing weights randomly, while working with a (deep) network can potentially lead to 2 issues — vanishing gradients or exploding gradients.

**Vanishing / Exploding Gradients**&#x20;

![](../.gitbook/assets/1-_YRWJr-jF7tKnmUq-e3ltw.png)

As the backpropagation algorithm advances downwards(or backward) from the output layer towards the input layer, the gradients often get smaller and smaller and approach zero which eventually leaves the weights of the initial or lower layers nearly unchanged. As a result, the gradient descent never converges to the optimum. This is known as the **vanishing gradients** problem.

On the contrary, in some cases, the gradients keep on getting larger and larger as the backpropagation algorithm progresses. This, in turn, causes very large weight updates and causes the gradient descent to diverge. This is known as the **exploding gradients** problem.

Following are some signs that can indicate that our gradients are exploding/vanishing :

|                      **`Exploding`**                    |                                                             **`Vanishing`**                                                            |
| :-----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: |
| There is an exponential growth in the model parameters. | The parameters of the higher layers change significantly whereas the parameters of lower layers would not change much (or not at all). |
|    The model weights may become NaN during training.    |                                             The model weights may become 0 during training.                                            |
|        The model experiences avalanche learning.        |           The model learns very slowly and perhaps the training stagnates at a very early stage just after a few iterations.           |

By using following techniques we can fix these problems:-

* Proper weight initialization&#x20;
* Using Non-saturating Activation function such as ReLU, Leaky ReLU&#x20;
* Batch Normalization
* Gradient Clipping&#x20;

{% hint style="info" %}
To learn about Batch Normalization and other regularization technique, follow Regularization section under deep Learning.&#x20;
{% endhint %}

**Learning Rate**&#x20;

Learning rate is a hyperparameter which determines to what extent newly acquired weights overrides old weights. In general it lies between 0 and 1. Momentum is used to decide the weight on nodes from the previous iterations and it helps to improve training speed and also in avoiding local minima.&#x20;











#### List of few free courses on Deep Learning by Top University

| Title                                           | Link                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| Introduction to Deep Learning(6.S191) **MIT**   | [YouTube](https://tinyurl.com/y2jmc89y)                                           |
| Deep Learning **NYU**                           | [WebSite](https://atcold.github.io/pytorch-Deep-Learning/)                        |
| Deep Learning Lecture Series **DeepMind x UCL** | [YouTube](https://tinyurl.com/create.php)                                         |
| Deep Learning (CS230) **Stanford**              | [WebSite](https://cs230.stanford.edu/lecture/)                                    |
| CNN for Visual Recognition(CS231n) **Stanford** | [WebSite ](https://cs231n.github.io/) \| [YouTube](https://tinyurl.com/y2gghbvs)  |
|                                                 |                                                                                   |
