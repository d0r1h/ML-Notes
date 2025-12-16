---
description: Activation Function in Deep Learning
---

# Activation Function

An [activation function](https://en.wikipedia.org/wiki/Activation_function) in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.  A network may have three types of layers: input layers that take raw input from the domain, hidden layers that take input from another layer and pass output to another layer, and output layers that make a prediction.

The ANN activation function helps in defining the output of a node when an input is given. The activation function takes in the output of the previous cell and converts it into some form that can be taken as input to the next cell. Different types of activation functions help in different tasks.

<figure><img src="../.gitbook/assets/Activation Functions A Short Summary - Hyunjulie - Medium" alt=""><figcaption></figcaption></figure>

In artificial neural network, activation function helps in defining  the output of a node when a input is given.  Different types of activation function helps in different task, some example are -&#x20;

* Sigmoid
* Tanh
* ReLU
* Binary Step Function&#x20;

![](../.gitbook/assets/1-BMSfafFNEpqGFCNU4smPkg.png)

The choice of activation function in the hidden layer will control how well the network model learns the training dataset. The modern default activation function for hidden layers is the ReLU function.

The choice of activation function in the output layer will define the type of predictions the model can make. The activation function for output layers depends on the type of prediction problem.

All hidden layers typically use the same activation function. The output layer will typically use a different activation function from the hidden layers and is dependent upon the type of prediction required by the model.&#x20;

**Activation Function for Hidden Layers:**

Typically, a differentiable nonlinear activation function is used in the hidden layers of a neural network. This allows the model to learn more complex functions than a network trained using a linear activation function. There are perhaps three activation functions you may want to consider for use in hidden layers; they are:&#x20;

* Rectified Linear Activation (ReLU)
* Logistic (Sigmoid)
* Hyperbolic Tangent (Tanh)

ReLU is the most common activation function used for Hidden layers, it is less susceptible to vanishing gradient that prevent deep learning models being trained, although it can suffer other problems such as saturated or dead units.&#x20;

It can be calculated as max(0.0,x), i.e. if input value is negative then 0.0 is returned, otherwise, the value is returned. When using the ReLU function for hidden layers, it is a good practice to use a “He Normal” or “He Uniform” weight initialization and scale input data to the range 0-1 (normalize) prior to training.

The sigmoid activation function is also called the logistic function. The function takes any real value as input and outputs values in the range 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0.

#### **Guide for choosing activation function.**

![](<../.gitbook/assets/Deep Learning Activation Function.png>)

Reference :-&#x20;

1. [https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)
2. [https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
