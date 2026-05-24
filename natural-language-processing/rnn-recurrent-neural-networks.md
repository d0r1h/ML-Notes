---
description: Deep Learning models based on RNNs
---

# RNN \[Recurrent Neural Networks]

### **RNN**&#x20;

**Sequence** **Data** :- In this elements in a sequence appear in a certain order and are not independent of each other. For example :- Predicting the market value of a particular stock, For instance, assume we have a sample of n training examples, where each training example represents the market value of a certain stock on a particular day. If our task is to predict the stock market value for the next three days, it would make sense to consider the previous stock prices in a date-sorted order to derive trends rather than utilize these training examples in a randomized order.

Time series data is a special type of sequential data where each example is associated with a dimension for time. In time series data, samples are taken at successive timestamps, and therefore, the time dimension determines the order among the data points. For example, stock prices and voice or speech records are time series data.

On the other hand, not all sequential data has the time dimension. For example, in text data or DNA sequences, the examples are ordered, but text or DNA does not qualify as time series data.

We’ve established that order among data points is important in sequential data, so we next need to find a way to leverage this ordering information in a machine learning model. We can represent the sequential data as (x1, x2, x3 … xT).

The standard NN models, such as multi-layer perceptrons (MLPs) and CNNs for image data, assume that the training examples are independent of each other and thus do not incorporate ordering information. We can say that such models do not have a memory of previously seen training examples. For instance, the samples are passed through the feedforward and back-propagation steps, and the weights are updated independently of the order in which the training examples are processed.

RNNs, by contrast, are designed for modeling sequences and are capable of remembering past information and processing new events accordingly, which is a clear advantage when working with sequence data.

A recurrent neural network is a deep neural network that can process sequential data by maintaining an internal memory, allowing it to keep track of past inputs to generate outputs. The “recurrent” in “recurrent neural network” refers to how the model combines information from past inputs with current inputs. Information from old inputs is stored in a kind of internal memory, called a “hidden state.” It recurs feeding previous computations back into itself to create a continuous flow of information. \
Suppose we wanted to use an RNN to detect the sentiment (either positive or negative) of the sentence “He ate the pie happily.” The RNN would process the word he, update its hidden state to incorporate that word, and then move on to ate, combine that with what it learned from he, and so on with each word until the sentence is done. To put it in perspective, a human reading this sentence would update their understanding with every word. Once they’ve read and understood the whole sentence, the human can say the sentence is positive or negative. This human process of understanding is what the hidden state tries to approximate.

**Different Categories of Sequence Modeling**

Sequence modeling has many fascinating applications, such as language translation, image captioning, and text generation. However, in order to choose an appropriate architecture and approach, we have to understand and be able to distinguish between these different sequence modeling tasks.

1. **Many-to-one:** The input data is a sequence, but the output is a fixed-size vector or scalar, not a sequence. For example, in sentiment analysis, the input is text-based (for example, a movie review) and the output is a class label (for example, a label denoting whether a reviewer liked the movie). Classification.&#x20;
2. **One-to-many:** The input data is in standard format and not a sequence, but the output is a sequence. An example of this category is image captioning the input is an image and the output is an English phrase summarizing the content of that image or Music Generation&#x20;
3. **Many-to-many:** Both the input and output arrays are sequences. This category can be further divided based on whether the input and output are synchronized. An example of a synchronized many-to-many modeling task is video classification, where each frame in a video is labeled. Or NER (Name Entity Recognition)&#x20;

An example of a delayed many-to-many modeling task would be translating one language into another. For instance, an entire English sentence must be read and processed by a machine before its translation into German is produced

The dataflow of a standard feedforward NN and an RNN

Both of these networks have the input layer (x), hidden layer (h), and output layer (o) are vectors that contain many units.

![](<../.gitbook/assets/unknown (14).png>)

In a standard feed forward network, information flows from the input to the hidden layer, and then from the hidden layer to the output layer. On the other hand, in an RNN, the hidden layer receives its input from both the input layer of the current time step and the hidden layer from the previous time step. The flow of information in adjacent time steps in the hidden layer allows the network to have a memory of past events. This flow of information is usually displayed as a loop, also known as a recurrent edge in graph notation, which is how this general RNN architecture got its name.

Similar to multilayer perceptrons, RNNs can consist of multiple hidden layers. Note that it’s a common convention to refer to RNNs with one hidden layer as a single-layer RNN, which is not to be confused with single-layer NNs without a hidden layer, logistic regression.

As we know, each hidden unit in a standard NN receives only one input, the net pre activation associated with the input layer. In contrast, each hidden unit in an RNN receives two distinct sets of input: the preactivation from the input layer and the activation of the same hidden layer from the previous time step, t-1.

At the first time step, t = 0, the hidden units are initialized to zeros or small random values. Then, at a time step where t > 0, the hidden units receive their input from the data point at the current time, x<sup>(t)</sup>, and the previous values of hidden units at t–1, indicated as h<sup>(t–1)</sup>.

**Computing activation in RNN**

Now that we understand the structure and general flow of information in an RNN, let’s get more specific and compute the actual activations of the hidden layers, as well as the output layer.&#x20;

Each directed edge (the connections between boxes) in the representation of an RNN that we just looked at is associated with a weight matrix. Those weights do not depend on time, t; therefore, they are shared across the time axis. The different weight matrices in a single-layer RNN are as follow :

W<sub>hx</sub> : The weight matrix between the input, x(t), and the hidden layer, h

W<sub>hh</sub> : The weight matrix associated with the recurrent edge

W<sub>yh</sub> : The weight matrix between the hidden layer and output layer

<img src="../.gitbook/assets/unknown (15).png" alt="" height="391" width="775">

Computing the activations is very similar to standard multilayer perceptrons and other types of feedforward NNs. For the hidden layer, the net input, zh (preactivation), is computed through a linear combination; that is, we compute the sum of the multiplications of the weight matrices with the corresponding vectors and add the bias unit:

Z<sub>h</sub><sup>(t)</sup> =&#x20;

h<sup><sub>(t)<sub></sup>   =&#x20;

h<sub>(t)</sub>  =

o<sub>(t)</sub>  =

<img src="../.gitbook/assets/unknown (17).png" alt="" height="381" width="808">

<img src="../.gitbook/assets/unknown (18).png" alt="Data Flow in RNN" height="219" width="630">

This RNN (BPTT-Backpropagation Through Time) introduces some new challenges. Because of the multiplicative factor 𝜕h(t) / 𝜕h(k )  in computing the gradients of a loss function, the so-called vanishing and exploding gradient problems arise.

Training an RNN is done by defining a loss function (L) that measures the error between the true label and the output, and minimizes it by using forward pass and backward pass. \
For a single time step, the following procedure is done: first, the input arrives, then it processes through a hidden layer/state, and the estimated label is calculated. In this phase, the loss function is computed to evaluate the difference between the true label and the estimated label. The total loss function, L, is computed, and by that, the forward pass is finished. The second part is the backward pass, where the various derivatives are calculated.

As we backpropagate gradients through layers and also through time. Hence, in each time step we have to sum up all the previous contributions until the current one and due to this vanishing and exploding gradient problem arises.

There are at least three solutions to this problem:

* Gradient clipping
* Truncated backpropagation through time (TBPTT)
* LSTM

Using gradient clipping, we specify a cut-off or threshold value for the gradients, and we assign this cut-off value to gradient values that exceed this value. In contrast, TBPTT simply limits the number of time steps that the signal can backpropagate after each forward pass. For example, even if the sequence has 100 elements or steps, we may only backpropagate the most recent 20 time steps.

While both gradient clipping and TBPTT can solve the exploding gradient problem, the truncation limits the number of steps that the gradient can effectively flow back and properly update the weights.

### LSTM (Long short-term memory cell)

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in “the clouds are in the sky,” we don’t need any further context – it’s pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.

But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.

In standard RNNs, the hidden state is heavily weighted toward recent parts of the input. In an input that’s thousands of words long, the RNN will forget important details from the opening sentences.&#x20;

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.&#x20;

LSTMs have a special architecture to get around this forgetting problem. They have modules that pick and choose which information to explicitly remember and forget. So recent but useless information will be forgotten, while old but relevant information will be retained.

![](<../.gitbook/assets/unknown (19).png>)

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn.

The building block of an LSTM is a memory cell, which essentially represents or replaces the hidden layer of standard RNNs. In each memory cell, there is a recurrent edge that has the desirable weight, w = 1, to overcome the vanishing and exploding gradient problems. The values associated with this recurrent edge are collectively called the cell state.

Notice that the cell state from the previous time step, C(t–1), is modified to get the cell state at the current time step, C(t), without being multiplied directly by any weight factor. The flow of information in this memory cell is controlled by several computation units (often called gates). ⨀ refers to the element-wise product (element-wise multiplication) and ⨁ means element-wise summation (element-wise addition). Furthermore, x<sup>(t)</sup> refers to the input data at time t, and h(t–1) indicates the hidden units at time t–1.&#x20;

\
Four boxes are indicated with an activation function, either the sigmoid function (𝜎) or tanh, and a set of weights; these boxes apply a linear combination by performing matrix-vector multiplications on their inputs (which are h<sup>(t–1)</sup> and x(<sup>t)</sup>). These units of computation with sigmoid activation functions, whose output units are passed through ⨀ , are called gates.

In an LSTM cell, there are three different types of gates, which are known as the forget gate, the input gate, and the output gate:<br>

1. The forget gate (ft) allows the memory cell to reset the cell state without growing indefinitely. In fact, the forget gate decides which information is allowed to go through and which information to suppress.
2. The input gate (it) and candidate value (𝑪𝑡 ) are responsible for updating the cell state
3. The output gate (ot) decides how to update the values of hidden units
4. Given this, the hidden units at the current time step are computed as follows: h<sub>t</sub>

The key to LSTMs is the cell state, The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”

An LSTM has three of these gates, to protect and control the cell state

**Walkthrough LSTM :-**&#x20;

1. The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1 and xt , and outputs a number between 0 and 1 for each number in the cell state Ct−1. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”
2. The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, Ct,  that could be added to the state. In the next step, we’ll combine these two to create an update to the state.
3. It’s now time to update the old cell state, Ct−1, into the new cell state Ct.  The previous steps already decided what to do, we just need to actually do it. We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it ∗Ct. This is the new candidate values, scaled by how much we decided to update each state value. In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.











