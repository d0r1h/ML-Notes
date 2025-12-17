# Classification











### One Shot Learning for Vision

Traditional Classification task the input image is fed into a series of layers, and finally a probability distribution over all the classes (typically using a Softmax activation function).

Two important points to be noted here

* Require large number of images for each class
* If the network is trained only on, let’s say, 3 classes of images, then we cannot expect to test it on any other class.

If we want our model to classify the images of other classes as well, then we need to first get a lot of images for that particular class and then we must re-train the model again.

For some applications in the real world, we neither have large enough data for each class and the total number of classes is huge and it keeps on changing. The cost and effort of data collection and periodical re -training is too high. And to overcome this we use Few/One Shot learning.

**Metric Learning**&#x20;

Metric is like a distance. It follows the following properties -

* Inverse of similarity
* It is symmetric
* It follows triangle inequality

Metric learning is the task of learning a distance function over objects.

If distance is considered, the objective is to minimize the distance measure such as Euclidean and Manhattan distance. If considering similarity, the objective is to maximize the similarity measure such as Dot product, RBF.

We can use Siamese Network as Metric Learning&#x20;

The Siamese network is used to find how similar two things are.  Some examples of such cases are Verification of signature, face recognition Any Siamese network has two identical subnetworks, which share common parameters and weights. Siamese neural networks has a unique structure to naturally rank similarity between inputs.

Application of Siamese networks are Signature verification, face verification, paraphrase scoring etc.&#x20;

Distance Example – L2 norm of difference(Euclidean) and L1 norm of difference (Manhattan)&#x20;

Similarity Example – Dot Product, arc cosine, radial basis function (RBF)

**Triplet Loss**

You can train the network by taking an anchor image and comparing it with both a positive sample and a negative sample. The dissimilarity between the anchor image and positive image must be low and the dissimilarity between the anchor image and the negative image must be high.

By using this loss function, we calculate the gradients and with the help of the gradients, we update the weights and biases of the Siamese network.

&#x20;                                                          Loss = max(d(a,p) – d(a,n) + margin, 0)

* “a” - represents the anchor image &#x20;
* “p” - represents a positive image &#x20;
* “n” - represents a negative image
* margin - is a hyperparameter. It defines how far away the dissimilarities should be.



Reference&#x20;

{% embed url="https://belvederef.github.io/cv-notebook/computer-vision-theory/one-shot-learning.html#triplet-loss" %}

