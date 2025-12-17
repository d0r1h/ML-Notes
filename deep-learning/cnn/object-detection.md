# Object Detection

Object detection is the task of detecting all the objects in the image that belong to a specific class and give their location. An image can contain more than one object with different classes.&#x20;

There are few challenges in the object detection task&#x20;

* Two tasks – classification and localization&#x20;
* Results/prediction take lot of time but we need fast predictions for real time task
* Variable number of boxes as output
* Different scales and aspect ratios
* Limited data and labelled data
* Imbalanced data-classes

Performance metrics for object detection&#x20;

1. Intersection over union (IOU)
2. Precision and recall
3. Mean average precision (mAP)

IoU is a function used to evaluate the object detection algorithm, it computes size of intersection and divides it by union, more generally IoU is a measure of the overlap between two bounding boxes i.e. predicted output box and labelled box. If IoU is more than 0.5 then it’s kind of good and the best answer will be 1.&#x20;

We use IoU for object detection because the more overlap predicted bounding boxes have with the ground truth bounding boxes the better(higher) their IoU scores will be. mAP

Here are the steps for calculating mAP

1. Sort Predictions according to confidence (usually classifier’s output after softmax)&#x20;
2. Calculate IoU of every predicted box with every ground truth box
3. Match predictions to ground truth using IoU, correct predictions are those with IoU > threshold&#x20;
4. Calculate precision and recall at every row(X)
5. Take the mean of maximum precision at X+1 recall values to get AP

Average across all classes to get the mAP.

Object Detection approaches

1. Brute force Approach&#x20;
2. Sliding window approach
3. State of Art
4. One Stage methods
5. Two stage methods

In brute force approach, we run a classifier for every possible box, for example we take an image with 15\*10 grid, there are 150 small boxes, doing this is very computationally expensive.&#x20;

In the sliding window approach, we run a classifier in sliding window fashion, then we apply a CNN to many different crops(windows) of the image and classify each crop as object or background. And the problem with this approach is that we need to apply CNN to a huge number of locations (and scales) which makes it very computationally expensive.&#x20;

Another best approach would be to reduce the number of boxes but HOW? We can find regions in the image which are likely to contain objects and run a classifier for region proposals likely to contain objects.&#x20;

**Region Proposal**&#x20;

Find “blobby” image regions that are likely to contain objects, relatively fast to run, for example Selective search gives 1000 region proposals in a few seconds on CPU.&#x20;

Selective Search: -- uses the best of both worlds, Exhaustive search and segmentation. Segmentation improves the sampling process of different boxes i.e. reduces considerably the search space. Selective search approach produces boxes that are good proposals for objects, it handles different image conditions, but most importantly it is fast enough to be used in a prediction pipeline (like Fast RCNN) to do real time object detection. Following are advantages of selective search&#x20;

* Capture All Scales - Objects can occur at any scale within the image.  Furthermore, some objects may not have clear boundaries. This is achieved by using a hierarchical algorithm.
* Diversification - Regions may form an object because of only color, only texture, or lighting conditions etc. Therefore, instead of a single strategy which works well in most cases, we prefer to have a diverse set of strategies to deal with all cases.
* Fast to compute

### Region (R) – CNN

R-CNN is an algorithm which can also be used for object detection. R-CNN stands for regions with Conv Nets. It tries to pick a few windows and run a Conv net (your confident classifier) on top of them.

R-CNN, first generates 2K region proposals (bounding box candidates), then detect object within each region proposal as below:

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

The algorithm R-C NN uses to pick windows is called a segmentation algorithm. And the drawback of RCNN is that it uses separate ConvNet for each box and thus it becomes slow. &#x20;

And to overcome this we use Fast R-CNN, in which whole image is forward through ConvNet and then we do feature maps and region proposal and to select

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>





**Region-of-Interest(RoI) Pooling:**

It is a type of pooling layer which performs max pooling on inputs (here, ConvNet feature maps) of non-uniform sizes and produces a small feature map of fixed size (say 7x7). The choice of this fixed size is a network hyper-parameter and is predefined.

The main purpose of doing such a pooling is to speed up the training and test time and also to train the whole system from end-to-end (in a joint manner).

It's because of the usage of this pooling layer the training & test time is faster compared to the original(vanilla?) R-CNN architecture and hence the name Fast R-CNN.

And Faster RCNN architecture contains 2 networks: 1. Region Proposal Network (RPN) and 2. Object Detection Network.&#x20;

All of the above object detection algorithms (RCNN, Fast RCNN, Faster RCNN) are based on region proposals. But there are algorithms which detect without proposal. YOLO and SSD.&#x20;

YOLO&#x20;

YOLO – You only look once, look at the image just once but in a clever way.  “We reframe the object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities”

Algorithmic flow:

* Actually Divides the image into a grid of say, 13\*13 cells (S=13)
* Each of these cells is responsible for predicting 5 bounding boxes (B=5) (A bounding box describes the rectangle that encloses an object)
* YOLO for each bounding box
* outputs a confidence score that tells us how good is the shape of the box
* the cell also predicts a class
* The confidence score of the bounding box and class prediction are combined into the final score -> probability that this bounding box contains a specific object.

**SSD (Single Shot Detection)**

In SSD, like YOLO, only one single pass is needed to detect multiple objects within the image. Two passes are needed in Regional proposal network (RPN) based approaches such as R-CNN, Fast R-CNN series. One pass for generating region proposals and another pass to loop over the proposals for detecting the object of each proposal.

SSD is much faster compared to two-shot RPN-based approaches. A feature layer of size m×n (number of locations) with p channels, for each location, we got k bounding boxes, and for each of the bounding box, we will compute c class scores and 4 offsets relative to the original default bounding box shape.

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

In the above diagram,

The SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 × 300 input size significantly outperforms its 448 × 448. YOLO counterpart in accuracy on VOC2007 test while also improving the run-  time speed, albeit YOLO customized network is faster than VGG16.

