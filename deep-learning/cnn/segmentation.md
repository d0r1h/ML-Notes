# Segmentation

There are two way to perform Segmentation

1. Instance Segmentation
2. Semantic Segmentation

The instance segmentation combines object detection, where the goal is to classify individual objects and localize them using a bounding box, and semantic segmentation, where the goal is to classify each pixel into the given classes. In instance segmentation, we care about detection and segmentation of the instances of objects separately. Mask R-CNN is a state-of-art model for Instance segmentation. It extends Faster R-CNN, the model used for object detection, by adding a parallel branch for predicting segmentation masks.

Consider instance segmentation a refined version of semantic segmentation. Categories like “vehicles” are split into “cars,” “motorcycles,” “buses,” and so on — instance segmentation detects the instances of each category.

In Semantic segmentation for instance, a street scene would be segmented by “pedestrians,” “bikes,” “vehicles,” “sidewalks,” and so on.

In other words, semantic segmentation treats multiple objects within a single category as one entity. Instance segmentation, on the other hand, identifies individual objects within these categories.

### UNet

The UNet architecture has –

* An Encoder - Downsampling part. It is used to get context in the image. It is just a stack of convolutional and max pooling layers.
* A Decoder - Symmetric Upsampling part. It is used for precise localization. Transposed convolution is used for Upsampling.

It is a fully convolutional network (FCN). it has Convolutional layers and it does not have any dense layer so it can work for image of any size. A general representation of fully convolutional networks. The encoder is composed of convolutional and pooling layers for Downsampling and the decoder is composed of deconvolution layers for Upsampling.

The U-Net is an elegant architecture that solves most of the occurring issues. It uses the concept of fully convolutional networks for this approach. The intent of the U-Net is to capture both the features of the context as well as the localization. This process is completed successfully by the type of architecture built. The main idea of the implementation is to utilize successive contracting layers, which are immediately followed by the Upsampling operators for achieving higher resolution outputs on the input images.

Dice coefficient

Dice coefficient is defined as follows: X is the predicted set of pixels and Y is the ground truth. A higher dice coefficient is better. A dice coefficient of 1 can be achieved when there is perfect overlap between X and Y. Since the denominator is constant, the only way to maximize this metric is to increase overlap between X and Y.

$$
Dice = 2 | X ∩ Y | / (|X |+| Y |)  =  2 TP / (2 TP + FP + FN)
$$





Reference :&#x20;

[Quick intro to Instance segmentation: Mask R-CNN](https://kharshit.github.io/blog/2019/08/23/quick-intro-to-instance-segmentation)

