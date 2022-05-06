---
description: Code and Guide for Deep Learning in Tensorflow & Keras
---

# Tensorflow

## Tensor

A tensor is multi-dimensional array. Similar to NumPy `ndarray` object, `tf.Tensor`   objects have data type and shape. Additionally `tf.tensor` can  reside in accelerator memory like (GPU). Tensorflow library offers operations           (`tf.add, tf.matmul, tf.linalg.inv` etc.) that consume and produce `tf.tensor`.  For example:&#x20;

```python
import tensorflow as tf 

tf.add(1,2)
tf.add([1,2], [3,4])
tf.square(5)
tf.reduce_sum([1,2,3])

tf.square(3) + tf.square(5)  # operator overloading 
```

The obvious difference between NumPy array and `tf.tensor` are :

* Tensor can be backed by accelerated memory like GPU and TPU
* Tensor are immutable





Following Notebook Contains implementation of FCNN. (Dense Network)
