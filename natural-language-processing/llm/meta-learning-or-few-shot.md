---
description: Also know as Zero-Shot Learning or Learning with Less data
---

# Meta-Learning or Few-Shot

1. Few-shot Natural Language Processing
2. Meta-Learning for Natural Language Processing&#x20;

The motivation for meta-learning comes from being able to learn from small amounts of data. In general DNN/ these techniques critically rely on large datasets, but when we don't have enough data for the task at hand then what we'll do?

Meta-learning offers a potential solution to these problems: by learning to learn across data from many previous tasks. Few-shot meta-learning algorithms can discover the structure among tasks to enable fast learning of new tasks.

The idea of meta-learning is that given data/experience on previous tasks, the model learns new tasks quickly and efficiently. In order words, the goal is to train a model on a variety of tasks with rich annotations, so that it can solve a new task with very few labeled data.

The model's initial parameters are trained in such a way that it has maximal performance on a new task after the parameters have been updated through zero or a couple of gradient steps.

Metric-based meta-learning → The metric-based method learns a distance function between data points to classify test instances by comparing them to K-labeled examples. If the distance function is learned well on the training tasks, it can work well on the target task without fine-tuning.

1. Siamese Network-
2. Matching Network-
3. Prototypical Network-
4. Relation Network-

Optimization-based meta-learning → These methods include parameter initialization for a neural model with a few steps of gradient descent. In order to achieve good performance on the validation of each training task, the meta-learning model uses a two-step procedure of using validation error as the optimization loss.

1. MAML (Model Agnostic Meta-Learning)
2. FOMAML- First Order MAML
3. Reptile

**NLP problems in meta-learning can be of two categories :**

1. Meta-learning on different domains of the same problem. This category usually has access to different domains of datasets that essentially belong to the same problem, such as different domains of datasets for sentiment classification, different domains of datasets for intent classification.
2. Meta-learning on diverse problems and then it is applied to solve a new problem.

