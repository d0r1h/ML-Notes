---
description: >-
  In Machine learning clustering is a technique of dividing the data into
  groups(clusters) based on similarity in them.
---

# Clustering

Types of Clustering

* Density Based: Clusters are formed based on the density of observations(DBSCAN). &#x20;
* Hierarchical Based: Clusters are formed based on the distance b/w the observations.
* Graph Based:&#x20;
* Partition Based: Observations are partitioned into the predetermined number of clusters based on distance(k-means clustering).
* Model Based&#x20;

A similarity measure for two objects returns the value between 0 and 1, for the completely similar objects the similarity measure is 1.

A dissimilarity measure for two objects returns the value between 0 and 1, for the completely similar objects the similarity measure is 0.

## k-means Algorithm

Used when data is numeric, it is recursive technique works based on proximity measures. It is a greedy algorithm also known as a non-deterministic algorithm.&#x20;

* Choose a distance measure and value of K
* Randomly choose k points as cluster centroid
* Assign the nearest cluster centroid to each data point
* Compute the means of clusters
* Reassign the cluster based on the means&#x20;
* Repeat the above steps, until the cluster means do not change.

In the case of categorical data use k-modes clustering and for mixed(numerical and categorical) data type use k-prototypes algorithm.

To find the optimal value of k, we use two methods Elbow plot and Silhouette Method.

### Elbow plot & Silhouette Score

K-means clustering aims to reduce the within-cluster variation, the elbow (or scree plot) is used to plot the `within-cluster sum of squares(WCSS)` for different values of K. Optimal K is the value corresponding to the elbow point. &#x20;

![](<../../.gitbook/assets/image (54).png>)

Higher the value of WCSS, higher is the variation within the cluster. Lower the value of WCSS, lower is the variation within the cluster.&#x20;

The plot shows that the WCSS is decreasing rapidly for the k value less than the optimal K value, after the elbow point, the wcss is steadily decreasing which implies that more clusters are formed by dividing the large cluster into subgroups. Selecting the k greater than optimal k leads to overfitting.&#x20;

```python
<!--In sklearn, the attribute `inertia_` returns the WCSS for 
  specific value of k  -->
  
  wcss = []
  
  for i in range(1,10):
        model = KMeans(n_clusters=i, random_state=4)
        model.fit(data)
        wcss.append(model.inertia_)
  
  plt.plot(k, wcss)
```

Variation with the cluster is maximum as all the observations are grouped into a single cluster, and our aim is to reduce this variation.&#x20;

{% hint style="success" %}
As per the objective of cluster analysis, the variation within the cluster should be minimum, and variation between clusters should be maximum.
{% endhint %}

Silhouette score is used to find the optimal number of clusters, it is the mean silhouette coefficient over all the instances, the value of the silhouette score lies between -1 to +1. We plot the silhouette score for different values of k and select the k with the highest score. Also used to validate the quality of cluster. Silhouette coefficient near to +1 indicates that the observation is well set inside its own cluster, and far from the cluster and close to -1 indicates that the observation has been assigned to the wrong cluster.&#x20;

There are several criteria to choose the optimal of k using silhouette score --                                     &#x20;

* Select the value of k such that there are no outliers in each cluster.
* Select the value of k for which all the silhouette coefficient is greater than the average silhouette score.&#x20;
* Select the value of K that has the highest average silhouette score.&#x20;

```python
from sklearn.metrics import silhouette_score

score  = []

for i in range(1,10):
    cluster = KMeans(n_clusters=i, random_state=4)
    predict = cluster.fit_predict(data)
    
    score.append(silhouette_score(data, predict))
```

## Hierarchical Clustering

As names suggest it is a hierarchy-based clustering method. In this no need to pre-define the number of clusters. There are two types of hierarchical clustering methods.

### Agglomerative clustering

It's the most popular hierarchical clustering method, it considers the bottom to top approach. The similar observations are clustered together to form a bigger cluster, considering each observation as a unique cluster in the initial step, and the process continues till all the observations are fused in a single cluster. **A dendrogram** is used to visualize such cluster formation.&#x20;

* Consider each observation as a unique cluster
* Calculate the pairwise distance between all the cluster
* Combine the two nearest clusters into a single cluster
* Calculate the distance between newly formed clusters and remaining clusters
* Repeat the above steps, until a single cluster is formed

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering()
model.fit(data)  
```

As I mentioned earlier, in the initial step of clustering each observation is considered as a cluster, and to calculate the pairwise distance between all these observations we use a distance matrix such as Manhattan, Euclidean, Minkowski, and so on to find the two closest cluster.&#x20;

Once we found the first two closest points(x,y) and form a cluster using them, we have to update the distance in the distance matrix, and the distance between the ungrouped clusters/elements will remain the same.  So now we have to calculate the distance between the ungrouped clusters and newly formed clusters (x,y), and to calculate this distance we use different **linkage methods.**&#x20;

Similarity between the clusters(inter-cluster distance)  can be measured using various types of linkage methods-

* Single **** linkage
* Complete linkage
* Average linkage
* Centroid linkage
* Ward Linkage

```python
from scipy.cluster.hierarchy import linkage

link = linkage(data, method='single')
print(link)
```

#### Dendrogram

Dendrogram is a technique to visualize the clusters, it is a tree-based hierarchical structure that can be used to decide the required number of clusters. Different linkage methods result in the formation of different dendrograms. Observations linked at a low height represent more similar observations, and dissimilar observations fuse at a higher level in the dendrogram.&#x20;

![](<../../.gitbook/assets/image (56).png>)

![](<../../.gitbook/assets/image (55).png>)

X-axis of the dendrogram represents the data point, each considered as a single cluster and the distance is given on the Y-axis. Each single cluster is known as ‘leaf’. The horizontal line is known as ‘clade’ which represents the merging of clusters.&#x20;

X-axis of the dendrogram represents the data point, each considered as a single cluster and the distance is given on the Y-axis and every single cluster is known as ‘leaf’. The horizontal line is known as ‘clade’ which represents the merging of clusters.&#x20;

**Cophenetic correlation coefficient**

****

**Cophenetic correlation coefficient:** Quantifies how the dendrogram has represented the dissimilarities between the observations. It is defined as the correlation coefficient between cophenetic distances and the actual distance between the observations. The value close to 1 represents the best linkage quality.

```python
from spicy.cluster.hierarchy import dendogram, cophenet

coeff, cophnet_dist = cophenet(linkage_matrix, actual_dist)
print(coeff)

dendrogram(linkage_matrix)
plt.show()
```



### Divisive Clustering  &#x20;





## DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is mostly used density-based clustering algorithm, use to form the clusters of non-linear shapes. It considers a cluster as a continuous region of high density and region of low density are identified as noise or outliers.&#x20;

There are mainly two parameter in the DBSCAN algorithm: epsilon(ε) and the minimum numbers of samples (min\_samples or minPts). Epsilon is the radius of neighborhood for a data point. Minimum number of samples is the lower bound for the count of data points in the neighborhood of a core point.&#x20;

* **A core point** is a data point which has at least a minimum number of samples in its ε- neighborhood (including itself) or otherwise; it is a non-core point.
* **A data point** is ‘directly density reachable’ from a core point if it is in the ε- neighborhood of a core point.
* **A border point** is a point which is not a core point but, it is directly density reachable from a core point.&#x20;

![](<../../.gitbook/assets/image (57).png>)

* Decide the parameters ε and min\_samples
* Choose a starting point(p) randomly and find its ε - neighborhood.
* If p is a core point, find all density-reachable points form P and form a cluster else mark P as a noise point.
* Find next unvisited point and follow the same steps as P
* Repeat this procedure till all the points are marked visited

```python
from sklean.cluster import DBSCAN

model = DBSCAN(eps=eps_radius, min_samples=m)
model.fit_predict(data)
```

![](<../../.gitbook/assets/image (58).png>)



## Dimension Reduction Technique

### Curse of dimensionality

