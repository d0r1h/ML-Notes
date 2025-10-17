---
description: Vector Database | RAG | LangChain |
icon: database
---

# Vector Database

#### Vector Embedding&#x20;

Vector embeddings are central to many NLP, recommendation, and search algorithms. Something special about vectors that makes them so useful. This representation makes it possible to translate[ semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity) as perceived by humans to proximity in a[ vector space](https://en.wikipedia.org/wiki/Vector_space).&#x20;

In other words, when we represent real-world objects and concepts such as images, audio recordings, news articles, user profiles, weather patterns, and political views as vector embeddings, the semantic similarity of these objects and concepts can be quantified by how close they are to each other as points in vector spaces. Vector embedding representations are thus suitable for common machine learning tasks such as clustering, recommendation, and classification.

For example, in a clustering task, clustering algorithms assign similar points to the same cluster while keeping points from different clusters as dissimilar as possible. In a recommendation task, when making recommendations for an unseen object, the recommender system would look for objects that are most similar to the object in question, as measured by their similarity as vector embeddings. In a classification task, we classify the label of an unseen object by the major vote over labels of the most similar objects.

Similarity search is one of the most popular uses of vector embeddings. Search algorithms like KNN and ANN require us to calculate distance between vectors to determine similarity. Vector embeddings can be used to calculate these distances. Nearest neighbor search in turn can be used for tasks like de-duplication, recommendations, anomaly detection, reverse image search, etc.

Searching through data for similar items is a common operation in databases, search engines, and many other applications. Finding similar items based on fixed numeric criteria is very straightforward using a query language when we are dealing with traditional databases. For example, finding employees in a database within a fixed salary range. But sometimes we have to answer questions like “Which objects in our inventory are similar to what a user searched for?” The search terms can be vague and can have a lot of variations. For example, a user can search for something generic like “shoes”,“black shoes” or something more precise like “Nike AF-1 LV8”

<figure><img src=".gitbook/assets/unknown (1).png" alt=""><figcaption></figcaption></figure>

Our system must be able to discern between these terms and must understand how a black shoe differs from other shoes. To handle such queries we need a representation that captures the deeper conceptual meaning of the objects. On top of that, in scenarios like these, we might have to work with data to the scale of billions of objects.

When dealing with data in this scale & context, this problem is quite unlike searching through traditional databases containing symbolic object representations. Hence we need something more powerful that can allow us to search through semantic representations efficiently. With similarity search, we can work with semantic representations of our data and find similar items fast. And in the sections below we will discuss how exactly it works.

We mentioned earlier that we find similarities between objects by calculating the distance between their vectors. We can calculate the distance between these vectors in the vector space according to the distance metric that fits our problem the best. Some of the commonly used distance metrics in ML are Euclidean, Manhattan, Cosine, and Chebyshev. The image below will help us understand the intuition behind each of these methods.

Now we know we can use vector embeddings to represent our objects, and the distances between vectors represent the similarity between the objects themselves. This is where the similarity search, or vector search, kicks in. Given a set of vectors and a query vector, we need to find the most similar items in our set for the query. We call this task the nearest neighbor search.

To reduce the computation complexity added by an exhaustive search like kNN we make use of approximate neighbor search (ANN).&#x20;

Instead of checking distances between each vector in the database, we retrieve a “good guess” of the nearest neighbor. In some use cases, we would rather lose some accuracy in favor of performance gain, thus allowing us to scale our search. ANN allows us to get a massive performance boost on similarity search when dealing with huge datasets. In approximately nearest neighbors (ANN), we build[ index structures](https://www.pinecone.io/learn/what-is-a-vector-index/) that narrow down the search space and improve lookup times. Apart from that, most ML models produce vectors that have high dimensionality which is another[ hurdle](https://en.wikipedia.org/wiki/Curse_of_dimensionality) to overcome. Approximate search relies on the fact that even though data is represented in a large number of dimensions, their actual complexity is low. It tries to work with the true intrinsic dimensionality of data. Hashing is a good example of a method that allows us to do it and is used widely for many applications.&#x20;

ANN algorithms make use of techniques like indexing, clustering, hashing, and quantization to significantly improve computation and storage at the cost of some loss in accuracy.



#### Vector Database&#x20;

A vector database indexes and stores vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, horizontal scaling, and serverless.

[What is a Vector Database & How Does it Work? Use Cases + Examples | Pinecone](https://www.pinecone.io/learn/vector-database/)&#x20;

