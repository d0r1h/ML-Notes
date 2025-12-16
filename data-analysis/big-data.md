---
description: Hadoop, Hive, Spark
---

# Big Data

**How to join datasets in Spark?**&#x20;

Generally, the interviewer wants to see if you how map-side join and reduce-side join. This is the best article about [Spark join](https://towardsdatascience.com/the-art-of-joining-in-spark-dcbd33d693c). If you have time you can read [Spark AQE](https://databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html) (Adaptive Query Execution, only available on Spark 3.0). The tldr version is Spark can automate parameters tuning for joining a big dataset. I never see any question about Spark AQE specifically since it’s relatively new.

**Finding common friends with MapReduce**

* This is the MOST common question for MapReduce coding questions. It’s also straightforward.
* Assume the friends are stored as Person-> \[List of Friends], our friends list can be visualized as this graph
* Mapper phase:
  * For map (A -> B C D) we generate (key, values) pairs and can be presented as this graph (we sorted the key in alphabetical orders so we can have a unique key).
  * For map (B -> A C D E) we generate (key, values) pairs and can be presented as this graph
* In the combiner/reducer phase we have this pair for the same key (AB): (A B) -> (A, C, D, E) (B, C, D). We simply take the common “friends” between (A, C, D, E) and (B, C, D) and yield (C, D) as common friends for A and B.

**How do you handle Spark OOM?**

* There are few areas to consider during parameters tuning. As any tuning practice we have to monitor the memory usage in the UI.
* Data Serialization: In network-intensive applications, try to use KryoSerializer instead of the default one.
* Increase parallelism level: either by increasing spark. Default. parallelism or create many partitions such that you end up with 2–3 tasks per CPU core in your cluster.
* Use the Broadcast variable when possible.
* Most interviewers will ask basic high level questions so if you faced Spark OOM before you might be able to answer. If you have time you can read this [Spark best practic](https://spark.apache.org/docs/latest/tuning.html)
