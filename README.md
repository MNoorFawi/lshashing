# lshashing

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://mnoorfawi.github.io/lshashing/) 
[![Downloads](https://static.pepy.tech/personalized-badge/lshashing?period=total&units=international_system&left_color=grey&right_color=yellowgreen&left_text=Downloads)](https://pepy.tech/project/lshashing)

python library to perform Locality-Sensitive Hashing to search for nearest neighbors in high dimensional data.

For now it only supports **random projections** but future versions will support more methods and techniques.

### Implementation

```bash
pip install lshashing
```
Make sure the data and query points are numpy arrays!.

```python
from lshashing import LSHRandom
import numpy as np

sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
point = np.random.randint(size = (1,20), low = 0, high = 100)

lshashing = LSHRandom(sample_data, hash_len = 4, num_tables = 2)

print(lshashing.tables[0].hash_table)
# {25: [0, 1, 6, 11, 13, 15], 
#  0: [2, 5], 
#  1: [3, 7, 16, 17, 18, 19], 
#  16: [4, 8, 14], 
# 144: [9], 
#  9: [10], 
#  81: [12]}

print(lshashing.knn_search(sample_data, point[0], k = 4, buckets = 3, radius = 2))
# [[150.33961554   2.        ]
#  [151.30432909   5.        ]
#  [155.11608556   3.        ]
#  [166.76030703  18.        ]]
 ```
First column is the distances while the second column is the indices of the neighbors.

lshashing also supports **parallelism** using **joblib** library.

```python
sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
point = np.random.randint(size = (1, 20), low = 0, high = 100)

lsh_random_parallel = LSHRandom(sample_data, 4, parallel = True)
lsh_random_parallel.knn_search(sample_data, point[0], 4, 3, parallel = True)
# [Neighbor(index=1, distance=152.6106156202772, value=[[47 51 23]...]),
#  Neighbor(index=16, distance=168.08331267558955, value=[[55 61 83]...]),
#  Neighbor(index=14, distance=171.8254928699464, value=[[98 43 81]...]),
#  Neighbor(index=7, distance=183.15294155431957, value=[[75 39 27]...])]
```


### Adding new entries
Simply you can add new entries to the hash tables using the **add_new_entry** method.

```python
from lshashing import LSHRandom
import numpy as np

sample_data = np.random.randint(size = (15, 20), low = 0, high = 100)
point = np.random.randint(size = (1, 20), low = 0, high = 100)

lshashing = LSHRandom(sample_data, hash_len = 3, num_tables = 2)
print(lshashing.tables[0].hash_table)
# {9: [0, 3, 9, 11], 36: [1, 4, 5, 10, 12, 13], 4: [2, 6, 7, 8], 49: [14]}

print(lshashing.n_rows)
# 15

lshashing.add_new_entry(point)

print(lshashing.n_rows)
# 16

print(lshashing.tables[0].hash_table)
# {9: [0, 3, 9, 11], 36: [1, 4, 5, 10, 12, 13], 4: [2, 6, 7, 8, 15], 49: [14]}
```

Locality-sensitive hashing is an **approximate nearest neighbors search technique** which means that the resulted neighbors may not always be the exact nearest neighbor to the query point.
To enhance and ensure better extactness, hash length used, number of hash tables and the buckets to search need to be tweaked. 

I also made some comparison between **lshashing**, linear method to get KNNs and **scikit-learn's BallTree and KDTree** and here are the results.

```bash
python examples/lshashing_compare.py

#                 ##### LSHashing Module #####
# sample data shape:  (15000, 30000)

# query point
# (30000,)

#         ##### Start comparison in searching for 5 nearest neighbors #####

# ##### search knn traditionaly
# time to search: 5.72 seconds

# [(69872.09976807624, 1844),
#  (69876.11714312695, 7288),
#  (69876.55885202132, 2554),
#  (69939.38269673246, 9227),
#  (69939.697440009, 5948)]


# ##### Search with lshashing package:
# time to construct 2 lsh tables of 15 hash length: 9.10 seconds
# time to search in 10 buckets with radius 5: 0.37 seconds

#          distances        indices
# array([[69939.38269673,  9227.        ],
#        [70032.03792551,  9782.        ],
#        [70094.87232316,  6840.        ],
#        [70117.99188511,  8432.        ],
#        [70137.69944901,   331.        ]])


# ##### Now with Scikit Learn
# time to construct ball_tree: 40.51 seconds
# time to search: 0.49 seconds

# (array([[69872.09976808, 69876.11714313, 69876.55885202, 69939.38269673,
#         69939.69744001]]),
#  array([[1844, 7288, 2554, 9227, 5948]]))


# ##### With sklearn KDTree
# time to construct the tree: 51.69 seconds
# time to search: 0.65 seconds

# (array([[69872.09976808, 69876.11714313, 69876.55885202, 69939.38269673,
#         69939.69744001]]),
#  array([[1844, 7288, 2554, 9227, 5948]]))


# ##### basic scikit-learn
# time to fit dataset: 0.24 seconds
# time to search: 4.36 seconds

# (array([[69872.09976808, 69876.11714313, 69876.55885202, 69939.38269673,
#         69939.69744001]]),
#  array([[1844, 7288, 2554, 9227, 5948]]))
 
```

LSHashing performs the search a little bit slower than sklearn tree implementations, sometimes better but much faster to construct. However, the main advantage comes when we need to add new entry or remove from our data, i.e. updating the table. In sklearn trees this can be hard as we may need to reconstruct the trees all over again. It is clearly obvious that it takes much more time to construct the trees than creating the buckets with LSHashing. LSHashing also allows addition of new data easily and in no time.

Also as we can see the nearest neighbors returned by the LSHashing are not the exact neighbors, that's why it is called approximate nearest neighbor search. Of course, when dealing with reasonable amount of data it is better to go with normal nearest neighbor searching. However with very big data, this will be time consuming so it is more efficient to approach it differently.
