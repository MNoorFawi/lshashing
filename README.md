# lshashing

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://mnoorfawi.github.io/lshashing/) 
[![Build Status](https://travis-ci.com/MNoorFawi/lshashing.svg?branch=master)](https://travis-ci.com/MNoorFawi/lshashing)
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
#{225: [0, 12],
# 121: [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 15, 16, 19],
# 81: [6, 11],
# 196: [14],
# 100: [17, 18]}

print(lshashing.knn_search(sample_data, point[0], k = 4, buckets = 3, radius = 2))
#[Neighbor(index=7, distance=159.8217757378512, value=[[78 35 94]...]),
# Neighbor(index=13, distance=174.1551032843999, value=[[86 48 32]...]),
# Neighbor(index=19, distance=174.5737666432159, value=[[53 52 22]...]),
# Neighbor(index=16, distance=180.87564789103038, value=[[81 91 70]...])]
```

lshashing also supports **parallelism** using **joblib** library.

```python
sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
point = np.random.randint(size = (1, 20), low = 0, high = 100)

lsh_random_parallel = LSHRandom(sample_data, 4, parallel = True)
lsh_random_parallel.knn_search(sample_data, point[0], 4, 3, parallel = True)
# [Neighbor(index=7, distance=137.0729732660673, value=[[76 16 41]...]),
#  Neighbor(index=1, distance=163.25133996387288, value=[[81 55 41]...]),
#  Neighbor(index=4, distance=172.41519654601214, value=[[33 21  0]...]),
#  Neighbor(index=8, distance=183.0327839486686, value=[[70 27 85]...])]
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
# {4: [0, 1, 3, 6], 49: [2, 4, 8, 11, 14], 36: [5, 9, 12], 9: [7, 10, 13]}

print(lshashing.n_rows)
# 15

lshashing.add_new_entry(point)

print(lshashing.n_rows)
# 16

print(lshashing.tables[0].hash_table)
# {4: [0, 1, 3, 6, 15], 49: [2, 4, 8, 11, 14], 36: [5, 9, 12], 9: [7, 10, 13]}
```

Locality-sensitive hashing is an **approximate nearest neighbors search technique** which means that the resulted neighbors may not always be the exact nearest neighbor to the query point.
To enhance and ensure better extactness, hash length used, number of hash tables and the buckets to search need to be tweaked. 

I also made some comparison between **lshashing**, linear method to get KNNs and **scikit-learn's BallTree and KDTree** and here are the results.

```bash
python examples/lshashing_compare.py
 
# lshashing module
# Sample data shape:  (500000, 1000)

# query point
# (1000,)

# Start comparison in searching for 4 NNs
# ##### search knn traditionaly
# time to search:  2.4206862449645996

# ##### Search with lshashing package:
# time to construct lsh:  8.619176864624023
# time to search:  0.06634998321533203

# ##### Now with Scikit Learn
# time to construct ball_tree:  102.57923364639282
# time to search:  0.6436479091644287

# ##### With sklearn KDTree
# time to construct the tree:  127.4918167591095
# time to search:  0.763181209564209

# ##### basic scikit-learn
# time to fit dataset:  0.2458806037902832
# time to search:  6.279686450958252
```

LSHashing performs a little bit slower than sklearn tree implementations, sometimes better. However, the main advantage comes when we need to add new entry or remove from our data, i.e. updating the table. In sklearn trees this can be hard as we may need to reconstruct the trees all over again. It is clearly obvious that it takes much more time to construct the trees than creating the buckets with LSHashing. LSHashing also allows addition of new data easily and in no time.
