# lshashing

[![Build Status](https://travis-ci.com/MNoorFawi/lshashing.svg?branch=master)](https://travis-ci.com/MNoorFawi/lshashing)

python library to perform Locality-Sensitive Hashing to search for nearest neighbors in high dimensional data.

For now it only supports **random projections** but future versions will support more methods and techniques.

### Implementation

```bash
pip install lshashing
```

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

Locality-sensitive hashing is an **approximate nearest neighbors search technique** which means that the resulted neighbors may not always be the exact nearest neighbor to the query point.
To enhance and ensure better extactness, hash length used, number of hash tables and the buckets to search need to be tweaked. 

I also made some comparison between **lshashing**, linear method to get KNNs and **scikit-learn's BallTree and KDTree** and here are the results.

```bash
python examples/lshashing_compare.py
 
# lshashing module
# Sample data shape:  (10000, 10000)

# query point
# (10000,)

# Start comparison in searching for 4 NNs
# ##### search knn traditionaly
# time to perform:  46.329522132873535

# ##### Search with lshashing package:
# time to construct lsh:  1.1631906032562256
# time to perform:  8.50294828414917

# ##### Now with Scikit Learn
# time to construct ball_tree:  14.541776418685913
# time to perform:  0.1249244213104248

# ##### With sklearn KDTree
# time to construct the tree:  21.14064049720764
# time to perform:  0.1249234676361084
```
