from lshashing import LSHRandom
import math

### Measure Performance
print("lshashing module")
def get_knn_naive(points, query_point, k):
    answer = []
    for i in points:
        x = (math.sqrt(sum([(a - b) ** 2 for a, b in zip(i, query_point)])), i)
        answer.append(x)
    return sorted(answer, key = lambda x: x[0])[:k]

import numpy as np
sample_data = np.random.randint(size = (10000, 10000), low = 0, high = 100)
print("Sample data shape: ", sample_data.shape, "\n")
point = np.random.randint(size = (1, 10000), low = 0, high = 100)[0]
print("query point")
print(point.shape, "\n")
from time import time
print("Start comparison in searching for 4 NNs")
print("##### search knn traditionaly")
start = time()
k = 4
naive_knn = get_knn_naive(sample_data, point, k)
print("time to perform: ", time() - start, "\n")

print("##### Search with lshashing package:")
start = time()
lsh_random = LSHRandom(sample_data, 10)
print("time to construct lsh: ", time() - start)
start = time()
nns = lsh_random.knn_search(sample_data, point, k, 4)
print("time to perform: ", time() - start, "\n")

print("##### Now with Scikit Learn")
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree as kdt

start = time()
nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree').fit(sample_data)
print("time to construct ball_tree: ", time() - start)
start = time()
distances, indices = nbrs.kneighbors([point])
print("time to perform: ", time() - start, "\n")
sklearn_knn = sample_data[indices]

print("##### With sklearn KDTree")
start = time()
kdt = kdt(sample_data, metric='euclidean')
print("time to construct the tree: ", time() - start)
start = time()
nearest_dist, nearest_ind = kdt.query([point], k = k)
print("time to perform: ", time() - start, "\n")
kdt_knn = sample_data[nearest_ind[0]]
