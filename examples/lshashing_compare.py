from lshashing import LSHRandom
import math

### Measure searchance
print("lshashing module")
def euclidean(v1, v2):
    d = (v1 - v2).sum() ** 2
    return math.sqrt(d)

def get_knn_naive(points, query_point, k):
    answer = []
    for i in points:
        x = (euclidean(i, query_point), i)
        answer.append(x)
    return sorted(answer, key = lambda x: x[0])[:k]

import numpy as np
sample_data = np.random.randint(size = (500000, 1000), low = 0, high = 100)
print("Sample data shape: ", sample_data.shape, "\n")
point = np.random.randint(size = (1, 1000), low = 0, high = 100)[0]
print("query point")
print(point.shape, "\n")
from time import time
print("Start comparison in searching for 4 NNs")
print("##### search knn traditionaly")
start = time()
k = 4
naive_knn = get_knn_naive(sample_data, point, k)
print("time to search: ", time() - start, "\n")

print("##### Search with lshashing package:")
start = time()
lsh_random = LSHRandom(sample_data, 5, num_tables = 1)
print("time to construct lsh: ", time() - start)
start = time()
nns = lsh_random.knn_search(sample_data, point, k, 3, radius = 2)
print("time to search: ", time() - start, "\n")

#print("##### Search with lshashing package in parallel:")
#start = time()
#lsh_random2 = LSHRandom(sample_data, 10, num_tables = 1, parallel = True)
#print("time to construct lsh: ", time() - start)
#start = time()
#nns2 = lsh_random.knn_search(sample_data, point, k, 5, radius = 2, parallel = True)
#print("time to search: ", time() - start, "\n")

print("##### Now with Scikit Learn")
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree as kdt

start = time()
nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree').fit(sample_data)
print("time to construct ball_tree: ", time() - start)
start = time()
distances, indices = nbrs.kneighbors([point])
print("time to search: ", time() - start, "\n")
sklearn_knn = sample_data[indices]

print("##### With sklearn KDTree")
start = time()
kdt = kdt(sample_data, metric='euclidean')
print("time to construct the tree: ", time() - start)
start = time()
nearest_dist, nearest_ind = kdt.query([point], k = k)
print("time to search: ", time() - start, "\n")
kdt_knn = sample_data[nearest_ind[0]]

print("##### basic scikit-learn")
start = time()
nbrs = NearestNeighbors(n_neighbors=k).fit(sample_data)
print("time to fit dataset: ", time() - start)
start = time()
distances, indices = nbrs.kneighbors([point])
print("time to search: ", time() - start, "\n")
sklearn_knn = sample_data[indices]
