from lshashing import LSHRandom
import math
from pprint import pprint

### Measure searchance
print("\t\t##### LSHashing Module #####")
def euclidean(v1, v2):
    d = ((v1 - v2) ** 2).sum()
    return math.sqrt(d)

def get_knn_naive(points, query_point, k):
    answer = []
    for i in range(points.shape[0]):
        ip = points[i, :]
        x = (euclidean(ip, query_point), i)
        answer.append(x)
    return sorted(answer, key = lambda x: x[0])[:k]

import numpy as np
sample_data = np.random.randint(size = (500000, 1000), low = 0, high = 100)
print("sample data shape: ", sample_data.shape, "\n")
point = np.random.randint(size = (1, 1000), low = 0, high = 100)[0]
print("query point")
print(point.shape, "\n")
from time import time
print("\t##### Start comparison in searching for 4 nearest neighbors #####\n")
print("##### search knn traditionaly")
start = time()
k = 4
naive_knn = get_knn_naive(sample_data, point, k)
print("time to search: %.2f seconds\n" % round(time() - start, 2))
pprint(naive_knn)
print("\n")

print("##### Search with lshashing package:")
start = time()
hash_len = 15
num_tables = 1
lsh_random = LSHRandom(sample_data, hash_len, num_tables = num_tables)
print("time to construct %d lsh tables of %d hash length: %.2f seconds" % (num_tables, hash_len, round(time() - start, 2)))
radius = 1
buckets = 3
start = time()
nns = lsh_random.knn_search(sample_data, point, k, buckets, radius = radius)
print("time to search in %d buckets with radius %d: %.2f seconds\n" % (buckets, radius, round(time() - start, 2)))
print("         distances        indices")
pprint(nns)
print("\n")

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
nbrs = NearestNeighbors(n_neighbors = k, algorithm = "ball_tree").fit(sample_data)
print("time to construct ball_tree: %.2f seconds" % round(time() - start, 2))
start = time()
sklearn_knn = nbrs.kneighbors([point])
print("time to search: %.2f seconds\n" % round(time() - start, 2))
pprint(sklearn_knn)
print("\n")

print("##### With sklearn KDTree")
start = time()
kdt = kdt(sample_data, metric = "euclidean")
print("time to construct the tree: %.2f seconds" % round(time() - start, 2))
start = time()
kdt_knn = kdt.query([point], k = k)
print("time to search: %.2f seconds\n" % round(time() - start, 2))
pprint(kdt_knn)
print("\n")

print("##### basic scikit-learn")
start = time()
nbrs = NearestNeighbors(n_neighbors=k).fit(sample_data)
print("time to fit dataset: %.2f seconds" % round(time() - start, 2))
start = time()
sklearn_knn = nbrs.kneighbors([point])
print("time to search: %.2f seconds\n" % round(time() - start, 2))
pprint(sklearn_knn)
print("\n")
