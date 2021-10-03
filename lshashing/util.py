from collections import namedtuple
import numpy as np
from itertools import combinations
from pyxdist import *
#import math

NN = namedtuple("NN", "index distance value")

class NNeighbor:
    def __init__(self, *args):
        self.nn = NN(*args)
        self.index = self.nn.index
        self.distance = self.nn.distance
        self.value = self.nn.value

    def __getitem__(self, item):
        return self.nn[item]

    def __lt__(self, other):
        return self.nn.distance < other.nn.distance

    def __repr__(self):
        return f"Neighbor(index={self.nn.index}, distance={self.nn.distance}, value=[{self.nn.value[:3]}...])"

def generate_rand_proj(hash_len, dim):
    return np.random.randn(dim, hash_len)

def unbinarize(binary):
    y = "".join((binary * 1).astype(str))
    return int(y, 2)

def hash_fun(data, rand_proj):
    binaries = np.dot(data, rand_proj) > 0
    return [unbinarize(binary) ** 2 for binary in binaries]

def all_combs(hash_len, radius):
    return list(combinations(range(hash_len), radius))

def visited(visited_bins, key):
    if key in visited_bins:
        return True
    else:
        visited_bins.update([key])
        return False

def euclidean_dist(a, b):
    return euclidean_dist_pyx(a, b)
    #return np.sum(np.array([(x - y) ** 2 for x, y in zip(a, b)]))

def get_distances(candidate, candidate_point, query_point, dist_func):
    return NNeighbor(candidate, dist_func(candidate_point, query_point), candidate_point)
    
def nn_search(data, new_point, k, cands):
    return knn_search_pyx(data, new_point, k, cands)
