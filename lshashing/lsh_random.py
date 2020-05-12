from .hash_table import HashTable
from .pq import DistHeap
from .util import NNeighbor, euclidean_dist

class LSHRandom:
    def __init__(self, data, hash_len, num_tables = 2):
        self.n_rows, self.dims = data.shape
        self.hash_len = hash_len
        self.tables = []
        for i in range(num_tables):
            t = HashTable(hash_len, self.dims)
            self.tables.append(t)
            self.tables[i].build_table(data)

    def add_new_entry(self, data_point):
        for t in self.tables:
            t.add_new_entry(data_point, self.n_rows)

    def __get_distances(self, candidate, candidate_point, query_point, dist_func):
        return NNeighbor(candidate, dist_func(candidate_point, query_point), candidate_point)

    def knn_search(self, data, query_point, k, buckets, dist_func = euclidean_dist, radius = 2):
        best_candidates = set()
        for t in self.tables:
            tbc = t.knn_search(query_point, k, buckets, radius)
            best_candidates.update(tbc)
        nn_heap = DistHeap()
        for c in best_candidates:
            candidate_point = data[c, :]
            nn = self.__get_distances(c, candidate_point, query_point, dist_func)
            nn_heap.push(nn)
        return nn_heap[:k]


