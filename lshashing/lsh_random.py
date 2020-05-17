from .hash_table import HashTable
from .pq import DistHeap
from .util import NNeighbor, euclidean_dist, get_distances
from .para_util import parallel_fill_table, parallel_knn_search

class LSHRandom:
    def __init__(self, data, hash_len, num_tables = 1, parallel = False):
        self.n_rows, self.dims = data.shape
        self.hash_len = hash_len
        #self.tables = []
        if parallel:
            self.tables = parallel_fill_table(data, self.dims, hash_len, num_tables)
            #self.tables.extend(tables)
        else:
            self.tables = []
            for i in range(num_tables):
                t = HashTable(hash_len, self.dims)
                self.tables.append(t)
                self.tables[i].build_table(data)

    def add_new_entry(self, data_point):
        for t in self.tables:
            t.add_new_entry(data_point, self.n_rows)
        self.n_rows += data_point.shape[0]

    def knn_search(self, data, query_point, k, buckets,
                   dist_func = euclidean_dist, radius = 2, parallel = False):
        if parallel:
            nn_heap = parallel_knn_search(self.tables, data, query_point, k, buckets, dist_func, radius)
        else:
            best_candidates = set()
            for t in self.tables:
                tbc = t._knn_search(query_point, k, buckets, radius)
                best_candidates.update(tbc)
            nn_heap = DistHeap()
            for c in best_candidates:
                candidate_point = data[c, :]
                nn = get_distances(c, candidate_point, query_point, dist_func)
                nn_heap.push(nn)
        return nn_heap[:k]



