from joblib import Parallel, delayed
from .hash_table import HashTable
from .pq import DistHeap
from .util import NNeighbor, euclidean_dist, get_distances

def parallel_fill_table(data, dims, hash_len, num_tables):
    def _hash_build(data, hash_len, table, dims):
        t = HashTable(hash_len, dims)
        t.build_table(data)
        return t
    tables = Parallel(n_jobs = 2, max_nbytes = None)(delayed(
        _hash_build)(data, hash_len, t, dims) for t in range(num_tables))
    return tables

def parallel_knn_search(tables, data, query_point, k, buckets, dist_func, radius):
    def tables_search(tables, query_point, k, buckets, radius):
        tables_cands = Parallel(n_jobs = 2, max_nbytes = None)(
            delayed(t._knn_search)(query_point, k, buckets, radius) for t in tables)
        return tables_cands
    def cand_distances(cands, data, query_point, dist_func):
        nns = Parallel(n_jobs = 2, max_nbytes = None)(
            delayed(get_distances)(c, data[c, :], query_point, dist_func) for c in cands)
        return nns
    candidates = tables_search(tables, query_point, k, buckets, radius)
    cand_set = set()
    for cand in candidates:
        cand_set.update(cand)
    nns = cand_distances(cand_set, data, query_point, dist_func)
    nn_heap = DistHeap()
    nn_heap.pushitems(nns)
    return nn_heap#[:k]
