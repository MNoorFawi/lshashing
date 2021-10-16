from .util import generate_rand_proj, hash_fun
from .hash_util import *

class HashTable:
    def __init__(self, hash_len, dim):
        self.hash_table = {}
        self.hash_len = hash_len
        self.rand_proj = generate_rand_proj(self.hash_len, dim)

    def build_table(self, data):
        data_hashes = hash_fun(data, self.rand_proj)
        return fill_table(self.hash_table, data_hashes)

    def add_new_entry(self, new_entry, n_rows):
        new_hash = hash_fun(new_entry, self.rand_proj)
        return fill_table(self.hash_table, new_hash, n_rows)

    def _knn_search(self, query_point, k, buckets,
                   radius = 2):
        if buckets > self.hash_len:
            buckets = self.hash_len - 1
        new_hash = hash_fun(query_point, self.rand_proj)
        candidates = set()
        for h in new_hash:
            if h in self.hash_table:
                candidates.update(self.hash_table[h])

            near_candidates = search_near_hashes(self.hash_table, h,
                                                 buckets, len(candidates), k, radius)
            candidates.update(near_candidates)
        return candidates

    #def __repr__(self):
    #    return self.hash_table

