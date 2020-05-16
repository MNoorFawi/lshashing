from .util import all_combs, visited
from random import shuffle, randint

def fill_table(dct, hashes, n_rows = 0):
    indices = list(range(n_rows, n_rows + len(hashes)))
    for i, h in zip(indices, hashes):
        if h in dct:
            dct[h].append(i)
        else:
            dct[h] = [i]

#def bincount(n):
#    return bin(n).count("1")

def bit_glue(bit, hash_len):
    bit_len = len(bin(bit)[2:])
    z = str(0b0) * (hash_len - bit_len)
    nb = list(z + bin(bit)[2:])
    return nb

def bit_mutate(combs, bit, hash_len):
    nb = bit_glue(bit, hash_len)
    shuffle(combs)
    if len(combs) == 0:
        combs.append((randint(0, hash_len)))
    mutation = combs.pop()
    if isinstance(mutation, int):
        if (len(nb) - 1) < mutation:
            mutation = len(nb) - 1
        nb[mutation] = "0" if nb[mutation] == "1" else "1"
    else:
        for i in mutation:
            nb[i] = "0" if nb[i] == "1" else "1"
    return int("".join(nb), 2)

def search_near_hashes(table, new_bin, bins_to_search, k_searched, k_to_search, radius):
    candidates = set()
    visited_bins = {new_bin}
    hash_len = max(table.keys()).bit_length()
    combs = all_combs(hash_len, radius)
    #tbl_keys = np.array(list(table.keys()))
    #bin_dist = np.array([bincount(new_bin ^ i) for i in tbl_keys])
    #bins = tbl_keys[np.where(bin_dist <= radius)]
    while k_searched < k_to_search or len(visited_bins) < bins_to_search:
        new_bin = bit_mutate(combs, new_bin, hash_len)
        new_bin = new_bin if new_bin in table else min(table.keys(), key = lambda x: abs(x - new_bin))
        if not visited(visited_bins, new_bin):
            candidates.update(table[new_bin])
            k_searched = k_searched + len(candidates)
    return candidates

    