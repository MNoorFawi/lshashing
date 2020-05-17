from lshashing import LSHRandom
import numpy as np

def test_lshashing():
    sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
    point = np.random.randint(size = (1, 20), low = 0, high = 100)
    k = 4
    ntables = 2
    lsh_random = LSHRandom(sample_data, 4, num_tables = 2)
    assert len(lsh_random.tables) == ntables
    nns = lsh_random.knn_search(sample_data, point[0], k, 3)
    assert len(nns) == k

def test_parallel():
    sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
    point = np.random.randint(size = (1, 20), low = 0, high = 100)
    k = 4
    lsh_random = LSHRandom(sample_data, 4, parallel = True)
    tbl0_elem = [elem for bucket in lsh_random.tables[0].hash_table.values() for elem in bucket]
    assert len(tbl0_elem) == len(sample_data)
    nns = lsh_random.knn_search(sample_data, point[0], 4, 3, parallel = True)
    assert len(nns) == k
