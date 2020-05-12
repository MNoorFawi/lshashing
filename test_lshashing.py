from lshashing import LSHRandom
import numpy as np

def test_lshashing():
    sample_data = np.random.randint(size = (20, 20), low = 0, high = 100)
    point = np.random.randint(size = (1, 20), low = 0, high = 100)

    lsh_random = LSHRandom(sample_data, 4)
    assert lsh_random.knn_search(sample_data, point[0], 4, 3)
