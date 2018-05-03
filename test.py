import numpy as np
def test_numpy():
    path = "ham_easy.npz"
    with np.load(path) as data:
        print(data['arr_0'].shape)

test_numpy()
