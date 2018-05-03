import numpy as np
def tfidf():
    with np.load("ham_easy.npz") as data:
        X = data['arr_0']
        y = data['arr_1']
    with np.load("ham_hard.npz") as data:
        X = np.vstack((X, data['arr_0']))
        y = np.vstack((y, data['arr_1']))
    with np.load("spam.npz") as data:
        X = np.vstack((X, data['arr_0']))
        y = np.vstack((y, data['arr_1']))
    X_freq = (X > 0).astype(int)
    idf = np.log(X.shape[0] / np.sum(X_freq, axis = 0)).reshape(X.shape[1])
    X =  np.multiply(X, idf)
    np.savez("email_vector", X, y)
tfidf()