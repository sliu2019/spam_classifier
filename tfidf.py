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

    
    #Randomly sample the training set
    indices = np.random.choice(X.shape[0], 1000, replace = False)
    X_training = X[indices, :]
    y_training = y[indices, :]
    np.savez("email_vector_training", X_training, y_training)
    
    #An ugly way to get the test set
    X_test = np.zeros((1, X.shape[1]))
    y_test = np.zeros((1, y.shape[1]))
    for i in range(X.shape[0]):
        if not i in indices:
            X_test = np.vstack((X_test, X[i, :]))
            y_test = np.vstack((y_test, y[i, :]))
    X_test = X_test[1:, :]
    y_test = y_test[1:, :]
    np.savez("email_vector_test", X_test, y_test)

tfidf()