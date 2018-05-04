import numpy as np
import scipy.linalg as LA
from math import log
import matplotlib.pyplot as plt

def PCA(X, k):
    """ Returns X_hat, which is the nxk approximation of X"""
    u, s, vt = np.linalg.svd(X)

    V_k = (vt[:k, :]).T
    X_hat = X @ V_k
    return X_hat

def PCA_Projector(X, k):
    """ Returns V_k, which is top k principal components of X"""
    u, s, vt = np.linalg.svd(X)

    V_k = (vt[:k, :]).T
    return V_k
def CCA(X, Y, k):
    """ Returns X_hat, Y_hat, which are nxk """
    #Assumes  k < d, k < p 
    d = X.shape[1]
    p = Y.shape[1]

    # U_k is d x k, V_k is p x k
    U_k = np.vstack((np.eye(k), np.zeros((d-k, k))))
    if p > k:
        V_k = np.vstack((np.eye(k), np.zeros((p-k, k))))
    if p == k: 
        V_k = np.eye(k)
    if p < k:
        V_k = np.hstack((np.eye(p), np.zeros((p, k-p))))

    u, s, vt = np.linalg.svd(X.T @ X) #EVD
    # print(X.T @ X)
    # print(np.diag(s))

    W_x = u @ LA.sqrtm(np.diag(s)) @ u.T

    u, s, vt = np.linalg.svd(Y.T @ Y) #EVD
    W_y = u @ LA.sqrtm(np.diag(s)) @ u.T

    u, s, vt = np.linalg.svd(W_x.T @ X.T @ Y @ W_y)
    D_x = u
    D_y = vt.T

    U = W_x @ D_x @ U_k
    V = W_y @ D_y @ V_k

    X_hat = X @ U
    Y_hat = Y @ V

    return X_hat, Y_hat

def CCA_Projector(X, Y, k):
    d = X.shape[1]
    p = Y.shape[1]

    # U_k is d x k, V_k is p x k
    U_k = np.vstack((np.eye(k), np.zeros((d-k, k))))
    if p > k:
        V_k = np.vstack((np.eye(k), np.zeros((p-k, k))))
    if p == k: 
        V_k = np.eye(k)
    if p < k:
        V_k = np.hstack((np.eye(p), np.zeros((p, k-p))))

    u, s, vt = np.linalg.svd(X.T @ X) #EVD
    # print(X.T @ X)
    # print(np.diag(s))

    W_x = u @ LA.sqrtm(np.diag(s)) @ u.T

    u, s, vt = np.linalg.svd(Y.T @ Y) #EVD
    W_y = u @ LA.sqrtm(np.diag(s)) @ u.T

    u, s, vt = np.linalg.svd(W_x.T @ X.T @ Y @ W_y)
    D_x = u
    D_y = vt.T

    U = W_x @ D_x @ U_k
    V = W_y @ D_y @ V_k

    return U 

def PCA_visualize(X, y):
    # Visualizes PCA in 2D with scatterplot
    X_hat = PCA(X, k=2)
    plt.scatter(X_hat[:,0], X_hat[:,1], c = y.reshape(-1))
    plt.show()

def CCA_visualize(X, Y):
    # Visualizes CCA in 2D with scatterplot
    X_hat, Y_hat = CCA(X, Y, k=2)
    plt.scatter(X_hat[:,0], X_hat[:,1], c = "xkcd:magenta")
    plt.scatter(Y_hat[:,0], Y_hat[:,1], c = "xkcd:teal")
    plt.show()

def main():
    #PCA_visualize(np.random.rand(20, 5)*10)
    #CCA_visualize(np.random.rand(20, 5)*10, np.random.rand(20, 3)*5)
    path = "email_vector.npz"
    with np.load(path) as data:
        X = data['arr_0']
        y = data['arr_1']
    print(X.shape)
    PCA_visualize(X, y)
    CCA_visualize(X, y)
    ##Simin
    # word_sophistication = False
    # cap_word_count = False
    # extra_cap = False
    # list_count = False

    # ##Chris
    # misspelled_word = False
    # longest_conseq_cap = False
    # HTML_check = False
    # style_check = False

    # ##Wiliam
    # image_count = False
    # link_count = False
    # number_count = False

    # ##Patrick
    # count_non_English = False
    # word_count = False
    # longest_conseq_char = False
    # character_count = False

    # output = []
    # if word_sophistication:
    #     pass
    # if cap_word_count:
    #     pass
    # if extra_cap:
    #     pass
    # if list_count:
    #     pass
    # if misspelled_word:
    #     pass
    # if longest_conseq_cap:
    #     pass
    # if HTML_check:
    #     pass
    # if style_check:
    #     pass
    # if image_count:
    #     pass
    # if link_count:
    #     pass
    # if number_count:
    #     pass
    # if count_non_English:
    #     pass
    # if word_count:
    #     pass
    # if longest_conseq_char:
    #     pass
    # if character_count:
    #     pass
    # output_array = np.array(output)

if __name__ == '__main__':
    main()

