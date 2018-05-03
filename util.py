import numpy as np
import scipy.linalg as LA
from math import log

def PCA(X, k):
	""" Returns X_hat, which is the nxk approximation of X"""
	u, s, vt = np.linalg.svd(X)

	V_k = (vt[:k, :]).T
	X_hat = X @ V_k
	return X_hat

def CCA(X, Y, k):
	""" Returns X_hat, Y_hat, which are nxk """
	#Assumes  k < d, k < p 
	d = X.shape[1]
	p = Y.shape[1]

	# U_k is d x k, V_k is p x k
	U_k = np.vstack((np.eye(k), np.zeros((d-k, k))))
	V_k = np.vstack((np.eye(k), np.zeros((p-k, k))))

	u, s, vt = np.linalg.svd(X.T @ X) #EVD
	W_x = u @ LA.sqrtm(s) @ u.T

	u, s, vt = np.linalg.svd(Y.T @ Y) #EVD
	W_y = u @ LA.sqrtm(s) @ u.T

	u, s, vt = np.linalg.svd(W_x.T @ X.T @ Y @ W_y)
	D_x = u
	D_y = vt.T

	U = W_x @ D_x @ U_k
	V = W_y @ D_y @ V_k

	X_hat = X @ U
	Y_hat = Y @ V

	return X_hat, Y_hat

def QDA(X_train, Y_train, X_test):
	""" Discriminative classification method: takes in training data with labels and test data, and outputs Y_hat corr. to X_test """
	d = X_train.shape[1]

	class_ids, counts = np.unique(Y_train, return_counts = True)
	class_means = {}
	class_covs = {}
	class_priors = {}

	for i in range(class_ids.size):
		class_id = class_ids[i]

		class_data = X_train[Y_train == class_id]
		class_mean = np.reshape(np.mean(class_data, axis=0), (1, d))
		class_cov = (1.0/class_data.shape[0])*(X_train - class_mean).T @ (X_train - class_mean)

		class_means[class_id] = class_mean
		class_covs[class_id] = class_cov
		class_priors[class_id] = counts[i]/np.sum(counts)

	n_test = X_test.shape[0]
	Y_hat = np.zeros(n_test)

	for i in range(n_test):
		best_class_id = None
		best_class_value = float('-inf')
		for class_id in class_ids: 
			sigma = class_covs[class_id]
			mu = class_means[class_id]
			prior = class_priors[class_id]

			x_i = X_test[i, :]
			value = -1.0/2*log(np.linalg.det(sigma)) - 1.0/2*(x_i - mu).T @ np.linalg.inv(sigma) @ (x_i - mu) + log(prior)

			if value > best_class_value:
				best_class_value = value
				best_class_id = class_id

		Y_hat[i] = class_id

	return Y_hat

def LDA(X_train, Y_train, X_test):
	""" Discriminative classification method: takes in training data with labels and test data, and outputs Y_hat corr. to X_test """
	d = X_train.shape[1]

	class_ids, counts = np.unique(Y_train, return_counts = True)
	class_means = {}
	#class_covs = {}
	class_priors = {}

	cov = np.zeros((d, d))
	for i in range(class_ids.size):
		class_id = class_ids[i]

		class_data = X_train[Y_train == class_id]
		class_mean = np.reshape(np.mean(class_data, axis=0), (1, d))
		class_cov = (1.0/class_data.shape[0])*(X_train - class_mean).T @ (X_train - class_mean)

		class_means[class_id] = class_mean
		#class_covs[class_id] = class_cov

		class_priors[class_id] = counts[i]/np.sum(counts)
		cov = cov + class_cov*class_priors[class_id]


	n_test = X_test.shape[0]
	Y_hat = np.zeros(n_test)

	for i in range(n_test):
		best_class_id = None
		best_class_value = float('-inf')
		for class_id in class_ids: 
			mu = class_means[class_id]
			prior = class_priors[class_id]

			x_i = X_test[i, :]
			value = -1.0/2*log(np.linalg.det(cov)) - 1.0/2*(x_i - mu).T @ np.linalg.inv(cov) @ (x_i - mu) + log(prior)

			if value > best_class_value:
				best_class_value = value
				best_class_id = class_id

		Y_hat[i] = class_id

	return Y_hat

def test_QDA():
	"""Creates some random matrices to test on"""
	return None