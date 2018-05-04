import numpy as np
from sklearn import linear_model
def linear_regression(X, y):
	y = 2 * y - 1
	return np.linalg.solve(X.T@X + 0.001 * np.identity(X.shape[1]), X.T@y)

def logistic_regression(X, y):
	lr = linear_model.LogisticRegression()
	lr.fit(X, y.reshape(-1))
	return lr 

