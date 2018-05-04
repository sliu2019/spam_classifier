import numpy as np
from sklearn import linear_model
def linear_regression(X, y):
    y = 2 * y - 1
    return np.linalg.solve(X.T@X + 0.001 * np.identity(X.shape[1]), X.T@y)
class LS_SVM:
    def train(self, X_training, y_trainig):
        self.w = linear_regression(X_training, y_trainig)

    def predict(self, X_test):
        pred_test = X_test @ self.w
        return (pred_test > 0).astype(int)

def logistic_regression(X, y):
    lr = linear_model.LogisticRegression()
    lr.fit(X, y.reshape(-1))
    return lr 

class logistic_regression_classifier:
	def train(self, X_training, y_trainig):
		self.lr = logistic_regression(X_training, y_trainig)

	def predict(self, X_test):
		return self.lr.predict(X_test).reshape(-1, 1)
