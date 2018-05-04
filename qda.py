import numpy as np
from math import log

class QDA_Classifier(object):

	def __init__(self):
		self.mu = 0.00001 #padding for log

	def train(self, X_train, Y_train):
	#Train the classifier. Compute the empirical covariances and means for each class, and the class priors. 
		d = X_train.shape[1]
		n = X_train.shape[0]

		Y_train = np.reshape(Y_train, (Y_train.size))

		self.class_ids, counts = np.unique(Y_train, return_counts = True)
		self.class_means = {}
		self.class_covs = {}
		self.class_priors = {}

		for i in range(self.class_ids.size):
			class_id = self.class_ids[i]

			class_data = X_train[Y_train == class_id, :]
			class_mean = np.reshape(np.mean(class_data, axis=0), (1, d))
			#print(class_data)
			#print(class_mean.shape)
			class_cov = (1.0/class_data.shape[0])*(class_data - class_mean).T @ (class_data - class_mean)
			# print(np.all(np.linalg.eigvals(class_cov) >= 0))
			# print(np.linalg.eigvals(class_cov))
			self.class_means[class_id] = class_mean
			self.class_covs[class_id] = class_cov
			self.class_priors[class_id] = counts[i]/np.sum(counts)
		return None

	def predict(self, X_test):
	# Predict labels for test data using this classifier.
		n_test = X_test.shape[0]
		d = X_test.shape[1]
		Y_hat = np.zeros(n_test)

		for i in range(n_test):
			best_class_id = None
			best_class_value = float('-inf')
			for class_id in self.class_ids: 
				sigma = self.class_covs[class_id]
				mu = self.class_means[class_id]
				prior = self.class_priors[class_id]

				x_i = np.reshape(X_test[i, :], (d, 1))
				value = -1.0/2*log(np.linalg.det(sigma) + self.mu) - 1.0/2*(x_i - mu.T).T @ np.linalg.inv(sigma + 0.0001* np.identity(d)) @ (x_i - mu.T) + log(prior)

				if value > best_class_value:
					best_class_value = value
					best_class_id = class_id
			Y_hat[i] = best_class_id

		return Y_hat

def main():
	qda_classifier = QDA()

	X_train = np.random.rand(20, 30)*10
	y_train = np.random.randint(0, 2, (20, 1))
	X_test = np.random.rand(10, 30)*10

	# qda_classifier.train(X_train, y_train)
	# y_hat = qda_classifier.predict(X_test)

if __name__ == "__main__":
	main()