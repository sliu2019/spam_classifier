class LDA(object):

	def __init__(self):
		pass

	def train(self, X_train, Y_train):
	"""
	Train the classifier. Compute the empirical covariances and means for each class, and the class priors.
	"""
		d = X_train.shape[1]

		class_ids, counts = np.unique(Y_train, return_counts = True)
		self.class_means = {}
		self.class_priors = {}

		self.cov = np.zeros((d, d))
		for i in range(class_ids.size):
			class_id = class_ids[i]

			class_data = X_train[Y_train == class_id]
			class_mean = np.reshape(np.mean(class_data, axis=0), (1, d))
			class_cov = (1.0/class_data.shape[0])*(X_train - class_mean).T @ (X_train - class_mean)

			self.class_means[class_id] = class_mean
			self.class_priors[class_id] = counts[i]/np.sum(counts)
			self.cov = self.cov + class_cov*class_priors[class_id]
		return None

	def predict(self, X_test):
	"""
	Predict labels for test data using this classifier.
	"""
		n_test = X_test.shape[0]
		Y_hat = np.zeros(n_test)

		for i in range(n_test):
			best_class_id = None
			best_class_value = float('-inf')
			for class_id in class_ids: 
				mu = self.class_means[class_id]
				prior = self.class_priors[class_id]

				x_i = X_test[i, :]
				value = -1.0/2*log(np.linalg.det(self.cov)) - 1.0/2*(x_i - mu).T @ np.linalg.inv(self.cov) @ (x_i - mu) + log(prior)

				if value > best_class_value:
					best_class_value = value
					best_class_id = class_id

			Y_hat[i] = class_id

		return Y_hat

