import numpy as np
from math import log

class LDA_Classifier(object):

	def __init__(self):
		self.epsilon = 0.00001 #padding for log

	def train(self, X_train, Y_train):

	#Train the classifier. Compute the empirical covariances and means for each class, and the class priors.

		d = X_train.shape[1]
		Y_train = np.reshape(Y_train, (Y_train.size))

		self.class_ids, counts = np.unique(Y_train, return_counts = True)
		self.class_means = {}
		self.class_priors = {}
		self.cov = np.zeros((d, d))
		
		for i in range(self.class_ids.size):
			class_id = self.class_ids[i]

			class_data = X_train[Y_train == class_id, :]
			class_mean = np.reshape(np.mean(class_data, axis=0), (1, d))
			class_cov = (1.0/class_data.shape[0])*(class_data - class_mean).T @ (class_data - class_mean)
			#print(class_cov)
			self.class_means[class_id] = class_mean
			self.class_priors[class_id] = counts[i]/np.sum(counts)
			self.cov = self.cov + class_cov*self.class_priors[class_id]
		return None

	def predict(self, X_test):
	#Predict labels for test data using this classifier.

		n_test = X_test.shape[0]
		d = X_test.shape[1]

		scores = np.empty((n_test, 1))
		for i in range(self.class_ids.size):
			class_id = self.class_ids[i]
			mu = self.class_means[class_id]
			prior = self.class_priors[class_id]

			score = (-1.0/2)*np.diag((X_test - mu) @ np.linalg.inv(self.cov + np.eye(self.cov.shape[0])*self.epsilon) @ (X_test -  mu).T) + log(prior)
			score = np.reshape(score, (n_test, 1))

			if i == 0:
				scores = scores + score
			else:
				scores = np.hstack((scores, score))

		Y_hat = self.class_ids[np.argmax(scores, axis = 1)]
		return Y_hat

def main():
	pass
	# lda_classifier = LDA_Classifier()

	# X_train = np.random.rand(20, 30)*10
	# y_train = np.random.randint(0, 2, (20, 1))
	# X_test = np.random.rand(10, 30)*10

	# lda_classifier.train(X_train, y_train)
	# y_hat = lda_classifier.predict(X_test)
	# print(y_hat)

if __name__ == "__main__":
	main()