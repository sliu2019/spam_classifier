import sklearn.svm
class SVM_Classifier():
	def __init__(self, C):
		self.C = C

	def train(self, X, y):
		y = y * 2 - 1
		self.svm_predictor = sklearn.svm.LinearSVC(C = self.C)
		self.svm_predictor.fit(X, y)

	def predict(self, X):
		return self.svm_predictor.predict(X)
