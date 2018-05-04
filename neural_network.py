import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



class NN_Classifier():

	trained = False

	def __init__(self):
		self.mlp_predictor = MLPClassifier()

	def train(self, X_train, y_train):
		self.scaler = StandardScaler()  
		self.scaler.fit(X_train)  
		X_train = self.scaler.transform(X_train)  
		self.mlp_predictor.fit(X_train, y_train)
		self.trained = True
		# return train accuracy
		return self.mlp_predictor.score(X_train, y_train)

	def predict(self, X_test):
		assert self.trained,"You should call train first."
		X_test = self.scaler.transform(X_test)
		return self.mlp_predictor.predict(X_test)

	def plot(self, X_train, y_train, X_test, y_test):
		
		alphas = np.logspace(-6, 3, 10)
		accuracy = []
		
		for i in alphas:	
			classifier = MLPClassifier(alpha = i, max_iter = 300)
			scaler = StandardScaler()  
			scaler.fit(X_train)  
			X_train = scaler.transform(X_train)  
			classifier.fit(X_train, y_train)
			X_test = scaler.transform(X_test)
			accuracy.append(classifier.score(X_test, y_test))

		plt.plot(np.log10(alphas), accuracy)
		plt.xlabel("alpha in log10")
		plt.ylabel("accuracy")
		plt.title("regularization comparison in neural_network")
		plt.savefig("neural_network_analysis.png")
		plt.show()




	
		



